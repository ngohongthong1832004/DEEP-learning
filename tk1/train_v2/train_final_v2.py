# %% [markdown]
# # Multi-Model Rice Disease Classification
# Train and compare 6 lightweight models (<50M params) optimized for rice leaf disease classification

# %%
# ===== IMPORTS =====
import os, shutil, random, torch, gc, time, sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple
from copy import deepcopy
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import cv2

try:
    from torch.amp import GradScaler, autocast
    _NEW_AMP = True
except:
    from torch.cuda.amp import GradScaler, autocast
    _NEW_AMP = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# %%
# ===== REPRODUCIBILITY =====
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(42)

# %%
# ===== CONFIGURATION =====
CONFIG = {
    # Models to train (all <50M params)
    'models': [
        'mobilenet_v3_small',    # ~2.5M params
        'mobilenet_v3_large',    # ~5.4M params
        'efficientnet_b0',       # ~5.3M params
        'efficientnet_v2_s',     # ~21M params
        'resnet18',              # ~11M params
        # 'shufflenet_v2_x1_0',    # ~2.3M params
    ],
    
    # Training
    'img_size': 224,
    'batch_size': 128,
    'epochs': 5,
    'lr': 2e-4,
    'num_workers': 8,
    'pin_memory': True,
    
    # Model enhancements
    'use_cbam': True,
    'use_better_head': True,
    
    # Optimizations
    'use_weighted_sampler': True,
    'use_sce_loss': True,
    'sce_alpha': 0.1,
    'sce_beta': 1.0,
    'use_ema': True,
    'ema_decay': 0.999,
    'use_mixup': True,
    'mixup_alpha': 0.1,
    
    # CLAHE preprocessing
    'use_clahe': True,
    'clahe_clip_limit': 2.0,
    'clahe_tile_size': (8, 8),
    
    # Early stopping
    'patience': 12,
    
    # Data splits
    'val_size': 0.15,
    'test_size': 0.15,
    
    # Ensemble
    'use_ensemble': True,
}

# %%
# ===== SETUP =====
def get_output_folder(parent_dir: str, env_name: str) -> str:
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = f"{env_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir = os.path.join(parent_dir, experiment_id)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

PATH_OUTPUT = get_output_folder("../output", "multi-model-classifier-v2-final")

def create_output_structure(base_path):
    folders = ["weights", "results", "plots", "logs", "demo", "comparison"]
    for folder in folders:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)
    return {folder: os.path.join(base_path, folder) for folder in folders}

OUTPUT_DIRS = create_output_structure(PATH_OUTPUT)

def setup_logging(output_path):
    log_file = os.path.join(output_path, "logs", "training.log")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging(PATH_OUTPUT)

# %%
# ===== LABELS =====
LABELS = {
    0: {"name": "brown_spot", "match_substrings": [
        "../data_total/brown_spot",
        # "../data/yolo_detected_epoch_40/paddy_disease_train/brown_spot/crops",
        # "../data/yolo_detected_epoch_40/sikhaok_train/BrownSpot/crops",
    ]},
    1: {"name": "leaf_blast", "match_substrings": [
        "../data_total/blast",
        # "../data/yolo_detected_epoch_40/paddy_disease_train/blast/crops",
        # "../data/yolo_detected_epoch_40/sikhaok_train/LeafBlast/crops",
    ]},
    2: {"name": "leaf_blight", "match_substrings": [
        "../data_total/bacterial_leaf_blight",
        # "../data/yolo_detected_epoch_40/paddy_disease_train/bacterial_leaf_blight/crops",
        # "../data/yolo_detected_epoch_40/sikhaok_train/Bacterialblight1/crops",
        # "../data/yolo_detected_epoch_40/trumanrase_train/bacterial_leaf_blight/crops",
    ]},
    3: {"name": "healthy", "match_substrings": [
        "../data_total/normal",
        # "../data/yolo_detected_epoch_40/paddy_disease_train/normal/crops",
        # "../data/yolo_detected_epoch_40/sikhaok_train/Healthy/crops",
        # "../data/raw/paddy_disease_classification/train_images/normal",
    ]},
}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device: {DEVICE}")
if torch.cuda.is_available():
    logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## MODEL COMPONENTS

# %%
# ===== CBAM ATTENTION =====
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, max(in_channels // reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(in_channels // reduction, 8), in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

# ===== ENHANCED HEAD =====
class EnhancedHead(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        hidden = max(256, in_features // 2)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.fc2 = nn.Linear(hidden, num_classes)
    
    def forward(self, x):
        avg_out = self.avg_pool(x).flatten(1)
        max_out = self.max_pool(x).flatten(1)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# ===== MODEL BUILDER =====
def build_classifier(backbone_name: str, num_classes: int):
    """Build classifier with specified backbone"""
    
    # Get backbone and feature channels
    if backbone_name == 'mobilenet_v3_small':
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        base_model = mobilenet_v3_small(weights=weights)
        features = base_model.features
        feat_channels = 576
        
    elif backbone_name == 'mobilenet_v3_large':
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        base_model = mobilenet_v3_large(weights=weights)
        features = base_model.features
        feat_channels = 960
        
    elif backbone_name == 'efficientnet_b0':
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        base_model = efficientnet_b0(weights=weights)
        features = base_model.features
        feat_channels = 1280
        
    elif backbone_name == 'efficientnet_v2_s':
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        base_model = efficientnet_v2_s(weights=weights)
        features = base_model.features
        feat_channels = 1280
        
    elif backbone_name == 'resnet18':
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1
        base_model = resnet18(weights=weights)
        # Remove avgpool and fc
        features = nn.Sequential(*list(base_model.children())[:-2])
        feat_channels = 512
        
    elif backbone_name == 'shufflenet_v2_x1_0':
        from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
        weights = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
        base_model = shufflenet_v2_x1_0(weights=weights)
        # Remove fc
        features = nn.Sequential(*list(base_model.children())[:-2])
        feat_channels = 1024
        
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    # Add CBAM if enabled
    if CONFIG['use_cbam']:
        cbam = CBAM(feat_channels, reduction=16)
        features = nn.Sequential(*list(features.children()), cbam)
    
    # Build model
    model = nn.Module()
    model.features = features
    
    # Add classification head
    if CONFIG['use_better_head']:
        model.head = EnhancedHead(feat_channels, num_classes, dropout=0.3)
    else:
        model.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(feat_channels, num_classes)
        )
    
    def _forward(self, x):
        x = self.features(x)
        return self.head(x)
    
    model.forward = _forward.__get__(model, type(model))
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return model, total_params, trainable_params

# %% [markdown]
# ## DATA COLLECTION

# %%
def collect_images_from_path(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    images = []
    try:
        for file in os.listdir(path):
            if file.endswith(image_extensions):
                images.append(os.path.join(path, file))
    except Exception as e:
        logging.warning(f"Error reading {path}: {e}")
    return images

def auto_collect_dataset():
    logging.info("="*60)
    logging.info("DATA COLLECTION")
    logging.info("="*60)
    
    all_data = []
    for label_id, label_info in LABELS.items():
        label_name = label_info['name']
        match_paths = label_info['match_substrings']
        
        logging.info(f"\nCollecting {label_name} (ID: {label_id})...")
        
        for path in match_paths:
            images = collect_images_from_path(path)
            if len(images) > 0:
                logging.info(f"  ✓ {len(images)} images from {path}")
                for img_path in images:
                    all_data.append({
                        'image_path': img_path,
                        'label_id': label_id,
                        'label_name': label_name,
                    })
    
    df = pd.DataFrame(all_data)
    logging.info(f"\nTotal: {len(df)} images")
    logging.info(f"\nBy label:\n{df.groupby('label_name').size()}")
    
    return df

collected_df = auto_collect_dataset()
collected_df.to_csv(os.path.join(OUTPUT_DIRS["results"], "collected_images.csv"), index=False)

# %%
# ===== DATA VISUALIZATION =====
def visualize_dataset_distribution(df, output_dir):
    """Create comprehensive data visualizations"""
    
    logging.info("\n" + "="*60)
    logging.info("CREATING DATA VISUALIZATIONS")
    logging.info("="*60)
    
    class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Class Distribution Bar Chart
    ax1 = plt.subplot(3, 3, 1)
    class_counts = df.groupby('label_name').size()
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
    bars = ax1.bar(class_counts.index, class_counts.values, color=colors, alpha=0.8)
    ax1.set_title('Dataset Distribution by Class', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Number of Images')
    ax1.set_xlabel('Disease Class')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, class_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Add percentage labels
    total = class_counts.sum()
    for i, (bar, count) in enumerate(zip(bars, class_counts.values)):
        pct = (count / total) * 100
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'{pct:.1f}%', ha='center', va='center', fontweight='bold', color='white')
    
    # 2. Pie Chart
    ax2 = plt.subplot(3, 3, 2)
    wedges, texts, autotexts = ax2.pie(class_counts.values, labels=class_counts.index, 
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Class Distribution (Pie Chart)', fontweight='bold', fontsize=12)
    
    # 3. Data Source Distribution
    ax3 = plt.subplot(3, 3, 3)
    source_counts = defaultdict(int)
    for _, row in df.iterrows():
        path = row['image_path']
        if 'paddy_disease_train' in path:
            source_counts['Paddy Disease'] += 1
        elif 'sikhaok_train' in path:
            source_counts['Sikhaok'] += 1
        elif './data/' in path:
            source_counts['Original Data'] += 1
        else:
            source_counts['Other'] += 1
    
    source_df = pd.Series(source_counts)
    colors_source = plt.cm.Pastel1(np.linspace(0, 1, len(source_df)))
    bars = ax3.bar(source_df.index, source_df.values, color=colors_source, alpha=0.8)
    ax3.set_title('Distribution by Data Source', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Number of Images')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    for bar, count in zip(bars, source_df.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # 4. Class Distribution by Source (Stacked Bar)
    ax4 = plt.subplot(3, 3, 4)
    source_class_data = []
    for source in source_df.index:
        source_class_counts = []
        for class_name in class_names:
            count = 0
            class_df = df[df['label_name'] == class_name]
            for _, row in class_df.iterrows():
                path = row['image_path']
                if source == 'Paddy Disease' and 'paddy_disease_train' in path:
                    count += 1
                elif source == 'Sikhaok' and 'sikhaok_train' in path:
                    count += 1
                elif source == 'Original Data' and './data/' in path:
                    count += 1
                elif source == 'Other' and not any(x in path for x in ['paddy_disease_train', 'sikhaok_train', './data/']):
                    count += 1
            source_class_counts.append(count)
        source_class_data.append(source_class_counts)
    
    # Create stacked bar chart
    bottom = np.zeros(len(class_names))
    for i, (source, counts) in enumerate(zip(source_df.index, source_class_data)):
        ax4.bar(class_names, counts, bottom=bottom, label=source, 
               color=colors_source[i], alpha=0.8)
        bottom += counts
    
    ax4.set_title('Class Distribution by Data Source', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Number of Images')
    ax4.legend()
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    # 5. Histogram of class imbalance
    ax5 = plt.subplot(3, 3, 5)
    class_counts_values = class_counts.values
    ax5.hist(class_counts_values, bins=10, color='skyblue', alpha=0.7, edgecolor='black')
    ax5.axvline(np.mean(class_counts_values), color='red', linestyle='--', 
               label=f'Mean: {np.mean(class_counts_values):.0f}')
    ax5.axvline(np.median(class_counts_values), color='green', linestyle='--', 
               label=f'Median: {np.median(class_counts_values):.0f}')
    ax5.set_title('Distribution of Class Sizes', fontweight='bold', fontsize=12)
    ax5.set_xlabel('Number of Images per Class')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    
    # 6. Class Imbalance Ratio
    ax6 = plt.subplot(3, 3, 6)
    max_count = class_counts.max()
    imbalance_ratios = max_count / class_counts.values
    bars = ax6.bar(class_names, imbalance_ratios, color='coral', alpha=0.8)
    ax6.set_title('Class Imbalance Ratio', fontweight='bold', fontsize=12)
    ax6.set_ylabel('Imbalance Ratio (Max/Current)')
    plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
    ax6.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Perfect Balance')
    ax6.legend()
    
    for bar, ratio in zip(bars, imbalance_ratios):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{ratio:.2f}x', ha='center', va='bottom', fontweight='bold')
    
    # 7. Cumulative Distribution
    ax7 = plt.subplot(3, 3, 7)
    sorted_counts = np.sort(class_counts.values)
    cumulative = np.cumsum(sorted_counts)
    cumulative_pct = (cumulative / total) * 100
    ax7.plot(range(1, len(sorted_counts)+1), cumulative_pct, 'bo-', linewidth=2, markersize=8)
    ax7.set_title('Cumulative Class Distribution', fontweight='bold', fontsize=12)
    ax7.set_xlabel('Class Rank (by size)')
    ax7.set_ylabel('Cumulative Percentage (%)')
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim(0, 100)
    
    # 8. Sample Images Grid (if we have access to images)
    ax8 = plt.subplot(3, 3, 8)
    try:
        # Sample one image from each class for preview
        sample_images = []
        sample_labels = []
        
        for class_name in class_names:
            class_data = df[df['label_name'] == class_name]
            if len(class_data) > 0:
                sample_path = class_data.iloc[0]['image_path']
                if os.path.exists(sample_path):
                    try:
                        img = Image.open(sample_path).convert('RGB')
                        img = img.resize((64, 64))  # Small size for grid
                        sample_images.append(np.array(img))
                        sample_labels.append(class_name)
                    except:
                        continue
        
        if sample_images:
            # Create a grid of sample images
            grid_size = int(np.ceil(np.sqrt(len(sample_images))))
            grid_img = np.zeros((grid_size * 64, grid_size * 64, 3), dtype=np.uint8)
            
            for i, img in enumerate(sample_images):
                row = i // grid_size
                col = i % grid_size
                grid_img[row*64:(row+1)*64, col*64:(col+1)*64] = img
            
            ax8.imshow(grid_img)
            ax8.set_title('Sample Images by Class', fontweight='bold', fontsize=12)
            ax8.axis('off')
        else:
            ax8.text(0.5, 0.5, 'No sample\\nimages available', 
                    ha='center', va='center', transform=ax8.transAxes, fontsize=12)
            ax8.set_title('Sample Images', fontweight='bold', fontsize=12)
    except Exception as e:
        ax8.text(0.5, 0.5, f'Error loading\\nsample images:\\n{str(e)[:30]}...', 
                ha='center', va='center', transform=ax8.transAxes, fontsize=10)
        ax8.set_title('Sample Images', fontweight='bold', fontsize=12)
    
    # 9. Statistics Summary Table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('tight')
    ax9.axis('off')
    
    stats_data = [
        ['Total Images', f'{total:,}'],
        ['Number of Classes', f'{len(class_counts)}'],
        ['Average per Class', f'{np.mean(class_counts.values):.0f}'],
        ['Std Dev', f'{np.std(class_counts.values):.0f}'],
        ['Min Class Size', f'{class_counts.min():,}'],
        ['Max Class Size', f'{class_counts.max():,}'],
        ['Imbalance Ratio', f'{class_counts.max()/class_counts.min():.2f}:1'],
        ['Most Common Class', f'{class_counts.idxmax()}'],
        ['Least Common Class', f'{class_counts.idxmin()}'],
    ]
    
    table = ax9.table(cellText=stats_data, 
                     colLabels=['Statistic', 'Value'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax9.set_title('Dataset Statistics', fontweight='bold', fontsize=12, pad=20)
    
    # Style the table
    for i in range(len(stats_data) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
    
    plt.suptitle('Dataset Analysis & Visualization', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save the visualization
    viz_path = os.path.join(output_dir, "dataset_visualization.png")
    plt.savefig(viz_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    logging.info(f"✓ Dataset visualization saved: {viz_path}")
    
    # Create individual charts for better clarity
    create_individual_charts(df, output_dir, class_counts)
    
    return viz_path

def create_individual_charts(df, output_dir, class_counts):
    """Create individual charts for better clarity"""
    
    class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
    
    # 1. Enhanced Class Distribution Chart
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, len(class_counts)))
    bars = plt.bar(class_counts.index, class_counts.values, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    plt.title('Rice Disease Dataset - Class Distribution', fontweight='bold', fontsize=16, pad=20)
    plt.ylabel('Number of Images', fontweight='bold', fontsize=12)
    plt.xlabel('Disease Class', fontweight='bold', fontsize=12)
    
    # Add value and percentage labels
    total = class_counts.sum()
    for bar, count in zip(bars, class_counts.values):
        height = bar.get_height()
        pct = (count / total) * 100
        plt.text(bar.get_x() + bar.get_width()/2, height + 20,
                f'{count:,}\\n({pct:.1f}%)', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    chart_path = os.path.join(output_dir, "class_distribution_detailed.png")
    plt.savefig(chart_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    # 2. Data Source Analysis
    plt.figure(figsize=(14, 10))
    
    # Create source analysis data
    source_data = {class_name: {'Paddy Disease': 0, 'Sikhaok': 0, 'Original Data': 0, 'Other': 0} 
                  for class_name in class_names}
    
    for _, row in df.iterrows():
        class_name = row['label_name']
        path = row['image_path']
        
        if 'paddy_disease_train' in path:
            source_data[class_name]['Paddy Disease'] += 1
        elif 'sikhaok_train' in path:
            source_data[class_name]['Sikhaok'] += 1
        elif './data/' in path:
            source_data[class_name]['Original Data'] += 1
        else:
            source_data[class_name]['Other'] += 1
    
    # Create grouped bar chart
    x = np.arange(len(class_names))
    width = 0.2
    sources = ['Paddy Disease', 'Sikhaok', 'Original Data', 'Other']
    colors_src = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    
    for i, source in enumerate(sources):
        values = [source_data[class_name][source] for class_name in class_names]
        plt.bar(x + i * width, values, width, label=source, color=colors_src[i], alpha=0.8)
    
    plt.title('Data Distribution by Source and Class', fontweight='bold', fontsize=16, pad=20)
    plt.xlabel('Disease Class', fontweight='bold', fontsize=12)
    plt.ylabel('Number of Images', fontweight='bold', fontsize=12)
    plt.xticks(x + width * 1.5, class_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    source_chart_path = os.path.join(output_dir, "data_source_analysis.png")
    plt.savefig(source_chart_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    logging.info(f"✓ Individual charts saved:")
    logging.info(f"  - {chart_path}")
    logging.info(f"  - {source_chart_path}")

# Create data visualizations
visualize_dataset_distribution(collected_df, OUTPUT_DIRS["plots"])

# %%
# Split data
train_val_df, test_df = train_test_split(
    collected_df, test_size=CONFIG['test_size'], random_state=42, stratify=collected_df['label_id']
)
train_df, val_df = train_test_split(
    train_val_df, test_size=CONFIG['val_size']/(1-CONFIG['test_size']), 
    random_state=42, stratify=train_val_df['label_id']
)

logging.info(f"\nData splits:")
logging.info(f"Train: {len(train_df)} ({len(train_df)/len(collected_df)*100:.1f}%)")
logging.info(f"Val:   {len(val_df)} ({len(val_df)/len(collected_df)*100:.1f}%)")
logging.info(f"Test:  {len(test_df)} ({len(test_df)/len(collected_df)*100:.1f}%)")

# %%
# ===== DATA SPLITS VISUALIZATION =====
def visualize_data_splits(train_df, val_df, test_df, output_dir):
    """Visualize train/validation/test splits"""
    
    logging.info("\n" + "="*60)
    logging.info("CREATING DATA SPLITS VISUALIZATION")
    logging.info("="*60)
    
    class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Split Distribution (Pie Chart)
    ax1 = axes[0, 0]
    splits_data = {
        'Train': len(train_df),
        'Validation': len(val_df), 
        'Test': len(test_df)
    }
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    wedges, texts, autotexts = ax1.pie(splits_data.values(), labels=splits_data.keys(), 
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Train/Val/Test Split Distribution', fontweight='bold', fontsize=14)
    
    # 2. Class Distribution across Splits (Stacked Bar)
    ax2 = axes[0, 1]
    split_class_data = {}
    
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        class_counts = split_df.groupby('label_name').size()
        split_class_data[split_name] = [class_counts.get(class_name, 0) for class_name in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    for i, (split_name, counts) in enumerate(split_class_data.items()):
        ax2.bar(x + i * width, counts, width, label=split_name, color=colors[i], alpha=0.8)
    
    ax2.set_title('Class Distribution Across Splits', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Disease Class')
    ax2.set_ylabel('Number of Images')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Percentage Distribution by Class
    ax3 = axes[1, 0]
    
    # Calculate percentages for each class across splits
    class_percentages = {}
    for class_name in class_names:
        train_count = len(train_df[train_df['label_name'] == class_name])
        val_count = len(val_df[val_df['label_name'] == class_name])
        test_count = len(test_df[test_df['label_name'] == class_name])
        total_class = train_count + val_count + test_count
        
        if total_class > 0:
            class_percentages[class_name] = {
                'Train': (train_count / total_class) * 100,
                'Val': (val_count / total_class) * 100,
                'Test': (test_count / total_class) * 100
            }
    
    # Create stacked percentage bar chart
    train_pcts = [class_percentages[cls]['Train'] for cls in class_names]
    val_pcts = [class_percentages[cls]['Val'] for cls in class_names]
    test_pcts = [class_percentages[cls]['Test'] for cls in class_names]
    
    ax3.bar(class_names, train_pcts, label='Train', color=colors[0], alpha=0.8)
    ax3.bar(class_names, val_pcts, bottom=train_pcts, label='Val', color=colors[1], alpha=0.8)
    ax3.bar(class_names, test_pcts, bottom=np.array(train_pcts) + np.array(val_pcts), 
           label='Test', color=colors[2], alpha=0.8)
    
    ax3.set_title('Split Percentage by Class', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Disease Class')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_ylim(0, 100)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for i, class_name in enumerate(class_names):
        # Train percentage
        ax3.text(i, train_pcts[i]/2, f'{train_pcts[i]:.0f}%', 
                ha='center', va='center', fontweight='bold', fontsize=9)
        # Val percentage  
        ax3.text(i, train_pcts[i] + val_pcts[i]/2, f'{val_pcts[i]:.0f}%', 
                ha='center', va='center', fontweight='bold', fontsize=9)
        # Test percentage
        ax3.text(i, train_pcts[i] + val_pcts[i] + test_pcts[i]/2, f'{test_pcts[i]:.0f}%', 
                ha='center', va='center', fontweight='bold', fontsize=9)
    
    # 4. Split Statistics Table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    # Calculate statistics
    total_images = len(train_df) + len(val_df) + len(test_df)
    
    stats_data = [
        ['Total Images', f'{total_images:,}'],
        ['Training Images', f'{len(train_df):,} ({len(train_df)/total_images*100:.1f}%)'],
        ['Validation Images', f'{len(val_df):,} ({len(val_df)/total_images*100:.1f}%)'],
        ['Test Images', f'{len(test_df):,} ({len(test_df)/total_images*100:.1f}%)'],
        ['Train/Val Ratio', f'{len(train_df)/len(val_df):.1f}:1'],
        ['Train/Test Ratio', f'{len(train_df)/len(test_df):.1f}:1'],
        ['Classes per Split', f'{len(class_names)} (all)'],
    ]
    
    # Add per-class statistics
    for class_name in class_names:
        train_count = len(train_df[train_df['label_name'] == class_name])
        val_count = len(val_df[val_df['label_name'] == class_name])
        test_count = len(test_df[test_df['label_name'] == class_name])
        stats_data.append([f'{class_name} (T/V/Te)', f'{train_count}/{val_count}/{test_count}'])
    
    table = ax4.table(cellText=stats_data, 
                     colLabels=['Statistic', 'Value'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.3)
    ax4.set_title('Split Statistics Summary', fontweight='bold', fontsize=14, pad=20)
    
    # Style the table
    for i in range(len(stats_data) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
    
    plt.suptitle('Data Splits Analysis', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save visualization
    splits_viz_path = os.path.join(output_dir, "data_splits_visualization.png")
    plt.savefig(splits_viz_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    logging.info(f"✓ Data splits visualization saved: {splits_viz_path}")
    
    return splits_viz_path

# Create data splits visualization
visualize_data_splits(train_df, val_df, test_df, OUTPUT_DIRS["plots"])

# %% [markdown]
# ## TRAINING COMPONENTS

# %%
# ===== SYMMETRIC CROSS-ENTROPY =====
class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1, beta=1.0, num_classes=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
    
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets)
        pred = F.softmax(logits, dim=1).clamp(min=1e-7, max=1-1e-7)
        onehot = F.one_hot(targets, self.num_classes).float()
        rce = (-torch.sum(onehot * torch.log(pred), dim=1)).mean()
        return self.alpha * ce + self.beta * rce

# ===== MIXUP =====
def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ===== EMA =====
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
    
    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])

# ===== CLAHE TRANSFORM =====
class CLAHETransform:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to PIL Images"""
    
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Args:
            clip_limit: Threshold for contrast limiting (higher = more contrast)
            tile_grid_size: Size of grid for histogram equalization (e.g., (8,8))
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    def __call__(self, pil_image):
        """
        Apply CLAHE to PIL Image
        
        Args:
            pil_image: PIL Image in RGB format
            
        Returns:
            PIL Image with CLAHE applied
        """
        # Convert PIL to numpy array
        img_array = np.array(pil_image)
        
        # Convert RGB to LAB color space (better for CLAHE)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel (luminance)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Convert back to PIL Image
        return Image.fromarray(enhanced.astype(np.uint8))
    
    def __repr__(self):
        return f"CLAHETransform(clip_limit={self.clip_limit}, tile_grid_size={self.tile_grid_size})"

# ===== DATASET =====
class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label_id"])
        if self.transform:
            img = self.transform(img)
        return img, label

# ===== TRANSFORMS =====
def get_transforms(size):
    """Get training and validation transforms with optional CLAHE preprocessing"""
    
    # Base transform components
    base_train_transforms = []
    base_val_transforms = []
    
    # Add CLAHE if enabled
    if CONFIG['use_clahe']:
        clahe_transform = CLAHETransform(
            clip_limit=CONFIG['clahe_clip_limit'],
            tile_grid_size=CONFIG['clahe_tile_size']
        )
        base_train_transforms.append(clahe_transform)
        base_val_transforms.append(clahe_transform)
        logging.info(f"CLAHE enabled: clip_limit={CONFIG['clahe_clip_limit']}, tile_size={CONFIG['clahe_tile_size']}")
    
    # Training transforms
    train_transforms = base_train_transforms + [
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    
    # Validation transforms
    val_transforms = base_val_transforms + [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    
    train_transform = transforms.Compose(train_transforms)
    val_transform = transforms.Compose(val_transforms)
    
    return train_transform, val_transform

# ===== DATA LOADERS =====
def make_loader(df, transform, batch_size, train=True):
    dataset = ImageDataset(df, transform=transform)
    
    if train and CONFIG['use_weighted_sampler']:
        counts = df['label_id'].value_counts().sort_index().values.astype(float)
        class_weights = 1.0 / (counts + 1e-6)
        sample_weights = df['label_id'].map({i:w for i,w in enumerate(class_weights)}).values
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        loader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'],
            drop_last=train
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=train,
            num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'],
            drop_last=train
        )
    
    return loader

# %% [markdown]
# ## TRAINING FUNCTION

# %%
def train_single_model(model_name: str, train_df, val_df, epochs):
    """Train a single model"""
    
    logging.info("\n" + "="*80)
    logging.info(f"TRAINING: {model_name}")
    logging.info("="*80)
    
    device = DEVICE
    num_classes = len(LABELS)
    
    # Build model
    model, total_params, trainable_params = build_classifier(model_name, num_classes)
    model = model.to(device)
    
    logging.info(f"Total params: {total_params:,} ({total_params/1e6:.2f}M)")
    logging.info(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Setup AMP
    use_cuda = torch.cuda.is_available()
    if _NEW_AMP:
        amp_ctx = autocast(device_type="cuda", enabled=use_cuda)
        scaler = GradScaler(device="cuda" if use_cuda else "cpu", enabled=use_cuda)
    else:
        from contextlib import nullcontext
        amp_ctx = autocast(enabled=use_cuda) if use_cuda else nullcontext()
        scaler = GradScaler(enabled=use_cuda)
    
    # Loss
    if CONFIG['use_sce_loss']:
        criterion = SymmetricCrossEntropy(CONFIG['sce_alpha'], CONFIG['sce_beta'], num_classes)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Prepare loaders first (needed for steps_per_epoch)
    train_transform, val_transform = get_transforms(CONFIG['img_size'])
    train_loader = make_loader(train_df, train_transform, CONFIG['batch_size'], train=True)
    val_loader = make_loader(val_df, val_transform, CONFIG['batch_size'], train=False)
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG['lr'],
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # EMA
    ema = ModelEMA(model, decay=CONFIG['ema_decay']) if CONFIG['use_ema'] else None
    
    best_val_acc = 0.0
    bad_epochs = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_path = os.path.join(OUTPUT_DIRS["weights"], f"{model_name}_best.pth")
    
    for epoch in range(1, epochs + 1):
        # ===== TRAIN =====
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch}/{epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.long().to(device)
            
            optimizer.zero_grad()
            
            # Apply mixup if enabled
            if CONFIG['use_mixup'] and random.random() < 0.3:
                imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, CONFIG['mixup_alpha'])
                
                with amp_ctx:
                    logits = model(imgs)
                    loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            else:
                with amp_ctx:
                    logits = model(imgs)
                    loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if ema is not None:
                ema.update(model)
            
            epoch_loss += loss.item() * imgs.size(0)
            pred = logits.argmax(1)
            correct += (pred == labels).sum().item()
            total += imgs.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss = epoch_loss / total
        train_acc = correct / total
        
        # ===== VALIDATE =====
        eval_model = ema.ema if (ema is not None) else model
        eval_model.eval()
        vloss, vcorrect, vtotal = 0.0, 0, 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.long().to(device)
                
                with amp_ctx:
                    logits = eval_model(imgs)
                    loss = F.cross_entropy(logits, labels)
                
                vloss += loss.item() * imgs.size(0)
                vcorrect += (logits.argmax(1) == labels).sum().item()
                vtotal += imgs.size(0)
        
        val_loss = vloss / vtotal
        val_acc = vcorrect / vtotal
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        scheduler.step()
        
        logging.info(f"[{model_name}] Epoch {epoch}: TL={train_loss:.4f} TA={train_acc:.4f} | VL={val_loss:.4f} VA={val_acc:.4f}")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            bad_epochs = 0
            torch.save({
                'model_state_dict': eval_model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'model_name': model_name,
            }, best_path)
            logging.info(f"  → Saved best (Val Acc {val_acc:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= CONFIG['patience']:
                logging.info(f"[{model_name}] Early stopping triggered")
                break
    
    # Clean up
    del model, optimizer, scheduler
    if ema is not None:
        del ema
    torch.cuda.empty_cache()
    gc.collect()
    
    return history, best_path, best_val_acc

# %% [markdown]
# ## TESTING FUNCTION

# %%
def benchmark_model(model_name: str, checkpoint_path: str, device, num_warmup=20, num_runs=100):
    """Benchmark model for FPS and inference time"""
    
    num_classes = len(LABELS)
    model, total_params, _ = build_classifier(model_name, num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, CONFIG['img_size'], CONFIG['img_size']).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_inference_time = (total_time / num_runs) * 1000  # Convert to ms
    fps = num_runs / total_time
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'fps': fps,
        'inference_time_ms': avg_inference_time,
        'total_params': total_params
    }

def test_model(model_name: str, checkpoint_path: str, test_df, device):
    """Test a single model"""
    
    num_classes = len(LABELS)
    model, _, _ = build_classifier(model_name, num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    _, val_transform = get_transforms(CONFIG['img_size'])
    test_loader = make_loader(test_df, val_transform, CONFIG['batch_size'], train=False)
    
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc=f"Testing {model_name}"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(outputs.argmax(1).cpu().numpy())
            all_labels.append(labels.numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_probs = np.concatenate(all_probs)
    
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'accuracy': acc,
        'precision': p,
        'recall': r,
        'f1': f1,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_probs': y_probs
    }

# %% [markdown]
# ## TRAIN ALL MODELS

# %%
# Train all models
all_histories = {}
all_checkpoints = {}
all_best_val_accs = {}

for model_name in CONFIG['models']:
    history, checkpoint, best_val_acc = train_single_model(
        model_name, train_df, val_df, CONFIG['epochs']
    )
    all_histories[model_name] = history
    all_checkpoints[model_name] = checkpoint
    all_best_val_accs[model_name] = best_val_acc
    
    logging.info(f"\n✓ {model_name} completed - Best Val Acc: {best_val_acc:.4f}\n")

# %% [markdown]
# ## TEST ALL MODELS

# %%
# Test all models
all_test_metrics = {}

logging.info("\n" + "="*80)
logging.info("TESTING ALL MODELS")
logging.info("="*80)

for model_name in CONFIG['models']:
    test_metrics = test_model(
        model_name, all_checkpoints[model_name], test_df, DEVICE
    )
    all_test_metrics[model_name] = test_metrics
    
    logging.info(f"\n{model_name}:")
    logging.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    logging.info(f"  Precision: {test_metrics['precision']:.4f}")
    logging.info(f"  Recall:    {test_metrics['recall']:.4f}")
    logging.info(f"  F1 Score:  {test_metrics['f1']:.4f}")

# %% [markdown]
# ## ENSEMBLE PREDICTION

# %%
def ensemble_predict(all_probs_dict):
    """Ensemble prediction using average probabilities"""
    # Stack all probability predictions
    all_probs = np.stack([probs for probs in all_probs_dict.values()])
    
    # Average probabilities
    avg_probs = np.mean(all_probs, axis=0)
    
    # Get predictions
    ensemble_preds = np.argmax(avg_probs, axis=1)
    
    return ensemble_preds, avg_probs

if CONFIG['use_ensemble'] and len(CONFIG['models']) > 1:
    logging.info("\n" + "="*80)
    logging.info("ENSEMBLE PREDICTION")
    logging.info("="*80)
    
    # Collect all probabilities
    all_probs_dict = {name: metrics['y_probs'] for name, metrics in all_test_metrics.items()}
    y_true = all_test_metrics[CONFIG['models'][0]]['y_true']
    
    # Ensemble prediction
    ensemble_preds, ensemble_probs = ensemble_predict(all_probs_dict)
    
    # Calculate metrics
    acc = accuracy_score(y_true, ensemble_preds)
    p, r, f1, _ = precision_recall_fscore_support(y_true, ensemble_preds, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, ensemble_preds)
    
    ensemble_metrics = {
        'accuracy': acc,
        'precision': p,
        'recall': r,
        'f1': f1,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': ensemble_preds,
        'y_probs': ensemble_probs
    }
    
    all_test_metrics['ensemble'] = ensemble_metrics
    
    logging.info(f"\nEnsemble Results:")
    logging.info(f"  Accuracy:  {acc:.4f}")
    logging.info(f"  Precision: {p:.4f}")
    logging.info(f"  Recall:    {r:.4f}")
    logging.info(f"  F1 Score:  {f1:.4f}")

# %% [markdown]
# ## VISUALIZATION & COMPARISON

# %%
def plot_training_curves(all_histories):
    """Plot training curves for all models"""
    
    n_models = len(all_histories)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (model_name, history) in enumerate(all_histories.items()):
        row, col = idx // cols, idx % cols
        ax_acc = axes[row, col] if rows > 1 else axes[col]
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Plot accuracy
        ax_acc.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        ax_acc.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.set_title(f'{model_name} - Accuracy', fontweight='bold')
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)
        ax_acc.set_ylim(0, 1)
        
        # Add loss on secondary axis
        ax_loss = ax_acc.twinx()
        ax_loss.plot(epochs, history['train_loss'], 'g--', label='Train Loss', alpha=0.7)
        ax_loss.plot(epochs, history['val_loss'], 'm--', label='Val Loss', alpha=0.7)
        ax_loss.set_ylabel('Loss')
        ax_loss.legend(loc='upper right')
    
    # Hide empty subplots
    for idx in range(n_models, rows * cols):
        if rows > 1:
            row, col = idx // cols, idx % cols
            axes[row, col].axis('off')
        elif cols > 1 and idx < len(axes):
            axes[idx].axis('off')
    
    plt.suptitle('Training Curves - All Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIRS["plots"], "training_curves.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    logging.info(f"✓ Training curves saved: {save_path}")

def plot_model_comparison(all_test_metrics, all_histories):
    """Plot comprehensive comparison of all models"""
    
    models = [m for m in CONFIG['models']]
    if 'ensemble' in all_test_metrics:
        models.append('ensemble')
    
    class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 14))
    
    # 1. Test Accuracy Comparison
    ax1 = plt.subplot(3, 3, 1)
    test_accs = [all_test_metrics[m]['accuracy'] for m in models]
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = ax1.bar(range(len(models)), test_accs, color=colors, alpha=0.8)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([m.replace('_', '\n') for m in models], rotation=0, ha='center', fontsize=9)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Test Accuracy Comparison', fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3, axis='y')
    for i, (bar, acc) in enumerate(zip(bars, test_accs)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. F1 Score Comparison
    ax2 = plt.subplot(3, 3, 2)
    f1_scores = [all_test_metrics[m]['f1'] for m in models]
    bars = ax2.bar(range(len(models)), f1_scores, color=colors, alpha=0.8)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels([m.replace('_', '\n') for m in models], rotation=0, ha='center', fontsize=9)
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score Comparison', fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3, axis='y')
    for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Precision & Recall Comparison
    ax3 = plt.subplot(3, 3, 3)
    precisions = [all_test_metrics[m]['precision'] for m in models]
    recalls = [all_test_metrics[m]['recall'] for m in models]
    x = np.arange(len(models))
    width = 0.35
    ax3.bar(x - width/2, precisions, width, label='Precision', alpha=0.8)
    ax3.bar(x + width/2, recalls, width, label='Recall', alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.replace('_', '\n') for m in models], rotation=0, ha='center', fontsize=9)
    ax3.set_ylabel('Score')
    ax3.set_title('Precision & Recall', fontweight='bold')
    ax3.set_ylim(0, 1.0)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4-6. Training curves for first 3 models
    for idx, model_name in enumerate(CONFIG['models'][:3], start=4):
        ax = plt.subplot(3, 3, idx)
        history = all_histories[model_name]
        epochs_range = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs_range, history['train_acc'], 'b-', label='Train', linewidth=2)
        ax.plot(epochs_range, history['val_acc'], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{model_name} - Training', fontweight='bold', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 7-9. Confusion matrices for best 3 models
    best_models = sorted(models, key=lambda m: all_test_metrics[m]['accuracy'], reverse=True)[:3]
    for idx, model_name in enumerate(best_models, start=7):
        ax = plt.subplot(3, 3, idx)
        cm = all_test_metrics[model_name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        acc = all_test_metrics[model_name]['accuracy']
        ax.set_title(f'{model_name}\nAcc: {acc:.4f}', fontweight='bold', fontsize=10)
    
    plt.suptitle('Multi-Model Training & Comparison', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIRS["comparison"], "model_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    logging.info(f"✓ Comparison plot saved: {save_path}")

# Plot training curves
plot_training_curves(all_histories)

# Plot model comparison
plot_model_comparison(all_test_metrics, all_histories)

# %%
def plot_detailed_comparison(all_test_metrics):
    """Plot per-class performance for all models"""
    
    models = [m for m in CONFIG['models']]
    if 'ensemble' in all_test_metrics:
        models.append('ensemble')
    
    class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
    
    # Calculate per-class metrics for each model
    fig, axes = plt.subplots(1, len(class_names), figsize=(20, 5))
    
    for class_idx, class_name in enumerate(class_names):
        ax = axes[class_idx]
        
        class_f1_scores = []
        for model_name in models:
            y_true = all_test_metrics[model_name]['y_true']
            y_pred = all_test_metrics[model_name]['y_pred']
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=[class_idx], zero_division=0
            )
            class_f1_scores.append(f1[0])
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        bars = ax.bar(range(len(models)), class_f1_scores, color=colors, alpha=0.8)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('F1 Score')
        ax.set_title(f'{class_name}', fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, f1 in zip(bars, class_f1_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{f1:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Per-Class F1 Score Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIRS["comparison"], "per_class_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    logging.info(f"✓ Per-class comparison saved: {save_path}")

plot_detailed_comparison(all_test_metrics)

# %%
def plot_advanced_metrics_visualization(all_test_metrics, all_histories, all_benchmarks, output_dir):
    """Create advanced metrics visualization"""
    
    logging.info("\n" + "="*60)
    logging.info("CREATING ADVANCED METRICS VISUALIZATION")
    logging.info("="*60)
    
    models = [m for m in CONFIG['models']]
    if 'ensemble' in all_test_metrics:
        models.append('ensemble')
    
    class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
    
    # Create large comprehensive figure
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Radar Chart for Model Performance
    ax1 = plt.subplot(4, 4, 1, projection='polar')
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Speed (norm)']
    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for model_name in CONFIG['models'][:4]:  # Show top 4 models
        metrics = all_test_metrics[model_name]
        benchmark = all_benchmarks[model_name]
        
        # Normalize speed (FPS) to 0-1 scale
        max_fps = max(all_benchmarks[m]['fps'] for m in CONFIG['models'])
        speed_norm = benchmark['fps'] / max_fps
        
        values = [
            metrics['accuracy'],
            metrics['precision'], 
            metrics['recall'],
            metrics['f1'],
            speed_norm
        ]
        values += values[:1]  # Complete the circle
        
        ax1.plot(angles, values, 'o-', linewidth=2, label=model_name, alpha=0.8)
        ax1.fill(angles, values, alpha=0.25)
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics_names)
    ax1.set_ylim(0, 1)
    ax1.set_title('Model Performance Radar Chart', fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    # 2. Loss Convergence Comparison
    ax2 = plt.subplot(4, 4, 2)
    colors = plt.cm.tab10(np.linspace(0, 1, len(CONFIG['models'])))
    
    for i, model_name in enumerate(CONFIG['models']):
        history = all_histories[model_name]
        epochs = range(1, len(history['val_loss']) + 1)
        ax2.plot(epochs, history['val_loss'], color=colors[i], 
                label=model_name, linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Loss Convergence Comparison', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Accuracy Convergence
    ax3 = plt.subplot(4, 4, 3)
    
    for i, model_name in enumerate(CONFIG['models']):
        history = all_histories[model_name]
        epochs = range(1, len(history['val_acc']) + 1)
        ax3.plot(epochs, history['val_acc'], color=colors[i], 
                label=model_name, linewidth=2, alpha=0.8)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Validation Accuracy')
    ax3.set_title('Accuracy Convergence', fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Model Efficiency (Accuracy vs Parameters)
    ax4 = plt.subplot(4, 4, 4)
    
    for i, model_name in enumerate(CONFIG['models']):
        metrics = all_test_metrics[model_name]
        benchmark = all_benchmarks[model_name]
        
        ax4.scatter(benchmark['total_params']/1e6, metrics['accuracy'], 
                   s=150, color=colors[i], alpha=0.7, label=model_name)
        ax4.annotate(model_name.replace('_', '\\n'), 
                    (benchmark['total_params']/1e6, metrics['accuracy']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Parameters (Millions)')
    ax4.set_ylabel('Test Accuracy')
    ax4.set_title('Model Efficiency', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5-8. Individual Confusion Matrices for top 4 models
    best_models = sorted(CONFIG['models'], key=lambda m: all_test_metrics[m]['accuracy'], reverse=True)[:4]
    
    for idx, model_name in enumerate(best_models, start=5):
        ax = plt.subplot(4, 4, idx)
        cm = all_test_metrics[model_name]['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, 
                   ax=ax, cbar=False, square=True)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        acc = all_test_metrics[model_name]['accuracy']
        ax.set_title(f'{model_name}\\n(Acc: {acc:.3f})', fontweight='bold', fontsize=10)
    
    # 9. Training Time Analysis (if we track it)
    ax9 = plt.subplot(4, 4, 9)
    
    # Estimate training time based on epochs completed
    training_times = []
    for model_name in CONFIG['models']:
        history = all_histories[model_name]
        epochs_completed = len(history['train_loss'])
        # Rough estimate: assume each epoch takes different time based on model complexity
        benchmark = all_benchmarks[model_name]
        est_time = epochs_completed * (benchmark['total_params'] / 1e6) * 2  # rough estimate
        training_times.append(est_time)
    
    bars = ax9.bar(CONFIG['models'], training_times, color=colors[:len(CONFIG['models'])], alpha=0.8)
    ax9.set_title('Estimated Training Time', fontweight='bold')
    ax9.set_ylabel('Estimated Time (minutes)')
    plt.setp(ax9.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    
    for bar, time_est in zip(bars, training_times):
        ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{time_est:.0f}m', ha='center', va='bottom', fontsize=8)
    
    # 10. Memory Usage Estimation
    ax10 = plt.subplot(4, 4, 10)
    
    memory_usage = []
    for model_name in CONFIG['models']:
        benchmark = all_benchmarks[model_name]
        # Rough memory estimate: params * 4 bytes * batch_size factor
        mem_est = (benchmark['total_params'] * 4 * CONFIG['batch_size']) / (1024**3)  # GB
        memory_usage.append(mem_est)
    
    bars = ax10.bar(CONFIG['models'], memory_usage, color=colors[:len(CONFIG['models'])], alpha=0.8)
    ax10.set_title('Estimated Memory Usage', fontweight='bold')
    ax10.set_ylabel('Memory (GB)')
    plt.setp(ax10.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    
    for bar, mem_est in zip(bars, memory_usage):
        ax10.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mem_est:.2f}GB', ha='center', va='bottom', fontsize=8)
    
    # 11. Error Analysis - Per Class Error Rates
    ax11 = plt.subplot(4, 4, 11)
    
    best_model = max(CONFIG['models'], key=lambda m: all_test_metrics[m]['accuracy'])
    cm = all_test_metrics[best_model]['confusion_matrix']
    
    # Calculate error rates per class
    error_rates = []
    for i in range(len(class_names)):
        total_class = cm[i].sum()
        correct = cm[i, i]
        error_rate = ((total_class - correct) / total_class) * 100 if total_class > 0 else 0
        error_rates.append(error_rate)
    
    bars = ax11.bar(class_names, error_rates, color='lightcoral', alpha=0.8)
    ax11.set_title(f'Error Rates by Class\\n({best_model})', fontweight='bold')
    ax11.set_ylabel('Error Rate (%)')
    plt.setp(ax11.get_xticklabels(), rotation=45, ha='right')
    
    for bar, err_rate in zip(bars, error_rates):
        ax11.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{err_rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 12. Model Ranking Summary
    ax12 = plt.subplot(4, 4, 12)
    ax12.axis('tight')
    ax12.axis('off')
    
    # Create ranking data
    ranking_data = []
    for i, model_name in enumerate(CONFIG['models']):
        metrics = all_test_metrics[model_name]
        benchmark = all_benchmarks[model_name]
        
        ranking_data.append([
            f"{i+1}. {model_name}",
            f"{metrics['accuracy']:.3f}",
            f"{benchmark['fps']:.0f}",
            f"{benchmark['total_params']/1e6:.1f}M"
        ])
    
    # Sort by accuracy
    ranking_data.sort(key=lambda x: float(x[1]), reverse=True)
    
    # Re-number rankings
    for i, row in enumerate(ranking_data):
        row[0] = f"{i+1}. {row[0].split('. ', 1)[1]}"
    
    table = ax12.table(cellText=ranking_data,
                      colLabels=['Model Ranking', 'Accuracy', 'FPS', 'Params'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.4, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax12.set_title('Model Performance Ranking', fontweight='bold', fontsize=12, pad=20)
    
    # Style table
    for i in range(len(ranking_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            else:
                # Highlight best model
                if i == 1:
                    cell.set_facecolor('#90EE90')  # Light green for best
                else:
                    cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
    
    # 13-16. Class-wise Performance Heatmaps
    for idx, metric_name in enumerate(['Precision', 'Recall', 'F1-Score'], start=13):
        ax = plt.subplot(4, 4, idx)
        
        # Create matrix of metric values
        metric_matrix = []
        for model_name in CONFIG['models']:
            y_true = all_test_metrics[model_name]['y_true']
            y_pred = all_test_metrics[model_name]['y_pred']
            
            if metric_name == 'Precision':
                scores = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)[0]
            elif metric_name == 'Recall':
                scores = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)[1]
            else:  # F1-Score
                scores = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)[2]
            
            metric_matrix.append(scores)
        
        metric_matrix = np.array(metric_matrix)
        
        sns.heatmap(metric_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   xticklabels=class_names, yticklabels=CONFIG['models'],
                   ax=ax, cbar=True, square=False)
        ax.set_title(f'{metric_name} Heatmap', fontweight='bold')
        ax.set_xlabel('Disease Class')
        ax.set_ylabel('Model')
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    
    plt.suptitle('Advanced Model Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save visualization
    advanced_viz_path = os.path.join(output_dir, "advanced_metrics_analysis.png")
    plt.savefig(advanced_viz_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    logging.info(f"✓ Advanced metrics visualization saved: {advanced_viz_path}")
    
    return advanced_viz_path

# %%
def create_clahe_demo(test_df, num_samples=8):
    """Create CLAHE comparison demo"""
    
    logging.info("\n" + "="*60)
    logging.info("CREATING CLAHE DEMONSTRATION")
    logging.info("="*60)
    
    # Sample images from each class
    class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
    demo_samples = []
    
    for class_id in range(len(class_names)):
        class_data = test_df[test_df['label_id'] == class_id]
        if len(class_data) > 0:
            samples = class_data.sample(min(2, len(class_data)), random_state=42)
            demo_samples.extend(samples.to_dict('records'))
    
    # Limit total samples
    if len(demo_samples) > num_samples:
        demo_samples = demo_samples[:num_samples]
    
    # Create CLAHE transform
    clahe_transform = CLAHETransform(
        clip_limit=CONFIG['clahe_clip_limit'],
        tile_grid_size=CONFIG['clahe_tile_size']
    )
    
    # Create comparison figure
    cols = 2  # Original, CLAHE
    rows = len(demo_samples)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample in enumerate(demo_samples):
        # Load image
        img_path = sample['image_path']
        original_img = Image.open(img_path).convert('RGB')
        
        # Apply CLAHE
        clahe_img = clahe_transform(original_img)
        
        # Display original
        axes[idx, 0].imshow(original_img)
        axes[idx, 0].set_title(f'Original - {sample["label_name"]}', fontweight='bold')
        axes[idx, 0].axis('off')
        
        # Display CLAHE enhanced
        axes[idx, 1].imshow(clahe_img)
        axes[idx, 1].set_title(f'CLAHE Enhanced - {sample["label_name"]}', fontweight='bold')
        axes[idx, 1].axis('off')
    
    plt.suptitle(f'CLAHE Comparison (clip_limit={CONFIG["clahe_clip_limit"]}, tile_size={CONFIG["clahe_tile_size"]})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save demo
    demo_path = os.path.join(OUTPUT_DIRS["demo"], "clahe_comparison.png")
    plt.savefig(demo_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    logging.info(f"✓ CLAHE demo saved: {demo_path}")
    
    return demo_path

def create_demo_predictions(all_checkpoints, test_df, num_samples=16):
    """Create demo predictions with sample images"""
    
    logging.info("\n" + "="*60)
    logging.info("CREATING DEMO PREDICTIONS")
    logging.info("="*60)
    
    # Sample images from each class
    class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
    demo_samples = []
    
    for class_id in range(len(class_names)):
        class_data = test_df[test_df['label_id'] == class_id]
        if len(class_data) > 0:
            samples = class_data.sample(min(4, len(class_data)), random_state=42)
            demo_samples.extend(samples.to_dict('records'))
    
    # Limit total samples
    if len(demo_samples) > num_samples:
        demo_samples = demo_samples[:num_samples]
    
    # Get transforms
    _, val_transform = get_transforms(CONFIG['img_size'])
    
    # Create figure
    cols = 4
    rows = (len(demo_samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Get best model for predictions
    best_model_name = max(CONFIG['models'], key=lambda m: all_test_metrics[m]['accuracy'])
    best_checkpoint = all_checkpoints[best_model_name]
    
    # Load best model
    num_classes = len(LABELS)
    model, _, _ = build_classifier(best_model_name, num_classes)
    checkpoint = torch.load(best_checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    demo_results = []
    
    for idx, sample in enumerate(demo_samples):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Load and preprocess image
        img_path = sample['image_path']
        img = Image.open(img_path).convert('RGB')
        img_tensor = val_transform(img).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(1).item()
            confidence = probs[0, pred_class].item()
        
        # True and predicted labels
        true_class = sample['label_id']
        true_name = class_names[true_class]
        pred_name = class_names[pred_class]
        
        # Display image
        ax.imshow(img)
        ax.axis('off')
        
        # Title with prediction
        color = 'green' if pred_class == true_class else 'red'
        title = f"True: {true_name}\nPred: {pred_name} ({confidence:.2f})"
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        
        # Store result
        demo_results.append({
            'image_path': img_path,
            'true_class': true_name,
            'predicted_class': pred_name,
            'confidence': confidence,
            'correct': pred_class == true_class
        })
    
    # Hide empty subplots
    for idx in range(len(demo_samples), rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Demo Predictions - {best_model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save demo
    demo_path = os.path.join(OUTPUT_DIRS["demo"], "predictions_demo.png")
    plt.savefig(demo_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    # Save demo results CSV
    demo_df = pd.DataFrame(demo_results)
    demo_csv_path = os.path.join(OUTPUT_DIRS["demo"], "demo_predictions.csv")
    demo_df.to_csv(demo_csv_path, index=False)
    
    accuracy = demo_df['correct'].mean()
    logging.info(f"✓ Demo accuracy: {accuracy:.2f} ({demo_df['correct'].sum()}/{len(demo_df)})")
    logging.info(f"✓ Demo images saved: {demo_path}")
    logging.info(f"✓ Demo CSV saved: {demo_csv_path}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return demo_results

# Create CLAHE demonstration
if CONFIG['use_clahe']:
    clahe_demo_path = create_clahe_demo(test_df)

# Create demo predictions
demo_results = create_demo_predictions(all_checkpoints, test_df)

# %%
# Benchmark all models for performance metrics
all_benchmarks = {}
logging.info("\n" + "="*60)
logging.info("BENCHMARKING MODELS")
logging.info("="*60)

for model_name in CONFIG['models']:
    logging.info(f"Benchmarking {model_name}...")
    benchmark = benchmark_model(model_name, all_checkpoints[model_name], DEVICE)
    all_benchmarks[model_name] = benchmark
    logging.info(f"  FPS: {benchmark['fps']:.1f}, Inference: {benchmark['inference_time_ms']:.2f}ms")

# Create advanced metrics visualization
plot_advanced_metrics_visualization(all_test_metrics, all_histories, all_benchmarks, OUTPUT_DIRS["comparison"])

# %%
# Create comprehensive results summary
results_summary = []
class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]

for model_name in CONFIG['models']:
    metrics = all_test_metrics[model_name]
    benchmark = all_benchmarks[model_name]
    cm = metrics['confusion_matrix']
    
    # Calculate per-class metrics
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        metrics['y_true'], metrics['y_pred'], average=None, zero_division=0
    )
    
    # Flatten confusion matrix for CSV
    cm_flat = {}
    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            cm_flat[f'CM_{true_class}_to_{pred_class}'] = cm[i, j]
    
    # Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[f'{class_name}_Precision'] = per_class_precision[i]
        per_class_metrics[f'{class_name}_Recall'] = per_class_recall[i]
        per_class_metrics[f'{class_name}_F1'] = per_class_f1[i]
    
    result_row = {
        'Model': model_name,
        'Parameters_M': f"{benchmark['total_params']/1e6:.2f}",
        'FPS': f"{benchmark['fps']:.1f}",
        'Inference_Time_ms': f"{benchmark['inference_time_ms']:.2f}",
        'Best_Val_Acc': f"{all_best_val_accs[model_name]:.4f}",
        'Test_Accuracy': f"{metrics['accuracy']:.4f}",
        'Test_Precision_Macro': f"{metrics['precision']:.4f}",
        'Test_Recall_Macro': f"{metrics['recall']:.4f}",
        'Test_F1_Macro': f"{metrics['f1']:.4f}",
        **per_class_metrics,
        **cm_flat
    }
    
    results_summary.append(result_row)

# Add ensemble if available
if 'ensemble' in all_test_metrics:
    metrics = all_test_metrics['ensemble']
    cm = metrics['confusion_matrix']
    
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        metrics['y_true'], metrics['y_pred'], average=None, zero_division=0
    )
    
    cm_flat = {}
    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            cm_flat[f'CM_{true_class}_to_{pred_class}'] = cm[i, j]
    
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[f'{class_name}_Precision'] = per_class_precision[i]
        per_class_metrics[f'{class_name}_Recall'] = per_class_recall[i]
        per_class_metrics[f'{class_name}_F1'] = per_class_f1[i]
    
    result_row = {
        'Model': 'ensemble',
        'Parameters_M': f"{sum(all_benchmarks[m]['total_params'] for m in CONFIG['models'])/1e6:.2f}",
        'FPS': f"{min(all_benchmarks[m]['fps'] for m in CONFIG['models']):.1f}",
        'Inference_Time_ms': f"{max(all_benchmarks[m]['inference_time_ms'] for m in CONFIG['models']):.2f}",
        'Best_Val_Acc': '-',
        'Test_Accuracy': f"{metrics['accuracy']:.4f}",
        'Test_Precision_Macro': f"{metrics['precision']:.4f}",
        'Test_Recall_Macro': f"{metrics['recall']:.4f}",
        'Test_F1_Macro': f"{metrics['f1']:.4f}",
        **per_class_metrics,
        **cm_flat
    }
    
    results_summary.append(result_row)

summary_df = pd.DataFrame(results_summary)
summary_path = os.path.join(OUTPUT_DIRS["results"], "comprehensive_results.csv")
summary_df.to_csv(summary_path, index=False)

# Also create a simplified summary
simple_summary = []
for model_name in CONFIG['models']:
    metrics = all_test_metrics[model_name]
    benchmark = all_benchmarks[model_name]
    simple_summary.append({
        'Model': model_name,
        'Parameters_M': f"{benchmark['total_params']/1e6:.2f}",
        'FPS': f"{benchmark['fps']:.1f}",
        'Inference_ms': f"{benchmark['inference_time_ms']:.2f}",
        'Test_Accuracy': f"{metrics['accuracy']:.4f}",
        'Test_F1': f"{metrics['f1']:.4f}",
        'Best_Val_Acc': f"{all_best_val_accs[model_name]:.4f}",
    })

if 'ensemble' in all_test_metrics:
    metrics = all_test_metrics['ensemble']
    simple_summary.append({
        'Model': 'ensemble',
        'Parameters_M': f"{sum(all_benchmarks[m]['total_params'] for m in CONFIG['models'])/1e6:.2f}",
        'FPS': f"{min(all_benchmarks[m]['fps'] for m in CONFIG['models']):.1f}",
        'Inference_ms': f"{max(all_benchmarks[m]['inference_time_ms'] for m in CONFIG['models']):.2f}",
        'Test_Accuracy': f"{metrics['accuracy']:.4f}",
        'Test_F1': f"{metrics['f1']:.4f}",
        'Best_Val_Acc': '-',
    })

simple_df = pd.DataFrame(simple_summary)
simple_path = os.path.join(OUTPUT_DIRS["results"], "models_summary.csv")
simple_df.to_csv(simple_path, index=False)

logging.info("\n" + "="*80)
logging.info("RESULTS SUMMARY")
logging.info("="*80)
print(simple_df.to_string(index=False))

# Save detailed reports for each model
for model_name, metrics in all_test_metrics.items():
    class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
    report = classification_report(
        metrics['y_true'], metrics['y_pred'],
        target_names=class_names, digits=4
    )
    
    report_path = os.path.join(OUTPUT_DIRS["results"], f"{model_name}_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"{model_name} - Test Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n\n")
        f.write(report)

# Create performance summary chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Accuracy vs Parameters
models_data = [(m, all_benchmarks[m]['total_params']/1e6, all_test_metrics[m]['accuracy']) 
               for m in CONFIG['models']]
models_data.sort(key=lambda x: x[2], reverse=True)  # Sort by accuracy

model_names = [x[0] for x in models_data]
params = [x[1] for x in models_data]
accs = [x[2] for x in models_data]

colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
scatter = ax1.scatter(params, accs, c=colors, s=100, alpha=0.7)
for i, name in enumerate(model_names):
    ax1.annotate(name.replace('_', '\n'), (params[i], accs[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)
ax1.set_xlabel('Parameters (M)')
ax1.set_ylabel('Test Accuracy')
ax1.set_title('Accuracy vs Model Size', fontweight='bold')
ax1.grid(True, alpha=0.3)

# FPS vs Accuracy
fps_data = [all_benchmarks[m]['fps'] for m in CONFIG['models']]
scatter = ax2.scatter(fps_data, accs, c=colors, s=100, alpha=0.7)
for i, name in enumerate(model_names):
    ax2.annotate(name.replace('_', '\n'), (fps_data[i], accs[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)
ax2.set_xlabel('FPS')
ax2.set_ylabel('Test Accuracy')
ax2.set_title('Accuracy vs Speed', fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
perf_chart_path = os.path.join(OUTPUT_DIRS["comparison"], "performance_summary.png")
plt.savefig(perf_chart_path, dpi=200, bbox_inches='tight')
plt.show()

logging.info("\n" + "="*80)
logging.info("TRAINING COMPLETE")
logging.info("="*80)
logging.info(f"Output directory: {PATH_OUTPUT}")
logging.info(f"Simple summary: {simple_path}")
logging.info(f"Comprehensive results: {summary_path}")
logging.info(f"Training curves: {os.path.join(OUTPUT_DIRS['plots'], 'training_curves.png')}")
logging.info(f"Demo predictions: {os.path.join(OUTPUT_DIRS['demo'], 'predictions_demo.png')}")
logging.info("="*80)