# %% [markdown]
# # Optimized Rice Disease Detection Pipeline
# Complete pipeline with lesion-aware filtering, robust training, and comprehensive visualization

# %%
# ===== IMPORTS =====
import os, shutil, random, cv2, torch, gc, time, subprocess, sys
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
import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# %%
# ===== OPTIMIZED CONFIGURATION =====
CONFIG = {
    # Backbone selection
    'backbone': 'efficientnet_v2_s',  # 'mobilenetv3_small' | 'efficientnet_v2_s'
    
    # Model Enhancements
    'use_cbam': True,
    'use_better_head': True,
    
    # Training - Conservative memory settings
    'img_size': 224,
    'batch_size': 128,  # Very safe for 22GB VRAM
    'epochs': 60,
    'lr': 3e-4,
    'warmup_epochs': 3,
    
    # Progressive Resizing - Smaller sizes
    'use_progressive_resize': True,
    'progressive_schedule': {
        1: 224,
        10: 256,
        25: 288,   # Reduced from 320
        45: 320    # Reduced from 384, later epoch
    },
    
    # Data loader - Ultra conservative
    'num_workers': 4,  # Further reduced to save memory
    'pin_memory': True,
    'persistent_workers': False,  # Disable to save memory
    'prefetch_factor': 2,  # Minimal prefetch
    
    # GPU Optimization - Conservative settings
    'gradient_accumulation_steps': 4,  # Effective batch size = 128*4 = 512
    
    # Memory optimization - Very conservative
    'max_memory_allocated': 15.0,  # Max GB to use (leave 7GB buffer)
    'memory_cleanup_frequency': 3,  # Clean memory every 3 batches
    
    # Robust Training
    'use_weighted_sampler': True,
    'use_sce_loss': True,           # Symmetric Cross-Entropy
    'sce_alpha': 0.1,
    'sce_beta': 1.0,
    'use_logit_adjustment': True,
    'logit_adjustment_tau': 1.0,
    'use_ema': True,
    'ema_decay': 0.999,
    
    # Augmentation (disabled initially for noisy labels)
    'use_mixup': False,
    'use_cutmix': False,
    'use_label_smoothing': False,
    
    # Inference
    'use_tta': True,
    'tta_transforms': 5,
    
    # YOLO (LEAF DETECTION ONLY)
    'yolo_epochs': 30,
    'yolo_imgsz': 640,
    'yolo_batch': 16,
    'yolo_conf': 0.25,
    
    # Splits
    'val_size': 0.15,
    'test_size': 0.15,
    
    # Crop Filtering
    'crops_per_image': 3,
    'min_crop_size': 64,
}

# %%
# ===== SETUP =====
def get_output_folder(parent_dir: str, env_name: str) -> str:
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = f"{env_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir = os.path.join(parent_dir, experiment_id)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

PATH_OUTPUT = get_output_folder("../output", "GK-optimized")

def create_output_structure(base_path):
    folders = ["field_images", "yolo_weights", "crops", "crop_samples",
               "weights", "results", "plots", "logs", "exports", "demo"]
    for folder in folders:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)
    return {folder: os.path.join(base_path, folder) for folder in folders}

OUTPUT_DIRS = create_output_structure(PATH_OUTPUT)

def setup_logging(output_path):
    log_file = os.path.join(output_path, "logs", "pipeline.log")
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
    0: {"name": "brown_spot", "match_substrings": ["../data/new_data_field_rice/brown_spot"]},
    1: {"name": "leaf_blast", "match_substrings": ["../data/new_data_field_rice/leaf_blast"]},
    2: {"name": "leaf_blight", "match_substrings": ["../data/new_data_field_rice/leaf_blight"]},
    3: {"name": "healthy", "match_substrings": ["../data/new_data_field_rice/healthy"]}
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device: {DEVICE}")
if torch.cuda.is_available():
    logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

# %%
# ===== INSTALL ULTRALYTICS =====
def install_ultralytics():
    try:
        import ultralytics
        logging.info("Ultralytics already installed")
        return True
    except ImportError:
        logging.info("Installing ultralytics...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        return True

YOLO_AVAILABLE = install_ultralytics()
if YOLO_AVAILABLE:
    from ultralytics import YOLO

# %% [markdown]
# ## ENHANCED MODEL COMPONENTS

# %%
# ===== CBAM ATTENTION =====
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
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
            nn.BatchNorm1d(hidden),
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

# ===== BACKBONE BUILDER =====
def build_backbone_and_channels(backbone_name):
    from torchvision.models import (
        mobilenet_v3_small, MobileNet_V3_Small_Weights,
        efficientnet_v2_s, EfficientNet_V2_S_Weights
    )
    
    if backbone_name == 'mobilenetv3_small':
        weights = MobileNet_V3_Small_Weights.DEFAULT
        net = mobilenet_v3_small(weights=weights)
        feat_channels = net.features[-1][0].out_channels
        features = net.features
        return net, features, feat_channels
    
    elif backbone_name == 'efficientnet_v2_s':
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        net = efficientnet_v2_s(weights=weights)
        features = net.features
        # Get output channels from last block
        feat_channels = 1280  # EfficientNetV2-S final conv channels
        return net, features, feat_channels
    
    else:
        raise ValueError(f'Unknown backbone: {backbone_name}')

def build_enhanced_classifier(num_classes):
    base, features, C = build_backbone_and_channels(CONFIG['backbone'])
    logging.info(f"Backbone: {CONFIG['backbone']} with {C} channels")
    
    # Add CBAM if enabled
    if CONFIG['use_cbam']:
        cbam = CBAM(C, reduction=16)
        features = nn.Sequential(*list(features.children()), cbam)
        logging.info("✓ CBAM attached")
    
    # Build model
    model = nn.Module()
    model.features = features
    
    if CONFIG['use_better_head']:
        model.head = EnhancedHead(C, num_classes, dropout=0.3)
        logging.info("✓ Enhanced head")
    else:
        model.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C, num_classes)
        )
    
    def _forward(self, x):
        x = self.features(x)
        return self.head(x)
    
    model.forward = _forward.__get__(model, type(model))
    return model

# %% [markdown]
# ## PHASE 1: Data Collection

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
                        'source_path': path
                    })
    
    df = pd.DataFrame(all_data)
    logging.info(f"\nTotal: {len(df)} images")
    logging.info(f"\nBy label:\n{df.groupby('label_name').size()}")
    
    return df

collected_df = auto_collect_dataset()
collected_df.to_csv(os.path.join(OUTPUT_DIRS["results"], "collected_images.csv"), index=False)

# %% [markdown]
# ## PHASE 2: Prepare Dataset for YOLO (1-class LEAF only)

# %%
def detect_image_type(img_path, edge_threshold=0.15):
    """Detect if image is single leaf or cluster"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return 'unknown'
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        edge_width = int(w * 0.1)
        edge_height = int(h * 0.1)
        
        top_edge = gray[:edge_height, :]
        bottom_edge = gray[h-edge_height:, :]
        left_edge = gray[:, :edge_width]
        right_edge = gray[:, w-edge_width:]
        
        edge_std = np.mean([
            np.std(top_edge),
            np.std(bottom_edge),
            np.std(left_edge),
            np.std(right_edge)
        ])
        
        center_std = np.std(gray[edge_height:h-edge_height, edge_width:w-edge_width])
        
        if center_std > 0:
            edge_ratio = edge_std / center_std
            if edge_ratio < edge_threshold:
                return 'single_leaf'
        
        return 'cluster'
        
    except Exception as e:
        logging.warning(f"Error detecting type for {img_path}: {e}")
        return 'unknown'

def create_pseudo_labels_for_cluster(img_path, class_id=0):
    """Create pseudo bboxes for YOLO (class_id=0 for 'leaf')"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return []
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([25, 30, 30])
        upper_green = np.array([90, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        min_area = (w * h) * 0.01
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            x, y, box_w, box_h = cv2.boundingRect(contour)
            
            aspect_ratio = box_w / box_h if box_h > 0 else 0
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                continue
            
            x_center = (x + box_w / 2) / w
            y_center = (y + box_h / 2) / h
            norm_w = box_w / w
            norm_h = box_h / h
            
            bboxes.append((class_id, x_center, y_center, norm_w, norm_h))
        
        if len(bboxes) == 0:
            bboxes = [(class_id, 0.5, 0.5, 1.0, 1.0)]
        
        return bboxes
        
    except Exception as e:
        logging.warning(f"Error creating pseudo labels: {e}")
        return [(class_id, 0.5, 0.5, 1.0, 1.0)]

def prepare_field_dataset(df, output_dir, val_size=0.15, test_size=0.15):
    """Prepare dataset for YOLO - 1 class 'leaf' only"""
    logging.info("\n" + "="*60)
    logging.info("DATASET PREPARATION FOR YOLO (1-CLASS LEAF)")
    logging.info("="*60)
    
    trainval_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df['label_id']
    )
    train_df, val_df = train_test_split(
        trainval_df, test_size=val_size/(1-test_size), random_state=42,
        stratify=trainval_df['label_id']
    )
    
    logging.info(f"\nTrain: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    logging.info(f"Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    logging.info(f"Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    field_data = {'train': [], 'val': [], 'test': []}
    type_stats = {'single_leaf': 0, 'cluster': 0, 'unknown': 0}
    
    for split, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        logging.info(f"\nProcessing {split}...")
        
        for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Preparing {split}"):
            src = row['image_path']
            dst = os.path.join(split_dir, f"{split}_{idx:06d}.jpg")
            
            try:
                shutil.copy2(src, dst)
                
                img_type = detect_image_type(src)
                type_stats[img_type] = type_stats.get(img_type, 0) + 1
                
                label_file = os.path.join(split_dir, f"{split}_{idx:06d}.txt")
                
                # YOLO labels: class_id=0 (leaf only)
                if img_type == 'single_leaf':
                    with open(label_file, 'w') as f:
                        f.write("0 0.5 0.5 1.0 1.0\n")
                else:
                    bboxes = create_pseudo_labels_for_cluster(src, class_id=0)
                    with open(label_file, 'w') as f:
                        for class_id, x_c, y_c, w, h in bboxes:
                            f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                
                # Keep disease label for later classification
                field_data[split].append({
                    'image_path': dst,
                    'label_path': label_file,
                    'label_name': row['label_name'],
                    'label_id': row['label_id'],
                    'image_type': img_type
                })
            except Exception as e:
                logging.warning(f"Error: {e}")
    
    logging.info(f"\n{'='*60}")
    logging.info("Image Type Statistics:")
    for k, v in type_stats.items():
        logging.info(f"  {k:15s}: {v:4d} ({v/len(df)*100:.1f}%)")
    logging.info(f"{'='*60}")
    
    stats_df = pd.DataFrame([type_stats])
    stats_df.to_csv(os.path.join(output_dir, 'image_type_stats.csv'), index=False)
    
    return field_data

def create_yolo_yaml(data_root, output_path):
    """YOLO YAML for 1-class leaf detection"""
    import yaml
    yaml_data = {
        'path': str(Path(data_root).absolute()),
        'train': 'train',
        'val': 'val',
        'test': 'test',
        'nc': 1,
        'names': ['leaf']
    }
    yaml_path = os.path.join(output_path, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    logging.info(f"Created YOLO YAML (1-class leaf): {yaml_path}")
    return yaml_path

field_data = prepare_field_dataset(
    collected_df, OUTPUT_DIRS["field_images"], 
    val_size=CONFIG['val_size'], test_size=CONFIG['test_size']
)
yaml_path = create_yolo_yaml(OUTPUT_DIRS["field_images"], OUTPUT_DIRS["field_images"])

# %% [markdown]
# ## PHASE 3: YOLO Training (Leaf Detection)

# %%
def train_yolo_detector(yaml_path, output_dir, epochs=30, imgsz=640, batch=16):
    if not YOLO_AVAILABLE:
        logging.error("YOLO not available")
        return None
    
    logging.info("\n" + "="*60)
    logging.info("YOLO TRAINING (LEAF DETECTION)")
    logging.info("="*60)
    
    yolo_device = '0' if torch.cuda.is_available() else 'cpu'
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data=yaml_path, epochs=epochs, imgsz=imgsz, batch=batch,
        device=yolo_device, patience=10, save_period=5, workers=4,
        project=output_dir, name='detector', exist_ok=True, verbose=True, plots=True
    )
    
    best_model = Path(output_dir) / 'detector' / 'weights' / 'best.pt'
    logging.info(f"✓ Best YOLO model: {best_model}")
    return str(best_model)

best_yolo_model = train_yolo_detector(
    yaml_path=yaml_path, output_dir=OUTPUT_DIRS["yolo_weights"],
    epochs=CONFIG['yolo_epochs'], imgsz=CONFIG['yolo_imgsz'], batch=CONFIG['yolo_batch']
)

# %% [markdown]
# ## PHASE 4: Extract Crops with Lesion-Aware Filtering

# %%
def lesion_score_rgb(crop_rgb: np.ndarray) -> float:
    """
    Score lesion presence in crop [0,1]
    High score = likely diseased
    """
    if crop_rgb is None or crop_rgb.size == 0:
        return 0.0
    h, w = crop_rgb.shape[:2]
    if h < CONFIG['min_crop_size'] or w < CONFIG['min_crop_size']:
        return 0.0
    
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    
    # Brown/yellow lesions
    lower_brown = np.array([5, 40, 40])
    upper_brown = np.array([30, 255, 220])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Dark necrotic regions
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 60])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    
    lesion = cv2.bitwise_or(mask_brown, mask_dark)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    lesion_ratio = lesion.mean() / 255.0
    
    # Texture variance
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    texture_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    texture_score = min(texture_var / 1000.0, 1.0)
    
    return float(0.7 * lesion_ratio + 0.3 * texture_score)

def visualize_crop_samples(model, field_images, output_dir, n_samples=6):
    """Visualize crop extraction with lesion scores"""
    logging.info("\n" + "="*60)
    logging.info("CREATING CROP VISUALIZATION SAMPLES")
    logging.info("="*60)
    
    samples_dir = os.path.join(output_dir, "crop_samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    train_samples = random.sample(field_images['train'], min(n_samples, len(field_images['train'])))
    
    for sample_idx, item in enumerate(train_samples):
        img_path = item['image_path']
        parent_label = item['label_name']
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            results = model.predict(img_path, conf=0.25, verbose=False)
            
            crops = []
            img_with_boxes = img_rgb.copy()
            
            for idx, result in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, result.xyxy[0].cpu().numpy())
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                padding = 10
                h, w = img_rgb.shape[:2]
                x1_p = max(0, x1 - padding)
                y1_p = max(0, y1 - padding)
                x2_p = min(w, x2 + padding)
                y2_p = min(h, y2 + padding)
                
                crop = img_rgb[y1_p:y2_p, x1_p:x2_p]
                if crop.size > 0 and crop.shape[0] >= 50 and crop.shape[1] >= 50:
                    score = lesion_score_rgb(crop)
                    crops.append((crop, score))
            
            if len(crops) == 0:
                continue
            
            n_crops = len(crops)
            fig = plt.figure(figsize=(16, 10))
            
            ax = plt.subplot(2, 4, (1, 5))
            ax.imshow(img_with_boxes)
            ax.set_title(f'Original (Label: {parent_label})\n{n_crops} crops detected', 
                        fontsize=12, fontweight='bold')
            ax.axis('off')
            
            for i, (crop, score) in enumerate(crops[:6]):
                ax = plt.subplot(2, 4, i+2 if i < 3 else i+3)
                ax.imshow(crop)
                ax.set_title(f'Crop {i+1}\nLesion: {score:.3f}', fontsize=10)
                ax.axis('off')
            
            plt.suptitle(f'Sample {sample_idx+1}: Lesion-Aware Crop Filtering', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            save_path = os.path.join(samples_dir, f'crop_sample_{sample_idx+1}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logging.info(f"  ✓ Sample {sample_idx+1}: {n_crops} crops")
            
        except Exception as e:
            logging.warning(f"Error creating sample {sample_idx+1}: {e}")

def extract_crops_with_lesion_filtering(yolo_model_path, field_images, output_dir, confidence=0.25):
    """
    Extract crops and filter by lesion score to reduce label noise
    - Diseased images: keep top-K lesion crops
    - Healthy images: keep top-K low-lesion crops
    """
    logging.info("\n" + "="*60)
    logging.info("EXTRACTING CROPS WITH LESION-AWARE FILTERING")
    logging.info("="*60)
    logging.info(f"Strategy: Keep top-{CONFIG['crops_per_image']} crops per image")
    
    model = YOLO(yolo_model_path)
    crops_data = []
    
    visualize_crop_samples(model, field_images, output_dir, n_samples=6)
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        logging.info(f"\nProcessing {split}...")
        
        for item in tqdm(field_images[split], desc=f"Extracting {split}"):
            img_path = item['image_path']
            parent_label_id = item['label_id']
            parent_label_name = item['label_name']
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                results = model.predict(img_path, conf=confidence, verbose=False)
                base_name = Path(img_path).stem
                
                # Collect all proposals with lesion scores
                proposals = []
                for result in results[0].boxes:
                    x1, y1, x2, y2 = map(int, result.xyxy[0].cpu().numpy())
                    
                    pad = 10
                    H, W = img_rgb.shape[:2]
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(W, x2 + pad)
                    y2 = min(H, y2 + pad)
                    
                    crop = img_rgb[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    
                    score = lesion_score_rgb(crop)
                    proposals.append((score, crop))
                
                if len(proposals) == 0:
                    continue
                
                K = CONFIG['crops_per_image']
                
                # Filter by lesion score based on parent label
                if parent_label_name == 'healthy':
                    # Keep K crops with LOWEST lesion scores
                    proposals = sorted(proposals, key=lambda x: x[0])[:K]
                else:
                    # Keep K crops with HIGHEST lesion scores
                    proposals = sorted(proposals, key=lambda x: x[0], reverse=True)[:K]
                
                # Save filtered crops
                for idx, (score, crop) in enumerate(proposals):
                    if crop.shape[0] < CONFIG['min_crop_size'] or crop.shape[1] < CONFIG['min_crop_size']:
                        continue
                    
                    crop_filename = f"{base_name}_crop{idx:03d}.jpg"
                    crop_path = os.path.join(split_dir, crop_filename)
                    Image.fromarray(crop).save(crop_path, quality=95)
                    
                    crops_data.append({
                        'crop_path': crop_path,
                        'parent_image': img_path,
                        'split': split,
                        'label_id': parent_label_id,
                        'label_name': parent_label_name,
                        'lesion_score': score
                    })
                    
            except Exception as e:
                logging.error(f"Error: {e}")
    
    crops_df = pd.DataFrame(crops_data)
    crops_csv = os.path.join(output_dir, "crops_metadata.csv")
    crops_df.to_csv(crops_csv, index=False)
    
    logging.info(f"\n✓ Extracted {len(crops_df)} filtered crops")
    for split in ['train', 'val', 'test']:
        count = len(crops_df[crops_df['split']==split])
        logging.info(f"  {split}: {count}")
    logging.info(f"\nLabel distribution:\n{crops_df.groupby(['split', 'label_name']).size()}")
    logging.info(f"\nLesion score stats by label:")
    for label in crops_df['label_name'].unique():
        scores = crops_df[crops_df['label_name']==label]['lesion_score']
        logging.info(f"  {label:15s}: mean={scores.mean():.3f}, std={scores.std():.3f}")
    
    return crops_df

crops_df = extract_crops_with_lesion_filtering(
    yolo_model_path=best_yolo_model,
    field_images=field_data,
    output_dir=OUTPUT_DIRS["crops"],
    confidence=CONFIG['yolo_conf']
)

# %% [markdown]
# ## PHASE 5: Data Analysis Report (KEEP ALL CHARTS)

# %%
def create_data_analysis_report(collected_df, crops_df, output_dir):
    logging.info("\n" + "="*60)
    logging.info("CREATING DATA ANALYSIS REPORT")
    logging.info("="*60)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Original images distribution
    ax1 = plt.subplot(3, 3, 1)
    label_counts = collected_df.groupby('label_name').size().sort_values(ascending=True)
    colors = plt.cm.Set3(range(len(label_counts)))
    label_counts.plot(kind='barh', ax=ax1, color=colors)
    ax1.set_xlabel('Number of Images', fontsize=10)
    ax1.set_title('Original Images per Disease Class', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Crops distribution
    ax2 = plt.subplot(3, 3, 2)
    crop_counts = crops_df.groupby('label_name').size().sort_values(ascending=True)
    crop_counts.plot(kind='barh', ax=ax2, color=colors)
    ax2.set_xlabel('Number of Crops', fontsize=10)
    ax2.set_title('Filtered Crops per Disease Class', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Crops per split
    ax3 = plt.subplot(3, 3, 3)
    split_counts = crops_df.groupby('split').size()
    split_colors = ['steelblue', 'coral', 'lightgreen']
    ax3.bar(split_counts.index, split_counts.values, color=split_colors)
    ax3.set_ylabel('Number of Crops', fontsize=10)
    ax3.set_title('Crops per Split', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(split_counts.values):
        ax3.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 4. Distribution by split and class
    ax4 = plt.subplot(3, 3, 4)
    pivot = crops_df.groupby(['label_name', 'split']).size().unstack(fill_value=0)
    pivot.plot(kind='bar', stacked=True, ax=ax4, color=split_colors)
    ax4.set_xlabel('Disease Class', fontsize=10)
    ax4.set_ylabel('Number of Crops', fontsize=10)
    ax4.set_title('Class Distribution Across Splits', fontsize=11, fontweight='bold')
    ax4.legend(title='Split', loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 5. Pie chart
    ax5 = plt.subplot(3, 3, 5)
    overall_counts = crops_df.groupby('label_name').size()
    ax5.pie(overall_counts.values, labels=overall_counts.index, autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax5.set_title('Overall Crop Distribution', fontsize=11, fontweight='bold')
    
    # 6. Lesion score distribution
    ax6 = plt.subplot(3, 3, 6)
    for label in crops_df['label_name'].unique():
        scores = crops_df[crops_df['label_name']==label]['lesion_score']
        ax6.hist(scores, bins=20, alpha=0.6, label=label)
    ax6.set_xlabel('Lesion Score', fontsize=10)
    ax6.set_ylabel('Frequency', fontsize=10)
    ax6.set_title('Lesion Score Distribution by Class', fontsize=11, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Crops per parent image stats
    ax7 = plt.subplot(3, 3, 7)
    crops_per_parent = crops_df.groupby('parent_image').size()
    ax7.hist(crops_per_parent.values, bins=range(1, max(crops_per_parent.values)+2), 
            color='steelblue', alpha=0.7, edgecolor='black')
    ax7.axvline(crops_per_parent.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {crops_per_parent.mean():.1f}')
    ax7.set_xlabel('Crops per Parent Image', fontsize=10)
    ax7.set_ylabel('Frequency', fontsize=10)
    ax7.set_title('Filtered Crops per Parent Image', fontsize=11, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Class balance
    ax8 = plt.subplot(3, 3, 8)
    train_dist = crops_df[crops_df['split']=='train'].groupby('label_name').size()
    val_dist = crops_df[crops_df['split']=='val'].groupby('label_name').size()
    test_dist = crops_df[crops_df['split']=='test'].groupby('label_name').size()
    
    x = np.arange(len(train_dist))
    width = 0.25
    ax8.bar(x - width, train_dist.values, width, label='Train', color='steelblue')
    ax8.bar(x, val_dist.values, width, label='Val', color='coral')
    ax8.bar(x + width, test_dist.values, width, label='Test', color='lightgreen')
    
    ax8.set_xlabel('Disease Class', fontsize=10)
    ax8.set_ylabel('Number of Samples', fontsize=10)
    ax8.set_title('Class Balance Across Splits', fontsize=11, fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_xticklabels(train_dist.index, rotation=45, ha='right')
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    stats_text = f"""
    DATA SUMMARY STATISTICS
    ═══════════════════════════
    
    Original Images: {len(collected_df):,}
    Total Crops: {len(crops_df):,}
    Classes: {len(crops_df['label_name'].unique())}
    
    Split Distribution:
      • Train: {len(crops_df[crops_df['split']=='train']):,} ({len(crops_df[crops_df['split']=='train'])/len(crops_df)*100:.1f}%)
      • Val:   {len(crops_df[crops_df['split']=='val']):,} ({len(crops_df[crops_df['split']=='val'])/len(crops_df)*100:.1f}%)
      • Test:  {len(crops_df[crops_df['split']=='test']):,} ({len(crops_df[crops_df['split']=='test'])/len(crops_df)*100:.1f}%)
    
    Avg Crops/Image: {crops_per_parent.mean():.2f}
    
    Optimizations:
      ✓ YOLO 1-class (leaf)
      ✓ Lesion-aware filtering
      ✓ {CONFIG['backbone']}
      ✓ SCE Loss + EMA
      ✓ Weighted Sampler
      ✓ Logit Adjustment
    """
    ax9.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Comprehensive Data Analysis Report', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "data_analysis_report.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    logging.info(f"✓ Data analysis report saved: {save_path}")

create_data_analysis_report(collected_df, crops_df, OUTPUT_DIRS["plots"])

# %% [markdown]
# ## PHASE 6: Robust Training Components

# %%
# ===== SYMMETRIC CROSS-ENTROPY =====
class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1, beta=1.0, num_classes=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
    
    def forward(self, logits, targets):
        # Standard CE
        ce = F.cross_entropy(logits, targets)
        
        # Reverse CE
        pred = F.softmax(logits, dim=1).clamp(min=1e-7, max=1-1e-7)
        onehot = F.one_hot(targets, self.num_classes).float()
        rce = (-torch.sum(onehot * torch.log(pred), dim=1)).mean()
        
        return self.alpha * ce + self.beta * rce

# ===== LOGIT ADJUSTMENT =====
def compute_priors(df, n_classes):
    counts = df['label_id'].value_counts().sort_index().values.astype(float)
    priors = torch.tensor(counts / counts.sum(), dtype=torch.float32)
    return priors

def apply_logit_adjustment(logits, priors, tau=1.0):
    return logits + tau * torch.log(priors.to(logits.device))

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

# ===== DATASET =====
class CropDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["crop_path"]).convert("RGB")
        label = int(row["label_id"])
        if self.transform:
            img = self.transform(img)
        return img, label

# ===== TRANSFORMS =====
def get_transforms(size):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform

# ===== DATA LOADERS =====
def make_loader(df, transform, batch_size, train=True):
    dataset = CropDataset(df, transform=transform)
    
    if train and CONFIG['use_weighted_sampler']:
        counts = df['label_id'].value_counts().sort_index().values.astype(float)
        class_weights = 1.0 / (counts + 1e-6)
        sample_weights = df['label_id'].map({i:w for i,w in enumerate(class_weights)}).values
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        loader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'],
            persistent_workers=CONFIG['persistent_workers']
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=train,
            num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'],
            persistent_workers=CONFIG['persistent_workers']
        )
    
    return loader

# %% [markdown]
# ## PHASE 7: Optimized Training

# %%
def setup_amp():
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        from contextlib import nullcontext
        amp_ctx = nullcontext()
    else:
        if _NEW_AMP:
            amp_ctx = autocast(device_type="cuda", enabled=True)
        else:
            amp_ctx = autocast(enabled=True)
    
    if _NEW_AMP:
        scaler = GradScaler(device="cuda" if use_cuda else "cpu", enabled=use_cuda)
    else:
        scaler = GradScaler(enabled=use_cuda)
    
    # Clear GPU cache and optimize memory
    if use_cuda:
        # Set memory allocator to avoid fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Aggressive cleanup
        cleanup_memory()
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        
        logging.info(f"GPU Memory before training: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
        logging.info(f"Max memory to use: {CONFIG.get('max_memory_allocated', 15.0):.1f}GB (leaving 7GB buffer for system)")
    
    return DEVICE, amp_ctx, scaler

def log_gpu_memory(epoch=None, step=None):
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        max_mem = torch.cuda.max_memory_allocated() / 1024**3
        
        prefix = ""
        if epoch is not None:
            prefix += f"Epoch {epoch}"
        if step is not None:
            prefix += f" Step {step}"
        if prefix:
            prefix += ": "
            
        logging.info(f"{prefix}GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB, Max: {max_mem:.2f}GB")
        
        # Warning if memory usage is high
        if allocated > CONFIG.get('max_memory_allocated', 15.0):
            logging.warning(f"High GPU memory usage: {allocated:.2f}GB > {CONFIG.get('max_memory_allocated', 15.0)}GB")

def cleanup_memory():
    """Aggressive memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def check_memory_and_cleanup(batch_idx):
    """Check memory usage and cleanup if needed"""
    if torch.cuda.is_available() and batch_idx > 0:
        if batch_idx % CONFIG.get('memory_cleanup_frequency', 3) == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            if allocated > CONFIG.get('max_memory_allocated', 15.0) * 0.8:  # 80% threshold
                cleanup_memory()
                logging.info(f"Memory cleanup at batch {batch_idx}, was {allocated:.2f}GB")

def train_optimized_model(model, train_df, val_df, epochs):
    device, amp_ctx, scaler = setup_amp()
    model = model.to(device)
    
    num_classes = len(LABELS)
    priors = compute_priors(train_df, num_classes)
    logging.info(f"Class priors: {priors.tolist()}")
    
    # Loss
    if CONFIG['use_sce_loss']:
        criterion = SymmetricCrossEntropy(CONFIG['sce_alpha'], CONFIG['sce_beta'], num_classes)
        logging.info("✓ Using Symmetric Cross-Entropy")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # EMA
    ema = ModelEMA(model, decay=CONFIG['ema_decay']) if CONFIG['use_ema'] else None
    if ema:
        logging.info("✓ Using EMA")
    
    best_val_acc = 0.0
    patience = 10
    bad_epochs = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_path = os.path.join(OUTPUT_DIRS["weights"], "best_classifier.pth")
    
    current_size = CONFIG['img_size']
    
    logging.info(f"\nTraining {CONFIG['backbone']}...")
    logging.info(f"Enhancements: CBAM={CONFIG['use_cbam']}, EnhancedHead={CONFIG['use_better_head']}")
    logging.info(f"Robust: SCE={CONFIG['use_sce_loss']}, WeightedSampler={CONFIG['use_weighted_sampler']}, EMA={CONFIG['use_ema']}")
    
    for epoch in range(1, epochs + 1):
        # Progressive resize with memory check
        if CONFIG['use_progressive_resize'] and epoch in CONFIG['progressive_schedule']:
            new_size = CONFIG['progressive_schedule'][epoch]
            
            # Check if we can afford larger images
            current_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            if new_size > current_size and current_mem > CONFIG.get('max_memory_allocated', 15.0) * 0.6:  # 60% threshold
                logging.warning(f"Skipping resize to {new_size} due to memory constraints ({current_mem:.1f}GB)")
            else:
                current_size = new_size
                logging.info(f"\n→ Progressive resize to {current_size}x{current_size}")
                # Cleanup after resize change
                cleanup_memory()
        
        # Prepare loaders
        train_transform, val_transform = get_transforms(current_size)
        train_loader = make_loader(train_df, train_transform, CONFIG['batch_size'], train=True)
        val_loader = make_loader(val_df, val_transform, CONFIG['batch_size'], train=False)
        
        # ===== TRAIN =====
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        
        # Gradient accumulation setup
        accumulation_steps = CONFIG['gradient_accumulation_steps']
        effective_batch_size = CONFIG['batch_size'] * accumulation_steps
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} (BS:{effective_batch_size})")
        for batch_idx, (imgs, labels) in enumerate(pbar):
            imgs = imgs.to(device)
            labels = labels.long().to(device)
            
            # Only zero gradients at start of accumulation
            if batch_idx % accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            
            with amp_ctx:
                logits = model(imgs)
                if CONFIG['use_logit_adjustment']:
                    logits = apply_logit_adjustment(logits, priors, CONFIG['logit_adjustment_tau'])
                loss = criterion(logits, labels)
                
                # Scale loss by accumulation steps
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            # Only step optimizer every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                
                if ema is not None:
                    ema.update(model)
                    
                optimizer.zero_grad(set_to_none=True)
            
            # Track metrics (remember to scale back loss for reporting)
            epoch_loss += (loss.item() * accumulation_steps) * imgs.size(0)
            pred = logits.argmax(1)
            correct += (pred == labels).sum().item()
            total += imgs.size(0)
            
            # Update progress bar with memory info
            current_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}', 
                'mem': f'{current_mem:.1f}GB'
            })
            
            # Periodic memory cleanup
            check_memory_and_cleanup(batch_idx)
        
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
                    if CONFIG['use_logit_adjustment']:
                        logits = apply_logit_adjustment(logits, priors, CONFIG['logit_adjustment_tau'])
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
        
        logging.info(f"Epoch {epoch}: TL={train_loss:.4f} TA={train_acc:.4f} | VL={val_loss:.4f} VA={val_acc:.4f}")
        
        # Log GPU memory usage periodically
        if epoch % 5 == 0 or epoch == 1:
            log_gpu_memory(epoch=epoch)
            
        # Aggressive memory cleanup
        cleanup_memory()  # Clean after every epoch
        
        # Extra cleanup for safety
        if epoch % 3 == 0:
            torch.cuda.synchronize()
            gc.collect()
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            bad_epochs = 0
            torch.save(eval_model.state_dict(), best_path)
            logging.info(f"  → Saved best (Val Acc {val_acc:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logging.info("Early stopping triggered")
                break
    
    return history, best_path

# Train
train_crops = crops_df[crops_df['split'] == 'train']
val_crops = crops_df[crops_df['split'] == 'val']

model = build_enhanced_classifier(len(LABELS))
history, best_checkpoint = train_optimized_model(model, train_crops, val_crops, CONFIG['epochs'])

# %% [markdown]
# ## PHASE 8: Training Report (KEEP ALL CHARTS)

# %%
def create_training_report(history, model_name, output_dir):
    logging.info("\n" + "="*60)
    logging.info("CREATING TRAINING REPORT")
    logging.info("="*60)
    
    fig = plt.figure(figsize=(18, 12))
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss curves
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(epochs_range, history['train_loss'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs_range, history['val_loss'], 'r-', label='Val', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.set_title('Training & Validation Loss', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy curves
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(epochs_range, history['train_acc'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs_range, history['val_acc'], 'r-', label='Val', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.set_title('Training & Validation Accuracy', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Loss zoomed
    ax3 = plt.subplot(3, 3, 3)
    best_epoch = np.argmin(history['val_loss']) + 1
    start = max(0, best_epoch - 10)
    end = min(len(epochs_range), best_epoch + 5)
    ax3.plot(epochs_range[start:end], history['train_loss'][start:end], 'b-', label='Train', linewidth=2)
    ax3.plot(epochs_range[start:end], history['val_loss'][start:end], 'r-', label='Val', linewidth=2)
    ax3.axvline(best_epoch, color='green', linestyle='--', linewidth=2, label=f'Best: {best_epoch}')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Loss (Zoomed)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Accuracy improvement
    ax4 = plt.subplot(3, 3, 4)
    train_acc_improvement = np.diff([0] + history['train_acc'])
    val_acc_improvement = np.diff([0] + history['val_acc'])
    ax4.plot(epochs_range, train_acc_improvement, 'b-', label='Train Δ', linewidth=2)
    ax4.plot(epochs_range, val_acc_improvement, 'r-', label='Val Δ', linewidth=2)
    ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy Change')
    ax4.set_title('Epoch-to-Epoch Improvement')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Overfitting gap
    ax5 = plt.subplot(3, 3, 5)
    acc_gap = np.array(history['train_acc']) - np.array(history['val_acc'])
    ax5.plot(epochs_range, acc_gap, 'purple', label='Acc Gap (Train-Val)', linewidth=2)
    ax5.axhline(0, color='black', linestyle='--', linewidth=1)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Gap')
    ax5.set_title('Overfitting Monitor')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Val trajectory
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Loss')
    ax6.set_title('Val Loss Trajectory')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Metrics summary
    ax7 = plt.subplot(3, 3, 7)
    ax7.axis('off')
    best_val_loss_epoch = np.argmin(history['val_loss']) + 1
    best_val_acc_epoch = np.argmax(history['val_acc']) + 1
    
    metrics_text = f"""
    TRAINING METRICS SUMMARY
    ════════════════════════════
    
    Model: {model_name}
    Total Epochs: {len(history['train_loss'])}
    
    Best Val Loss:
      Epoch: {best_val_loss_epoch}
      Loss: {min(history['val_loss']):.4f}
      Acc:  {history['val_acc'][best_val_loss_epoch-1]:.4f}
    
    Best Val Accuracy:
      Epoch: {best_val_acc_epoch}
      Acc:  {max(history['val_acc']):.4f}
      Loss: {history['val_loss'][best_val_acc_epoch-1]:.4f}
    
    Final Epoch:
      Train Loss: {history['train_loss'][-1]:.4f}
      Train Acc:  {history['train_acc'][-1]:.4f}
      Val Loss:   {history['val_loss'][-1]:.4f}
      Val Acc:    {history['val_acc'][-1]:.4f}
    
    Optimizations:
      Backbone: {CONFIG['backbone']}
      SCE Loss: {CONFIG['use_sce_loss']}
      EMA: {CONFIG['use_ema']}
      Weighted Sampler: {CONFIG['use_weighted_sampler']}
    """
    ax7.text(0.1, 0.5, metrics_text, fontsize=9, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # 8. Stability
    ax8 = plt.subplot(3, 3, 8)
    window = 3
    if len(history['val_loss']) >= window:
        val_loss_rolling_std = pd.Series(history['val_loss']).rolling(window=window).std()
        ax8.plot(epochs_range, val_loss_rolling_std, 'orange', linewidth=2)
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Rolling Std')
        ax8.set_title(f'Stability (Window={window})')
        ax8.grid(True, alpha=0.3)
    
    # 9. Final metrics
    ax9 = plt.subplot(3, 3, 9)
    metrics_names = ['Train Acc', 'Val Acc', 'Train Loss', 'Val Loss']
    final_values = [history['train_acc'][-1], history['val_acc'][-1],
                   history['train_loss'][-1], history['val_loss'][-1]]
    colors_bar = ['steelblue', 'coral', 'lightblue', 'salmon']
    bars = ax9.barh(metrics_names, final_values, color=colors_bar)
    ax9.set_xlabel('Value')
    ax9.set_title('Final Epoch Metrics')
    ax9.grid(True, alpha=0.3, axis='x')
    for bar, value in zip(bars, final_values):
        ax9.text(value, bar.get_y() + bar.get_height()/2, f' {value:.4f}',
                va='center', fontweight='bold')
    
    plt.suptitle(f'{model_name} - Training Report', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'training_report.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    logging.info(f"✓ Training report saved: {save_path}")

create_training_report(history, f"Optimized_{CONFIG['backbone']}", OUTPUT_DIRS["plots"])

# %% [markdown]
# ## PHASE 9: Test with TTA

# %%
def test_with_tta(model, test_df, device):
    """Test with Test-Time Augmentation"""
    model = model.to(device)
    model.eval()
    
    tta_transforms = [
        transforms.Compose([
            transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
            transforms.RandomRotation((90, 90)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
            transforms.RandomRotation((180, 180)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    ]
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing with TTA"):
            img = Image.open(row["crop_path"]).convert("RGB")
            label = int(row["label_id"])
            
            outputs_list = []
            for transform in tta_transforms[:CONFIG['tta_transforms']]:
                img_tensor = transform(img).unsqueeze(0).to(device)
                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                outputs_list.append(probs)
            
            avg_probs = torch.stack(outputs_list).mean(0)
            pred = avg_probs.argmax(1)
            
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label)
    
    y_pred = np.concatenate(all_preds)
    y_true = np.array(all_labels)
    
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1, 
            'confusion_matrix': cm, 'y_true': y_true, 'y_pred': y_pred}

# Test
test_crops = crops_df[crops_df['split'] == 'test']
model.load_state_dict(torch.load(best_checkpoint, map_location=DEVICE))
model = model.to(DEVICE)

if CONFIG['use_tta']:
    logging.info(f"\nTesting with TTA ({CONFIG['tta_transforms']} augmentations)")
    test_metrics = test_with_tta(model, test_crops, DEVICE)
else:
    _, val_transform = get_transforms(CONFIG['img_size'])
    test_loader = make_loader(test_crops, val_transform, CONFIG['batch_size'], train=False)
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            all_preds.append(outputs.argmax(1).cpu().numpy())
            all_labels.append(labels.numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    test_metrics = {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1,
                   'confusion_matrix': cm, 'y_true': y_true, 'y_pred': y_pred}

logging.info("\n" + "="*60)
logging.info("TEST RESULTS")
logging.info("="*60)
logging.info(f"Accuracy:  {test_metrics['accuracy']:.4f}")
logging.info(f"Precision: {test_metrics['precision']:.4f}")
logging.info(f"Recall:    {test_metrics['recall']:.4f}")
logging.info(f"F1 Score:  {test_metrics['f1']:.4f}")

# %% [markdown]
# ## PHASE 10: Test Results Report (KEEP ALL CHARTS)

# %%
def create_test_results_report(test_metrics, class_names, output_dir):
    logging.info("\n" + "="*60)
    logging.info("CREATING TEST RESULTS REPORT")
    logging.info("="*60)
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Confusion Matrix - Count
    ax1 = plt.subplot(2, 3, 1)
    sns.heatmap(test_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted', fontsize=10)
    ax1.set_ylabel('True', fontsize=10)
    ax1.set_title('Confusion Matrix (Count)', fontsize=11, fontweight='bold')
    
    # 2. Confusion Matrix - Normalized
    ax2 = plt.subplot(2, 3, 2)
    cm_norm = test_metrics['confusion_matrix'].astype('float') / test_metrics['confusion_matrix'].sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn', vmin=0, vmax=1,
                xticklabels=class_names, yticklabels=class_names, ax=ax2, cbar_kws={'label': 'Proportion'})
    ax2.set_xlabel('Predicted', fontsize=10)
    ax2.set_ylabel('True', fontsize=10)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=11, fontweight='bold')
    
    # 3. Per-class metrics
    ax3 = plt.subplot(2, 3, 3)
    p, r, f1, support = precision_recall_fscore_support(
        test_metrics['y_true'], test_metrics['y_pred'], labels=range(len(class_names))
    )
    
    x = np.arange(len(class_names))
    width = 0.2
    ax3.bar(x - width, p, width, label='Precision', alpha=0.8, color='steelblue')
    ax3.bar(x, r, width, label='Recall', alpha=0.8, color='coral')
    ax3.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='lightgreen')
    
    ax3.set_xlabel('Disease Class', fontsize=10)
    ax3.set_ylabel('Score', fontsize=10)
    ax3.set_title('Per-Class Performance Metrics', fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_names, rotation=45, ha='right')
    ax3.set_ylim(0, 1.1)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Samples per class
    ax4 = plt.subplot(2, 3, 4)
    unique, counts = np.unique(test_metrics['y_true'], return_counts=True)
    class_names_ordered = [class_names[i] for i in unique]
    ax4.barh(class_names_ordered, counts, color='mediumpurple')
    ax4.set_xlabel('Number of Samples', fontsize=10)
    ax4.set_title('Test Set Class Distribution', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(counts):
        ax4.text(v, i, f' {v}', va='center', fontweight='bold')
    
    # 5. Accuracy per class
    ax5 = plt.subplot(2, 3, 5)
    class_accuracies = []
    for i in range(len(class_names)):
        mask = test_metrics['y_true'] == i
        if mask.sum() > 0:
            acc = (test_metrics['y_pred'][mask] == i).sum() / mask.sum()
            class_accuracies.append(acc)
        else:
            class_accuracies.append(0)
    
    colors_acc = ['green' if acc >= 0.8 else 'orange' if acc >= 0.6 else 'red' for acc in class_accuracies]
    bars = ax5.barh(class_names, class_accuracies, color=colors_acc)
    ax5.set_xlabel('Accuracy', fontsize=10)
    ax5.set_title('Per-Class Accuracy', fontsize=11, fontweight='bold')
    ax5.set_xlim(0, 1.1)
    ax5.grid(True, alpha=0.3, axis='x')
    
    for bar, acc in zip(bars, class_accuracies):
        ax5.text(acc, bar.get_y() + bar.get_height()/2, f' {acc:.1%}',
                va='center', fontweight='bold')
    
    # 6. Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
    TEST SET RESULTS SUMMARY
    ════════════════════════════════
    
    Overall Metrics:
      • Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)
      • Precision: {test_metrics['precision']:.4f}
      • Recall:    {test_metrics['recall']:.4f}
      • F1-Score:  {test_metrics['f1']:.4f}
    
    Total Samples: {len(test_metrics['y_true'])}
    Correct:       {(test_metrics['y_true'] == test_metrics['y_pred']).sum()}
    Incorrect:     {(test_metrics['y_true'] != test_metrics['y_pred']).sum()}
    
    Per-Class Accuracy:
    """
    
    for i, (name, acc) in enumerate(zip(class_names, class_accuracies)):
        summary_text += f"\n    {name:15s}: {acc:.4f}"
    
    summary_text += f"""
    
    Best Class:  {class_names[np.argmax(class_accuracies)]} ({max(class_accuracies):.1%})
    Worst Class: {class_names[np.argmin(class_accuracies)]} ({min(class_accuracies):.1%})
    
    Optimizations:
      Backbone: {CONFIG['backbone']}
      SCE Loss: {CONFIG['use_sce_loss']}
      EMA: {CONFIG['use_ema']}
      TTA: {CONFIG['use_tta']} ({CONFIG['tta_transforms']} aug)
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    plt.suptitle('Comprehensive Test Results Report', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "test_results_report.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    logging.info(f"✓ Test results report saved: {save_path}")

class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
create_test_results_report(test_metrics, class_names, OUTPUT_DIRS["plots"])

# Classification report
report = classification_report(test_metrics['y_true'], test_metrics['y_pred'], 
                               target_names=class_names, digits=4)
report_path = os.path.join(OUTPUT_DIRS["results"], 'test_classification_report.txt')
with open(report_path, 'w') as f:
    f.write(f"Optimized {CONFIG['backbone']} Test Results\n")
    f.write("="*60 + "\n\n")
    f.write(f"Test Accuracy: {test_metrics['accuracy']:.4f}\n\n")
    f.write("Optimizations:\n")
    f.write(f"  Backbone: {CONFIG['backbone']}\n")
    f.write(f"  YOLO: 1-class leaf\n")
    f.write(f"  Lesion filtering: Top-{CONFIG['crops_per_image']} per image\n")
    f.write(f"  SCE Loss: {CONFIG['use_sce_loss']}\n")
    f.write(f"  Weighted Sampler: {CONFIG['use_weighted_sampler']}\n")
    f.write(f"  Logit Adjustment: {CONFIG['use_logit_adjustment']}\n")
    f.write(f"  EMA: {CONFIG['use_ema']}\n")
    f.write(f"  TTA: {CONFIG['use_tta']} ({CONFIG['tta_transforms']} transforms)\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# %% [markdown]
# ## PHASE 11: Export Models

# %%
def export_for_production(model, model_name, save_dir, example_input):
    model.eval()
    
    # TorchScript
    try:
        traced = torch.jit.trace(model, example_input)
        traced_path = os.path.join(save_dir, f'{model_name}_traced.pt')
        torch.jit.save(traced, traced_path)
        logging.info(f"✓ Exported TorchScript: {traced_path}")
    except Exception as e:
        logging.warning(f"TorchScript export failed: {e}")
    
    # ONNX
    try:
        onnx_path = os.path.join(save_dir, f'{model_name}.onnx')
        torch.onnx.export(model, example_input, onnx_path,
                         input_names=['input'], output_names=['output'],
                         dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                         opset_version=11)
        logging.info(f"✓ Exported ONNX: {onnx_path}")
    except Exception as e:
        logging.warning(f"ONNX export failed: {e}")

example_input = torch.randn(1, 3, CONFIG['img_size'], CONFIG['img_size']).to(DEVICE)
export_for_production(model, f"Optimized_{CONFIG['backbone']}", OUTPUT_DIRS["exports"], example_input)

# %% [markdown]
# ## PHASE 12: Inference Class

# %%
class OptimizedRiceFieldAnalyzer:
    def __init__(self, yolo_model_path, classification_model_path, device=DEVICE):
        self.device = device
        self.yolo_model = YOLO(yolo_model_path)
        
        self.classifier = build_enhanced_classifier(len(LABELS))
        self.classifier.load_state_dict(torch.load(classification_model_path, map_location=device))
        self.classifier.to(device)
        self.classifier.eval()
        
        _, self.transform = get_transforms(CONFIG['img_size'])
        self.class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
    
    @torch.no_grad()
    def analyze_field_image(self, image_path, yolo_conf=0.3, use_tta=False):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        yolo_results = self.yolo_model.predict(image_path, conf=yolo_conf, verbose=False)
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        predictions = []
        
        for result in yolo_results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0].cpu().numpy())
            crop = img_rgb[y1:y2, x1:x2]
            
            if crop.size == 0 or crop.shape[0] < 50 or crop.shape[1] < 50:
                continue
            
            crop_pil = Image.fromarray(crop)
            
            if use_tta and CONFIG['use_tta']:
                tta_transforms = [
                    self.transform,
                    transforms.Compose([
                        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
                        transforms.RandomHorizontalFlip(p=1.0),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                ]
                
                outputs_list = []
                for transform in tta_transforms:
                    crop_tensor = transform(crop_pil).unsqueeze(0).to(self.device)
                    output = self.classifier(crop_tensor)
                    probs = F.softmax(output, dim=1)
                    outputs_list.append(probs)
                
                avg_probs = torch.stack(outputs_list).mean(0)
                conf, pred = torch.max(avg_probs, 1)
            else:
                crop_tensor = self.transform(crop_pil).unsqueeze(0).to(self.device)
                output = self.classifier(crop_tensor)
                probs = F.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
            
            pred_class = self.class_names[pred.item()]
            conf_score = conf.item()
            
            predictions.append({
                'class': pred_class,
                'confidence': conf_score,
                'bbox': (x1, y1, x2, y2)
            })
        
        return predictions, img_rgb
    
    def visualize_results(self, image_path, predictions, img_rgb, save_path=None):
        if img_rgb is None or len(predictions) == 0:
            print("No predictions")
            return
        
        img_draw = img_rgb.copy()
        
        for pred in predictions:
            x1, y1, x2, y2 = pred['bbox']
            color = (0, 255, 0) if pred['class'] == 'healthy' else (255, 0, 0)
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 3)
            
            label = f"{pred['class']}: {pred['confidence']:.2f}"
            cv2.putText(img_draw, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(img_draw)
        plt.title(f'Detection: {len(predictions)} leaves', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        # Summary
        disease_counts = {}
        for p in predictions:
            disease_counts[p['class']] = disease_counts.get(p['class'], 0) + 1
        
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total leaves: {len(predictions)}")
        for disease, count in sorted(disease_counts.items()):
            pct = count/len(predictions)*100
            print(f"  {disease:15s}: {count:3d} ({pct:5.1f}%)")
        print("="*60)

analyzer = OptimizedRiceFieldAnalyzer(
    yolo_model_path=best_yolo_model,
    classification_model_path=best_checkpoint,
    device=DEVICE
)

logging.info("\n✓ Optimized Analyzer ready for inference")

# %% [markdown]
# ## PHASE 13: Demo

# %%
test_images = field_data['test']
samples = random.sample(test_images, min(3, len(test_images)))

for i, sample in enumerate(samples):
    image_path = sample['image_path']
    
    print(f"\n{'='*60}")
    print(f"Test Image {i+1}")
    print(f"True label: {sample['label_name']}")
    print(f"{'='*60}")
    
    predictions, img_rgb = analyzer.analyze_field_image(image_path, yolo_conf=0.3, use_tta=False)
    save_path = os.path.join(OUTPUT_DIRS["demo"], f"demo_{i+1}.png")
    analyzer.visualize_results(image_path, predictions, img_rgb, save_path)

# %% [markdown]
# ## Summary

# %%
logging.info("\n" + "="*80)
logging.info("OPTIMIZED PIPELINE COMPLETE")
logging.info("="*80)
logging.info(f"Output: {PATH_OUTPUT}")
logging.info(f"\nModel: {CONFIG['backbone']}")
logging.info(f"\nKey Optimizations:")
logging.info("  ✓ YOLO 1-class (leaf detection only)")
logging.info(f"  ✓ Lesion-aware filtering (top-{CONFIG['crops_per_image']} crops)")
logging.info(f"  ✓ Symmetric Cross-Entropy (alpha={CONFIG['sce_alpha']}, beta={CONFIG['sce_beta']})")
logging.info(f"  ✓ EMA (decay={CONFIG['ema_decay']})")
logging.info("  ✓ Weighted Random Sampler")
logging.info(f"  ✓ Logit Adjustment (tau={CONFIG['logit_adjustment_tau']})")
if CONFIG['use_cbam']:
    logging.info("  ✓ CBAM Attention")
if CONFIG['use_better_head']:
    logging.info("  ✓ Enhanced Head (Dual Pooling)")

logging.info(f"\nFinal Test Results:")
logging.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
logging.info(f"  Precision: {test_metrics['precision']:.4f}")
logging.info(f"  Recall:    {test_metrics['recall']:.4f}")
logging.info(f"  F1 Score:  {test_metrics['f1']:.4f}")

logging.info(f"\nAll Visualizations:")
logging.info("  ✓ Crop extraction samples (with lesion scores)")
logging.info("  ✓ Data analysis report (9 charts)")
logging.info("  ✓ Training report (9 charts)")
logging.info("  ✓ Test results report (6 visualizations)")
logging.info("  ✓ Demo outputs")

logging.info(f"\nExports:")
logging.info(f"  ✓ Best model: {best_checkpoint}")
logging.info(f"  ✓ TorchScript & ONNX in: {OUTPUT_DIRS['exports']}")

logging.info("="*80)
logging.info("Ready for production deployment!")
logging.info("="*80)