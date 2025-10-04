# %% [markdown]
# # YOLO-First Rice Disease Classification - Complete Pipeline
# Auto-collects data from multiple dataset sources

# %%
# ===== IMPORTS =====
import os, shutil, random, cv2, torch, gc, time, math, subprocess, sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
from datetime import datetime
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from contextlib import nullcontext
from typing import Dict, List, Tuple

# PyTorch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
)

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# AMP
try:
    from torch.amp import GradScaler, autocast
    _NEW_AMP = True
except Exception:
    from torch.cuda.amp import GradScaler, autocast
    _NEW_AMP = False

# %%
# ===== SETUP =====
def get_output_folder(parent_dir: str, env_name: str) -> str:
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = f"{env_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir = os.path.join(parent_dir, experiment_id)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

PATH_OUTPUT = get_output_folder("../output", "yolo_first_complete")

def create_output_structure(base_path):
    folders = [
        "field_images",        
        "yolo_weights",        
        "yolo_crops",          
        "labeled_crops",       
        "classification_weights",
        "results",
        "plots",
        "training_curves",
        "evaluation",
        "comparison",
        "logs"
    ]
    
    for folder in folders:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)
    
    return {folder: os.path.join(base_path, folder) for folder in folders}

OUTPUT_DIRS = create_output_structure(PATH_OUTPUT)

def setup_logging(output_path):
    log_file = os.path.join(output_path, "logs", "pipeline.log")
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logging.info(f"Logging initialized: {log_file}")
    return logger

logger = setup_logging(PATH_OUTPUT)

# %%
# ===== DATASET CONFIGURATION =====
LABELS = {
    0: {
        "name": "brown_spot",
        "match_substrings": [
            "../data/rice-disease-dataset/Rice_Leaf_AUG/Brown Spot",
            "../data/rice-leaf-disease-image/Brownspot",
            "../data/rice-leaf-diseases/rice_leaf_diseases/Brown spot",
            "../data/rice-leafs-disease-dataset/RiceLeafsDisease/train/brown_spot",
            "../data/rice-leaf-images/rice_images/_BrownSpot",
            "../data/rice-diseases-image-dataset/RiceDiseaseDataset/train/BrownSpot",
        ]
    },
    1: {
        "name": "leaf_blast",
        "match_substrings": [
            "../data/rice-disease-dataset/Rice_Leaf_AUG/Leaf Blast",
            "../data/rice-leafs-disease-dataset/RiceLeafsDisease/train/leaf_blast",
            "../data/rice-leaf-images/rice_images/_LeafBlast",
            "../data/rice-diseases-image-dataset/RiceDiseaseDataset/train/LeafBlast",
        ]
    },
    2: {
        "name": "leaf_blight",
        "match_substrings": [
            "../data/rice-disease-dataset/Rice_Leaf_AUG/Sheath Blight",
            "../data/rice-leaf-diseases/rice_leaf_diseases/Bacterial leaf blight",
            "../data/rice-leaf-disease-image/Bacterialblight",
            "../data/rice-leafs-disease-dataset/RiceLeafsDisease/train/bacterial_leaf_blight",
        ]
    },
    3: {
        "name": "healthy",
        "match_substrings": [
            "../data/rice-disease-dataset/Rice_Leaf_AUG/Healthy Rice Leaf",
            "../data/rice-leafs-disease-dataset/RiceLeafsDisease/train/healthy",
            "../data/rice-leaf-images/rice_images/_Healthy",
            "../data/rice-diseases-image-dataset/RiceDiseaseDataset/train/Healthy",
        ]
    }
}

DATASET_SOURCES = {
    "rice-disease-dataset": "dataset_1",
    "rice-leaf-disease-image": "dataset_2",
    "rice-leaf-diseases": "dataset_3", 
    "rice-leafs-disease-dataset": "dataset_4",
    "rice-leaf-images": "dataset_5",
    "rice-diseases-image-dataset": "dataset_6"
}

# Device detection with proper validation
def get_device_info():
    """Get device information and validate CUDA setup"""
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if cuda_available else 0
    
    if cuda_available and cuda_count > 0:
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        logging.info(f"CUDA available: {cuda_available}")
        logging.info(f"CUDA devices: {cuda_count}")
        logging.info(f"Device name: {device_name}")
    else:
        device = torch.device("cpu")
        if cuda_available:
            logging.warning("CUDA available but no devices detected, falling back to CPU")
        else:
            logging.info("CUDA not available, using CPU")
    
    return device, cuda_available, cuda_count

DEVICE, CUDA_AVAILABLE, CUDA_COUNT = get_device_info()
logging.info(f"Using device: {DEVICE}")

# %% [markdown]
# ## PHASE 1: YOLO Setup

# %%
def install_ultralytics():
    try:
        import ultralytics
        logging.info("Ultralytics already installed")
        return True
    except ImportError:
        logging.info("Installing ultralytics...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            logging.info("Ultralytics installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Installation failed: {e}")
            return False

YOLO_AVAILABLE = install_ultralytics()

if YOLO_AVAILABLE:
    from ultralytics import YOLO
    import ultralytics
    logging.info(f"YOLO version: {ultralytics.__version__}")

# %% [markdown]
# ## PHASE 2: Auto-Collect Data from Multiple Sources

# %%
def collect_images_from_path(path: str) -> List[str]:
    """Collect all image files from a directory"""
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

def auto_collect_dataset_from_labels():
    """
    Auto-collect images from all paths defined in LABELS
    Returns DataFrame with image paths and metadata
    """
    logging.info("=" * 60)
    logging.info("AUTO-COLLECTING DATASET FROM MULTIPLE SOURCES")
    logging.info("=" * 60)
    
    all_data = []
    
    for label_id, label_info in LABELS.items():
        label_name = label_info['name']
        match_paths = label_info['match_substrings']
        
        logging.info(f"\nCollecting {label_name} (ID: {label_id})...")
        
        for path in match_paths:
            images = collect_images_from_path(path)
            
            if len(images) > 0:
                # Identify dataset source
                dataset_source = "unknown"
                for source_key, source_id in DATASET_SOURCES.items():
                    if source_key in path:
                        dataset_source = source_id
                        break
                
                logging.info(f"  ✓ Found {len(images)} images in {path}")
                logging.info(f"    Source: {dataset_source}")
                
                for img_path in images:
                    all_data.append({
                        'image_path': img_path,
                        'label_id': label_id,
                        'label_name': label_name,
                        'dataset_source': dataset_source,
                        'source_path': path
                    })
            else:
                logging.info(f"  ✗ No images found in {path}")
    
    df = pd.DataFrame(all_data)
    
    logging.info("\n" + "=" * 60)
    logging.info("COLLECTION SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Total images collected: {len(df)}")
    logging.info(f"\nDistribution by label:")
    logging.info(df.groupby(['label_id', 'label_name']).size())
    logging.info(f"\nDistribution by source:")
    logging.info(df.groupby('dataset_source').size())
    
    return df

# Collect all images
collected_df = auto_collect_dataset_from_labels()

# Save collection metadata
collection_csv = os.path.join(OUTPUT_DIRS["results"], "collected_images.csv")
collected_df.to_csv(collection_csv, index=False)
logging.info(f"\nCollection metadata saved: {collection_csv}")

# %% [markdown]
# ## Visualization: Data Distribution

# %%
def plot_data_distribution(df, save_dir):
    """Plot comprehensive data distribution"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dataset Collection Overview', fontsize=16, fontweight='bold')
    
    # 1. Label distribution
    ax = axes[0, 0]
    label_counts = df.groupby('label_name').size().sort_values(ascending=True)
    label_counts.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Number of Images')
    ax.set_title('Images per Disease Category')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 2. Source distribution
    ax = axes[0, 1]
    source_counts = df.groupby('dataset_source').size().sort_values(ascending=False)
    ax.bar(range(len(source_counts)), source_counts.values, color='coral')
    ax.set_xticks(range(len(source_counts)))
    ax.set_xticklabels(source_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Number of Images')
    ax.set_title('Images per Dataset Source')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Stacked bar: Label × Source
    ax = axes[1, 0]
    pivot = df.groupby(['label_name', 'dataset_source']).size().unstack(fill_value=0)
    pivot.plot(kind='bar', stacked=True, ax=ax, colormap='Set3')
    ax.set_xlabel('Disease Category')
    ax.set_ylabel('Number of Images')
    ax.set_title('Label Distribution Across Sources')
    ax.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Pie chart: Overall distribution
    ax = axes[1, 1]
    label_counts = df.groupby('label_name').size()
    colors = plt.cm.Set3(range(len(label_counts)))
    ax.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax.set_title('Overall Label Distribution')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'data_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    logging.info(f"Data distribution plot saved: {save_path}")

plot_data_distribution(collected_df, OUTPUT_DIRS["plots"])

# %% [markdown]
# ## PHASE 3: Prepare Mock Field Dataset

# %%
def prepare_mock_field_dataset_from_collected(df, output_dir, test_size=0.2):
    """
    Convert collected single leaf images to mock field dataset
    Each image treated as a "field image" with full-image YOLO annotation
    """
    logging.info("\n" + "=" * 60)
    logging.info("CREATING MOCK FIELD DATASET")
    logging.info("=" * 60)
    
    # Split train/val
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=42,
        stratify=df['label_id']
    )
    
    logging.info(f"Split: Train={len(train_df)}, Val={len(val_df)}")
    
    field_data = {'train': [], 'val': []}
    
    for split, split_df in [('train', train_df), ('val', val_df)]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        logging.info(f"\nProcessing {split} split...")
        
        for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Creating {split}"):
            # Copy image with new name
            src = row['image_path']
            dst = os.path.join(split_dir, f"{split}_{idx:06d}.jpg")
            
            try:
                shutil.copy2(src, dst)
                
                # Create YOLO annotation (full image bbox)
                label_file = os.path.join(split_dir, f"{split}_{idx:06d}.txt")
                with open(label_file, 'w') as f:
                    # YOLO format: class_id x_center y_center width height (normalized)
                    f.write(f"{row['label_id']} 0.5 0.5 1.0 1.0\n")
                
                field_data[split].append({
                    'image_path': dst,
                    'label_path': label_file,
                    'true_label': row['label_name'],
                    'label_id': row['label_id'],
                    'dataset_source': row['dataset_source']
                })
                
            except Exception as e:
                logging.warning(f"Error processing {src}: {e}")
                continue
    
    logging.info(f"\nMock dataset created:")
    logging.info(f"  Train: {len(field_data['train'])} images")
    logging.info(f"  Val: {len(field_data['val'])} images")
    
    return field_data

def create_yolo_yaml(data_root, output_path):
    """Create data.yaml for YOLO training"""
    import yaml
    
    yaml_data = {
        'path': str(Path(data_root).absolute()),
        'train': 'train',
        'val': 'val',
        'nc': len(LABELS),
        'names': [LABELS[i]['name'] for i in sorted(LABELS.keys())]
    }
    
    yaml_path = os.path.join(output_path, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    
    logging.info(f"Created data.yaml at: {yaml_path}")
    logging.info(f"YAML content: {yaml_data}")
    
    return yaml_path

# Create mock field dataset
field_data = prepare_mock_field_dataset_from_collected(
    collected_df, 
    OUTPUT_DIRS["field_images"],
    test_size=0.2
)

yaml_path = create_yolo_yaml(OUTPUT_DIRS["field_images"], OUTPUT_DIRS["field_images"])

# %% [markdown]
# ## PHASE 4: YOLO Training

# %%
def train_yolo_detector(yaml_path, output_dir, epochs=5, imgsz=640, batch=16):
    """Train YOLO model for leaf detection"""
    if not YOLO_AVAILABLE:
        logging.error("YOLO not available")
        return None
    
    logging.info("\n" + "=" * 60)
    logging.info("STARTING YOLO TRAINING")
    logging.info("=" * 60)
    
    # Determine proper device for YOLO
    if CUDA_AVAILABLE and CUDA_COUNT > 0:
        yolo_device = '0'  # Use first CUDA device
        logging.info(f"YOLO training on GPU: {yolo_device}")
    else:
        yolo_device = 'cpu'
        logging.info("YOLO training on CPU")
    
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=yolo_device,  # Fixed device selection
        patience=10,
        save_period=5,
        workers=4,
        project=output_dir,
        name='rice_leaf_detector',
        exist_ok=True,
        verbose=True,
        plots=True
    )
    
    best_model = Path(output_dir) / 'rice_leaf_detector' / 'weights' / 'best.pt'
    logging.info(f"\n✓ YOLO training completed")
    logging.info(f"✓ Best model: {best_model}")
    
    return str(best_model)

# Train YOLO
if field_data and len(field_data['train']) > 0:
    best_yolo_model = train_yolo_detector(
        yaml_path=yaml_path,
        output_dir=OUTPUT_DIRS["yolo_weights"],
        epochs=5,
        imgsz=640,
        batch=16
    )
else:
    logging.error("No training data available")
    best_yolo_model = None

# %% [markdown]
# ## PHASE 5: Extract Crops with YOLO

# %%
def extract_crops_with_yolo(yolo_model_path, field_images, output_dir, confidence=0.25):
    """Extract leaf crops using YOLO"""
    logging.info("\n" + "=" * 60)
    logging.info("EXTRACTING CROPS WITH YOLO")
    logging.info("=" * 60)
    
    model = YOLO(yolo_model_path)
    crops_data = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'val']:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        logging.info(f"\nProcessing {split} split...")
        
        for item in tqdm(field_images[split], desc=f"Extracting {split}"):
            img_path = item['image_path']
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            results = model.predict(img_path, conf=confidence, verbose=False)
            
            base_name = Path(img_path).stem
            true_label = item.get('true_label', 'unknown')
            
            for idx, result in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, result.xyxy[0].cpu().numpy())
                conf = float(result.conf.cpu())
                
                # Padding
                padding = 10
                h, w = img_rgb.shape[:2]
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                
                crop = img_rgb[y1:y2, x1:x2]
                
                if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
                    continue
                
                crop_filename = f"{base_name}_crop{idx:03d}.jpg"
                crop_path = os.path.join(split_dir, crop_filename)
                
                Image.fromarray(crop).save(crop_path)
                
                crops_data.append({
                    'crop_path': crop_path,
                    'field_source': img_path,
                    'split': split,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'crop_id': f"{base_name}_crop{idx:03d}",
                    'true_label': true_label,
                    'label_id': item.get('label_id')
                })
    
    crops_df = pd.DataFrame(crops_data)
    crops_csv = os.path.join(output_dir, "crops_metadata.csv")
    crops_df.to_csv(crops_csv, index=False)
    
    logging.info(f"\n✓ Extracted {len(crops_df)} crops")
    logging.info(f"  Train: {len(crops_df[crops_df['split']=='train'])}")
    logging.info(f"  Val: {len(crops_df[crops_df['split']=='val'])}")
    
    return crops_df

if best_yolo_model and os.path.exists(best_yolo_model):
    crops_df = extract_crops_with_yolo(
        yolo_model_path=best_yolo_model,
        field_images=field_data,
        output_dir=OUTPUT_DIRS["yolo_crops"],
        confidence=0.25
    )
else:
    logging.error("No YOLO model available")
    crops_df = None

# %% [markdown]
# ## PHASE 6: Auto-Label Crops

# %%
def auto_label_crops(crops_df):
    """Auto-label crops using ground truth"""
    logging.info("\n" + "=" * 60)
    logging.info("AUTO-LABELING CROPS")
    logging.info("=" * 60)
    
    labeled_df = crops_df.copy()
    labeled_df['label_name'] = labeled_df['true_label']
    
    # Remove any without labels
    labeled_df = labeled_df.dropna(subset=['label_id'])
    labeled_df['label_id'] = labeled_df['label_id'].astype(int)
    
    logging.info(f"✓ Labeled {len(labeled_df)} crops")
    logging.info(f"\nLabel distribution:")
    logging.info(labeled_df.groupby(['label_id', 'label_name']).size())
    
    labels_csv = os.path.join(OUTPUT_DIRS["labeled_crops"], "labeled_crops.csv")
    labeled_df.to_csv(labels_csv, index=False)
    
    return labeled_df

if crops_df is not None:
    labeled_crops_df = auto_label_crops(crops_df)
else:
    labeled_crops_df = None

# %% [markdown]
# ## PHASE 7: Classification Training

# %%
# ===== MODELS =====
def build_efficientnet(num_classes):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    in_feat = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_feat, num_classes)
    )
    return model

def build_mobilenet(num_classes):
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    in_feat = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_feat, num_classes)
    )
    return model

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
train_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomResizedCrop(384, scale=(0.8, 1.0)),  # ← Add this
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def setup_amp():
    """Setup AMP with proper device validation"""
    use_cuda = CUDA_AVAILABLE and CUDA_COUNT > 0
    device = DEVICE
    
    if not use_cuda:
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
    
    logging.info(f"AMP setup: device={device}, cuda_enabled={use_cuda}")
    return device, amp_ctx, scaler

def train_classification_model(model, train_loader, val_loader, epochs=20, lr=1e-4, model_name="model"):
    device, amp_ctx, scaler = setup_amp()
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_model_path = None
    
    logging.info(f"\nTraining {model_name}...")
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False):
            imgs = imgs.to(device)
            labels = labels.long().to(device)
            
            optimizer.zero_grad()
            with amp_ctx:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * imgs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += imgs.size(0)
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.long().to(device)
                
                with amp_ctx:
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        logging.info(f"Epoch {epoch}/{epochs}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(OUTPUT_DIRS["classification_weights"], f"{model_name}_best.pth")
            torch.save(model.state_dict(), best_model_path)
    
    return history, best_model_path

def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

@torch.no_grad()
def measure_latency(model, device, input_size=(1, 3, 224, 224), runs=50):
    model.eval()
    x = torch.randn(*input_size, device=device)
    
    for _ in range(10):
        _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(runs):
        _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    return (time.time() - start) / runs * 1000

@torch.no_grad()
def evaluate_model(model, loader, device):
    model = model.to(device)  # Ensure model is on correct device
    model.eval()
    all_preds, all_labels = [], []
    
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.long().to(device)
        
        outputs = model(imgs)
        preds = outputs.argmax(1)
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': acc,
        'precision': p,
        'recall': r,
        'f1': f1,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred
    }

# ===== TRAIN MODELS =====
if labeled_crops_df is not None and len(labeled_crops_df) > 0:
    logging.info("\n" + "=" * 60)
    logging.info("CLASSIFICATION TRAINING")
    logging.info("=" * 60)
    
    train_crops = labeled_crops_df[labeled_crops_df['split'] == 'train']
    val_crops = labeled_crops_df[labeled_crops_df['split'] == 'val']
    
    logging.info(f"Train: {len(train_crops)}, Val: {len(val_crops)}")
    
    train_dataset = CropDataset(train_crops, transform=train_transform)
    val_dataset = CropDataset(val_crops, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    num_classes = len(LABELS)
    models_to_train = {
        'MobileNetV2': lambda: build_mobilenet(num_classes),
        'EfficientNetB0': lambda: build_efficientnet(num_classes)
    }
    
    results = []
    histories = {}
    
    for model_name, model_builder in models_to_train.items():
        logging.info(f"\n{'='*60}")
        logging.info(f"Training {model_name}")
        logging.info(f"{'='*60}")
        
        model = model_builder()
        
        history, best_path = train_classification_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=20,
            lr=1e-4,
            model_name=model_name
        )
        
        histories[model_name] = history
        
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        model = model.to(DEVICE)  # Ensure model is on correct device
        metrics = evaluate_model(model, val_loader, DEVICE)
        
        params = count_params(model)
        latency = measure_latency(model, DEVICE)
        fps = 1000 / latency
        
        results.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1'],
            'Params (M)': params,
            'Latency (ms)': latency,
            'FPS': fps,
            'Checkpoint': best_path
        })
        
        logging.info(f"\n{model_name} Results:")
        logging.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"  F1: {metrics['f1']:.4f}")
        logging.info(f"  Speed: {fps:.1f} FPS")
        
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(OUTPUT_DIRS["results"], "model_comparison.csv")
    results_df.to_csv(results_csv, index=False)
    
    logging.info(f"\n✓ Results saved: {results_csv}")

else:
    results_df = None
    histories = {}

# %% [markdown]
# ## PHASE 8: Comprehensive Visualization

# %%
def plot_training_curves(histories, save_dir):
    """Training curves comparison"""
    if not histories:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves Comparison', fontsize=16, fontweight='bold')
    
    # Loss
    ax = axes[0, 0]
    for model_name, history in histories.items():
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], label=f"{model_name} Train", linewidth=2)
        ax.plot(epochs, history['val_loss'], label=f"{model_name} Val", linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[0, 1]
    for model_name, history in histories.items():
        epochs = range(1, len(history['train_acc']) + 1)
        ax.plot(epochs, history['train_acc'], label=f"{model_name} Train", linewidth=2)
        ax.plot(epochs, history['val_acc'], label=f"{model_name} Val", linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training & Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Val Loss comparison
    ax = axes[1, 0]
    for model_name, history in histories.items():
        epochs = range(1, len(history['val_loss']) + 1)
        ax.plot(epochs, history['val_loss'], label=model_name, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Val Accuracy comparison
    ax = axes[1, 1]
    for model_name, history in histories.items():
        epochs = range(1, len(history['val_acc']) + 1)
        ax.plot(epochs, history['val_acc'], label=model_name, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    logging.info(f"Training curves saved: {save_path}")

def plot_model_comparison(results_df, save_dir):
    """Comprehensive model comparison"""
    if results_df is None or len(results_df) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    models = results_df['Model'].values
    
    # 1. Metrics comparison
    ax = axes[0, 0]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = results_df[metric].values
        ax.bar(x + i*width, values, width, label=metric)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Classification Metrics')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    # 2. Size vs Accuracy
    ax = axes[0, 1]
    ax.scatter(results_df['Params (M)'], results_df['Accuracy'], s=200, alpha=0.6)
    for idx, row in results_df.iterrows():
        ax.annotate(row['Model'], (row['Params (M)'], row['Accuracy']), 
                   xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax.set_xlabel('Model Size (M parameters)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Size vs Accuracy')
    ax.grid(True, alpha=0.3)
    
    # 3. Speed comparison
    ax = axes[1, 0]
    ax.barh(models, results_df['FPS'].values, color='steelblue')
    ax.set_xlabel('FPS')
    ax.set_title('Inference Speed')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 4. Overall ranking
    ax = axes[1, 1]
    norm_acc = (results_df['Accuracy'] - results_df['Accuracy'].min()) / (results_df['Accuracy'].max() - results_df['Accuracy'].min())
    norm_fps = (results_df['FPS'] - results_df['FPS'].min()) / (results_df['FPS'].max() - results_df['FPS'].min())
    overall_score = 0.7 * norm_acc + 0.3 * norm_fps
    
    colors = plt.cm.RdYlGn(overall_score)
    ax.barh(models, overall_score, color=colors)
    ax.set_xlabel('Overall Score (0.7×Acc + 0.3×Speed)')
    ax.set_title('Overall Ranking')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_confusion_matrices(results_df, val_loader, save_dir):
    """Confusion matrices for all models"""
    if results_df is None:
        return
    
    n_models = len(results_df)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
    
    class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
    
    for idx, row in results_df.iterrows():
        model_name = row['Model']
        checkpoint = row['Checkpoint']
        
        if model_name == 'MobileNetV2':
            model = build_mobilenet(len(LABELS))
        else:
            model = build_efficientnet(len(LABELS))
        
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        model = model.to(DEVICE)  # Ensure model is on correct device
        metrics = evaluate_model(model, val_loader, DEVICE)
        
        cm = metrics['confusion_matrix']
        
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=class_names,
                   yticklabels=class_names)
        ax.set_title(f'{model_name}\nAcc: {metrics["accuracy"]:.3f}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        del model
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'confusion_matrices.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_yolo_crops_samples(crops_df, n_samples=12, save_dir=None):
    """YOLO extracted crops visualization"""
    if crops_df is None or len(crops_df) == 0:
        return
    
    samples = crops_df.sample(min(n_samples, len(crops_df)))
    
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten()
    
    fig.suptitle('YOLO Extracted Crops', fontsize=16, fontweight='bold')
    
    for idx, (_, row) in enumerate(samples.iterrows()):
        if idx >= n_samples:
            break
        
        img = Image.open(row['crop_path'])
        axes[idx].imshow(img)
        axes[idx].set_title(f"{row.get('label_name', 'Unknown')}\nConf: {row['confidence']:.2f}", 
                           fontsize=10)
        axes[idx].axis('off')
    
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, 'yolo_crops_samples.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

# Generate visualizations
if histories and results_df is not None:
    logging.info("\n" + "=" * 60)
    logging.info("GENERATING VISUALIZATIONS")
    logging.info("=" * 60)
    
    plot_training_curves(histories, OUTPUT_DIRS["training_curves"])
    plot_model_comparison(results_df, OUTPUT_DIRS["comparison"])
    
    if labeled_crops_df is not None:
        val_crops = labeled_crops_df[labeled_crops_df['split'] == 'val']
        val_dataset = CropDataset(val_crops, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
        
        plot_confusion_matrices(results_df, val_loader, OUTPUT_DIRS["evaluation"])
    
    plot_yolo_crops_samples(crops_df, n_samples=12, save_dir=OUTPUT_DIRS["plots"])

# %% [markdown]
# ## PHASE 9: Best Model Evaluation

# %%
def generate_detailed_report(results_df, val_loader, save_dir):
    """Detailed evaluation of best model"""
    if results_df is None:
        return
    
    best_row = results_df.loc[results_df['F1'].idxmax()]
    best_model_name = best_row['Model']
    best_checkpoint = best_row['Checkpoint']
    
    logging.info("\n" + "=" * 60)
    logging.info("BEST MODEL EVALUATION")
    logging.info("=" * 60)
    logging.info(f"Model: {best_model_name}")
    logging.info(f"F1: {best_row['F1']:.4f}")
    logging.info(f"Accuracy: {best_row['Accuracy']:.4f}")
    
    if best_model_name == 'MobileNetV2':
        model = build_mobilenet(len(LABELS))
    else:
        model = build_efficientnet(len(LABELS))
    
    model.load_state_dict(torch.load(best_checkpoint, map_location=DEVICE))
    model = model.to(DEVICE)  # Ensure model is on correct device
    metrics = evaluate_model(model, val_loader, DEVICE)
    
    class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
    report = classification_report(
        metrics['y_true'], 
        metrics['y_pred'], 
        target_names=class_names,
        digits=4
    )
    
    logging.info("\nClassification Report:\n" + report)
    
    report_path = os.path.join(save_dir, 'best_model_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Checkpoint: {best_checkpoint}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    return best_checkpoint, best_model_name

if results_df is not None and labeled_crops_df is not None:
    val_crops = labeled_crops_df[labeled_crops_df['split'] == 'val']
    val_dataset = CropDataset(val_crops, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    best_model_path, best_model_name = generate_detailed_report(
        results_df, val_loader, OUTPUT_DIRS["evaluation"]
    )

# %% [markdown]
# ## PHASE 10: Final Summary

# %%
def export_pipeline_summary(output_dir):
    """Export comprehensive summary"""
    summary = {
        'pipeline': 'YOLO-First Classification',
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'total_sources': len(DATASET_SOURCES),
        'phases': []
    }
    
    if 'collected_df' in globals():
        summary['phases'].append('Data Collection')
        summary['total_images'] = len(collected_df)
    
    if best_yolo_model:
        summary['phases'].append('YOLO Training')
        summary['yolo_model'] = best_yolo_model
    
    if crops_df is not None:
        summary['phases'].append('Crop Extraction')
        summary['total_crops'] = len(crops_df)
    
    if labeled_crops_df is not None:
        summary['phases'].append('Labeling')
        summary['labeled_crops'] = len(labeled_crops_df)
    
    if results_df is not None:
        summary['phases'].append('Classification')
        summary['models_trained'] = len(results_df)
        summary['best_model'] = {
            'name': best_model_name,
            'path': best_model_path,
            'f1': float(results_df['F1'].max())
        }
    
    import json
    summary_path = os.path.join(output_dir, 'pipeline_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

summary = export_pipeline_summary(OUTPUT_DIRS["results"])

logging.info("\n" + "=" * 80)
logging.info("PIPELINE COMPLETE")
logging.info("=" * 80)
logging.info(f"Output: {PATH_OUTPUT}")
logging.info(f"Phases: {len(summary['phases'])}")

for phase in summary['phases']:
    logging.info(f"  ✓ {phase}")

if 'best_model' in summary:
    logging.info(f"\nBest: {summary['best_model']['name']}")
    logging.info(f"F1: {summary['best_model']['f1']:.4f}")

logging.info("\n" + "=" * 80)