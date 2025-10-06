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
# ===== CONFIGURATION =====
CONFIG = {
    # Models to train (all <50M params)
    'models': [
        # 'mobilenet_v3_small',    # ~2.5M params
        # 'mobilenet_v3_large',    # ~5.4M params
        # 'efficientnet_b0',       # ~5.3M params
        'efficientnet_v2_s',     # ~21M params
        # 'resnet18',              # ~11M params
        # 'shufflenet_v2_x1_0',    # ~2.3M params
    ],
    
    # Training
    'img_size': 224,
    'batch_size': 64,
    'epochs': 25,
    'lr': 3e-4,
    'num_workers': 2,
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
    'mixup_alpha': 0.2,
    
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

PATH_OUTPUT = get_output_folder("../output", "multi-model-classifier")

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
        "./data/brown_spot",
        "../data/yolo_detected/paddy_disease_train/brown_spot/crops",
        "../data/yolo_detected/sikhaok_train/BrownSpot/crops",
        ]},
    1: {"name": "leaf_blast", "match_substrings": [
        "./data/blast",
        "../data/yolo_detected/paddy_disease_train/blast/crops",
        "../data/yolo_detected/sikhaok_train/LeafBlast/crops",
        ]},
    2: {"name": "leaf_blight", "match_substrings": [
        "./data/bacterial_leaf_blight",
        "../data/yolo_detected/paddy_disease_train/bacterial_leaf_blight/crops",
        "../data/yolo_detected/sikhaok_train/Bacterialblight1/crops",
        ]},
    3: {"name": "healthy", "match_substrings": [
        "./data/normal",
        "../data/yolo_detected/paddy_disease_train/normal/crops",
        "../data/yolo_detected/sikhaok_train/Healthy/crops",
        ]}
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
        features = nn.Sequential(*list(base_model.children())[:-1])
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
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # EMA
    ema = ModelEMA(model, decay=CONFIG['ema_decay']) if CONFIG['use_ema'] else None
    
    # Prepare loaders
    train_transform, val_transform = get_transforms(CONFIG['img_size'])
    train_loader = make_loader(train_df, train_transform, CONFIG['batch_size'], train=True)
    val_loader = make_loader(val_df, val_transform, CONFIG['batch_size'], train=False)
    
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
            if CONFIG['use_mixup'] and random.random() < 0.5:
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
# Save results summary
results_summary = []
for model_name in CONFIG['models']:
    metrics = all_test_metrics[model_name]
    results_summary.append({
        'Model': model_name,
        'Test Accuracy': f"{metrics['accuracy']:.4f}",
        'Test Precision': f"{metrics['precision']:.4f}",
        'Test Recall': f"{metrics['recall']:.4f}",
        'Test F1': f"{metrics['f1']:.4f}",
        'Best Val Acc': f"{all_best_val_accs[model_name]:.4f}",
    })

if 'ensemble' in all_test_metrics:
    metrics = all_test_metrics['ensemble']
    results_summary.append({
        'Model': 'ensemble',
        'Test Accuracy': f"{metrics['accuracy']:.4f}",
        'Test Precision': f"{metrics['precision']:.4f}",
        'Test Recall': f"{metrics['recall']:.4f}",
        'Test F1': f"{metrics['f1']:.4f}",
        'Best Val Acc': '-',
    })

summary_df = pd.DataFrame(results_summary)
summary_path = os.path.join(OUTPUT_DIRS["results"], "models_summary.csv")
summary_df.to_csv(summary_path, index=False)

logging.info("\n" + "="*80)
logging.info("RESULTS SUMMARY")
logging.info("="*80)
print(summary_df.to_string(index=False))

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

logging.info("\n" + "="*80)
logging.info("TRAINING COMPLETE")
logging.info("="*80)
logging.info(f"Output directory: {PATH_OUTPUT}")
logging.info(f"Results summary: {summary_path}")
logging.info("="*80)