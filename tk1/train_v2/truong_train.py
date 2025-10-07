# %% [markdown]
# # Multi-Model Rice Disease Classification - 10 Processes on Single GPU
# Train 10 models simultaneously on GPU 0 using multi-processing

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

# Multi-processing imports
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
from concurrent.futures import ProcessPoolExecutor

try:
    from torch.amp import GradScaler, autocast
    _NEW_AMP = True
except:
    from torch.cuda.amp import GradScaler, autocast
    _NEW_AMP = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Set multiprocessing start method
try:
    mp.set_start_method('spawn', force=True)
except:
    pass

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
    # GPU Configuration - SINGLE GPU ONLY
    'target_gpu': 0,  # Use only GPU 0
    'num_parallel_processes': 10,  # Train 10 models simultaneously
    
    # Models to train
    'models': [
        'mobilenet_v3_small',
        'mobilenet_v3_large',
        'efficientnet_b0',
        'efficientnet_v2_s',
        'resnet18',
        'shufflenet_v2_x1_0',
        'mobilenet_v2',
        'squeezenet1_1',
        'densenet121',
        'resnext50_32x4d',
    ],
    
    # Training - Optimized for parallel execution on single GPU
    'img_size': 224,
    'batch_size': 64,  # Smaller batch per process
    'epochs': 50,
    'lr': 3e-4,
    'num_workers': 2,  # Fewer workers per process
    'pin_memory': True,
    'prefetch_factor': 2,
    'persistent_workers': False,
    
    # Model enhancements
    'use_cbam': True,
    'use_better_head': True,
    
    # Optimizations
    'use_weighted_sampler': True,
    'use_sce_loss': True,
    'sce_alpha': 0.1,
    'sce_beta': 1.0,
    'use_ema': True,
    'ema_decay': 0.9995,
    'use_mixup': True,
    'mixup_alpha': 0.3,
    'use_cutmix': True,
    'cutmix_alpha': 1.0,
    
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
    
    # Memory optimization
    'gradient_accumulation_steps': 1,
    'empty_cache_every_n_batches': 50,
}

# %%
# ===== SETUP =====
def get_output_folder(parent_dir: str, env_name: str) -> str:
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = f"{env_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir = os.path.join(parent_dir, experiment_id)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

PATH_OUTPUT = get_output_folder("../output", "10-processes-single-gpu")

def create_output_structure(base_path):
    folders = ["weights", "results", "plots", "logs", "demo", "comparison"]
    for folder in folders:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)
    return {folder: os.path.join(base_path, folder) for folder in folders}

OUTPUT_DIRS = create_output_structure(PATH_OUTPUT)

# %%
# ===== LABELS =====
LABELS = {
    0: {"name": "brown_spot", "match_substrings": ["../data_total/brown_spot"]},
    1: {"name": "leaf_blast", "match_substrings": ["../data_total/blast"]},
    2: {"name": "leaf_blight", "match_substrings": ["../data_total/bacterial_leaf_blight"]},
    3: {"name": "healthy", "match_substrings": ["../data_total/normal"]},
}

# %%
# ===== GPU SETUP =====
def setup_gpu(gpu_id=0):
    """Setup specific GPU"""
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        
        # Enable optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'
        
        device = torch.device(f'cuda:{gpu_id}')
        return device
    else:
        return torch.device('cpu')

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
        features = nn.Sequential(*list(base_model.children())[:-2])
        feat_channels = 512
        
    elif backbone_name == 'shufflenet_v2_x1_0':
        from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
        weights = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
        base_model = shufflenet_v2_x1_0(weights=weights)
        features = nn.Sequential(*list(base_model.children())[:-2])
        feat_channels = 1024
        
    elif backbone_name == 'mobilenet_v2':
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        base_model = mobilenet_v2(weights=weights)
        features = base_model.features
        feat_channels = 1280
        
    elif backbone_name == 'squeezenet1_1':
        from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
        weights = SqueezeNet1_1_Weights.IMAGENET1K_V1
        base_model = squeezenet1_1(weights=weights)
        features = base_model.features
        feat_channels = 512
        
    elif backbone_name == 'densenet121':
        from torchvision.models import densenet121, DenseNet121_Weights
        weights = DenseNet121_Weights.IMAGENET1K_V1
        base_model = densenet121(weights=weights)
        features = base_model.features
        feat_channels = 1024
        
    elif backbone_name == 'resnext50_32x4d':
        from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
        weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        base_model = resnext50_32x4d(weights=weights)
        features = nn.Sequential(*list(base_model.children())[:-2])
        feat_channels = 2048
        
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    if CONFIG['use_cbam']:
        cbam = CBAM(feat_channels, reduction=16)
        features = nn.Sequential(*list(features.children()), cbam)
    
    model = nn.Module()
    model.features = features
    
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
        print(f"Error reading {path}: {e}")
    return images

def auto_collect_dataset():
    print("="*60)
    print("DATA COLLECTION")
    print("="*60)
    
    all_data = []
    for label_id, label_info in LABELS.items():
        label_name = label_info['name']
        match_paths = label_info['match_substrings']
        
        print(f"\nCollecting {label_name} (ID: {label_id})...")
        
        for path in match_paths:
            images = collect_images_from_path(path)
            if len(images) > 0:
                print(f"  ✓ {len(images)} images from {path}")
                for img_path in images:
                    all_data.append({
                        'image_path': img_path,
                        'label_id': label_id,
                        'label_name': label_name,
                    })
    
    df = pd.DataFrame(all_data)
    print(f"\nTotal: {len(df)} images")
    print(f"\nBy label:\n{df.groupby('label_name').size()}")
    
    return df

collected_df = auto_collect_dataset()
collected_df.to_csv(os.path.join(OUTPUT_DIRS["results"], "collected_images.csv"), index=False)

# Split data
train_val_df, test_df = train_test_split(
    collected_df, test_size=CONFIG['test_size'], random_state=42, stratify=collected_df['label_id']
)
train_df, val_df = train_test_split(
    train_val_df, test_size=CONFIG['val_size']/(1-CONFIG['test_size']), 
    random_state=42, stratify=train_val_df['label_id']
)

print(f"\nData splits:")
print(f"Train: {len(train_df)} ({len(train_df)/len(collected_df)*100:.1f}%)")
print(f"Val:   {len(val_df)} ({len(val_df)/len(collected_df)*100:.1f}%)")
print(f"Test:  {len(test_df)} ({len(test_df)/len(collected_df)*100:.1f}%)")

# Save splits
train_df.to_csv(os.path.join(OUTPUT_DIRS["results"], "train_split.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIRS["results"], "val_split.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIRS["results"], "test_split.csv"), index=False)

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

# ===== MIXUP & CUTMIX =====
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

def mixup_criterion(criterion, pred, y_a, y_b, lam):
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

# ===== CLAHE TRANSFORM (FIXED FOR MULTIPROCESSING) =====
class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, pil_image):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        
        img_array = np.array(pil_image)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced.astype(np.uint8))

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
    base_train_transforms = []
    base_val_transforms = []
    
    if CONFIG['use_clahe']:
        clahe_transform = CLAHETransform(
            clip_limit=CONFIG['clahe_clip_limit'],
            tile_grid_size=CONFIG['clahe_tile_size']
        )
        base_train_transforms.append(clahe_transform)
        base_val_transforms.append(clahe_transform)
    
    train_transforms = base_train_transforms + [
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ]
    
    val_transforms = base_val_transforms + [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    
    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)

# ===== DATA LOADERS =====
def make_loader(df, transform, batch_size, train=True):
    dataset = ImageDataset(df, transform=transform)
    
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': CONFIG['num_workers'],
        'pin_memory': CONFIG['pin_memory'],
        'drop_last': train,
    }
    
    if CONFIG['num_workers'] > 0 and CONFIG.get('prefetch_factor'):
        loader_kwargs['prefetch_factor'] = CONFIG['prefetch_factor']
    
    if train and CONFIG['use_weighted_sampler']:
        counts = df['label_id'].value_counts().sort_index().values.astype(float)
        class_weights = 1.0 / (counts + 1e-6)
        sample_weights = df['label_id'].map({i:w for i,w in enumerate(class_weights)}).values
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        loader_kwargs['sampler'] = sampler
    else:
        loader_kwargs['shuffle'] = train
    
    return DataLoader(dataset, **loader_kwargs)

# %% [markdown]
# ## MULTI-PROCESS TRAINING ON SINGLE GPU

# %%
def train_model_process(model_name, gpu_id, process_id, result_queue, config, output_dirs):
    """Train a single model in isolated process on specified GPU"""
    
    try:
        # Set random seed
        seed = 42 + process_id
        seed_everything(seed)
        
        # Setup GPU
        device = setup_gpu(gpu_id)
        
        print(f"\n[Process {process_id}] [GPU {gpu_id}] Starting {model_name}")
        
        # Load data
        train_df = pd.read_csv(os.path.join(output_dirs["results"], "train_split.csv"))
        val_df = pd.read_csv(os.path.join(output_dirs["results"], "val_split.csv"))
        
        num_classes = len(LABELS)
        
        # Build model
        model, total_params, trainable_params = build_classifier(model_name, num_classes)
        model = model.to(device)
        
        print(f"[Process {process_id}] {model_name} - Params: {total_params/1e6:.2f}M")
        
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
        if config['use_sce_loss']:
            criterion = SymmetricCrossEntropy(config['sce_alpha'], config['sce_beta'], num_classes)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Prepare loaders
        train_transform, val_transform = get_transforms(config['img_size'])
        train_loader = make_loader(train_df, train_transform, config['batch_size'], train=True)
        val_loader = make_loader(val_df, val_transform, config['batch_size'], train=False)
        
        # Optimizer & Scheduler
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['lr'],
            epochs=config['epochs'],
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # EMA
        ema = ModelEMA(model, decay=config['ema_decay']) if config['use_ema'] else None
        
        best_val_acc = 0.0
        bad_epochs = 0
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_path = os.path.join(output_dirs["weights"], f"{model_name}_best.pth")
        
        start_time = time.time()
        
        for epoch in range(1, config['epochs'] + 1):
            # TRAIN
            model.train()
            epoch_loss, correct, total = 0.0, 0, 0
            
            for batch_idx, (imgs, labels) in enumerate(train_loader):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.long().to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                # Apply augmentation
                use_aug = random.random() < 0.5
                if use_aug:
                    if config['use_mixup'] and random.random() < 0.5:
                        imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, config['mixup_alpha'])
                    elif config['use_cutmix']:
                        imgs, labels_a, labels_b, lam = cutmix_data(imgs, labels, config['cutmix_alpha'])
                    else:
                        use_aug = False
                
                with amp_ctx:
                    logits = model(imgs)
                    if use_aug:
                        loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                    else:
                        loss = criterion(logits, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                if ema is not None:
                    ema.update(model)
                
                epoch_loss += loss.item() * imgs.size(0)
                pred = logits.argmax(1)
                correct += (pred == labels).sum().item()
                total += imgs.size(0)
                
                if batch_idx % config['empty_cache_every_n_batches'] == 0:
                    torch.cuda.empty_cache()
            
            train_loss = epoch_loss / total
            train_acc = correct / total
            
            # VALIDATE
            eval_model = ema.ema if (ema is not None) else model
            eval_model.eval()
            vloss, vcorrect, vtotal = 0.0, 0, 0
            
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    labels = labels.long().to(device, non_blocking=True)
                    
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
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"[P{process_id}] [{model_name}] Epoch {epoch}/{config['epochs']}: "
                      f"TL={train_loss:.4f} TA={train_acc:.4f} | VL={val_loss:.4f} VA={val_acc:.4f}")
            
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
            else:
                bad_epochs += 1
                if bad_epochs >= config['patience']:
                    print(f"[P{process_id}] [{model_name}] Early stopping at epoch {epoch}")
                    break
        
        training_time = time.time() - start_time
        
        # Clean up
        del model, optimizer, scheduler
        if ema is not None:
            del ema
        torch.cuda.empty_cache()
        gc.collect()
        
        result = {
            'model_name': model_name,
            'gpu_id': gpu_id,
            'process_id': process_id,
            'history': history,
            'checkpoint': best_path,
            'best_val_acc': best_val_acc,
            'training_time': training_time,
            'total_params': total_params,
            'status': 'success'
        }
        
        result_queue.put(result)
        print(f"[P{process_id}] ✓ {model_name} completed - Val Acc: {best_val_acc:.4f} - Time: {training_time/60:.1f}min")
        
    except Exception as e:
        print(f"[P{process_id}] ✗ {model_name} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        result = {
            'model_name': model_name,
            'gpu_id': gpu_id,
            'process_id': process_id,
            'status': 'failed',
            'error': str(e)
        }
        result_queue.put(result)

def train_models_multiprocess_single_gpu(models_list, config, output_dirs):
    """Train multiple models on single GPU using processes"""
    
    print("\n" + "="*80)
    print("MULTI-PROCESS TRAINING ON SINGLE GPU")
    print(f"Training {len(models_list)} models on GPU {config['target_gpu']}")
    print(f"Number of parallel processes: {config['num_parallel_processes']}")
    print("="*80)
    
    # Create result queue
    manager = Manager()
    result_queue = manager.Queue()
    
    # All models go to the same GPU
    gpu_id = config['target_gpu']
    
    print("\nProcess Assignments:")
    for idx, model_name in enumerate(models_list):
        print(f"  Process {idx}: {model_name} → GPU {gpu_id}")
    
    # Start all processes
    processes = []
    for idx, model_name in enumerate(models_list):
        p = Process(
            target=train_model_process,
            args=(model_name, gpu_id, idx, result_queue, config, output_dirs)
        )
        p.start()
        processes.append(p)
        time.sleep(1)  # Stagger starts slightly
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    # Collect results
    results = {}
    while not result_queue.empty():
        result = result_queue.get()
        results[result['model_name']] = result
    
    print("\n" + "="*80)
    print("MULTI-PROCESS TRAINING COMPLETED")
    print("="*80)
    
    # Summary
    successful = [k for k, v in results.items() if v.get('status') == 'success']
    failed = [k for k, v in results.items() if v.get('status') == 'failed']
    
    print(f"\nSuccessful: {len(successful)}/{len(models_list)}")
    if successful:
        print("Models completed:")
        for model_name in successful:
            result = results[model_name]
            print(f"  ✓ {model_name}: Val Acc={result['best_val_acc']:.4f}, "
                  f"Time={result['training_time']/60:.1f}min")
    
    if failed:
        print(f"\nFailed: {len(failed)}/{len(models_list)}")
        for model_name in failed:
            print(f"  ✗ {model_name}: {results[model_name].get('error', 'Unknown')}")
    
    return results

# %% [markdown]
# ## MAIN EXECUTION

# %%
if __name__ == '__main__':
    print("\n" + "="*80)
    print("10 PROCESSES ON SINGLE GPU TRAINING")
    print(f"Target GPU: {CONFIG['target_gpu']}")
    print(f"Models to train: {len(CONFIG['models'])}")
    print(f"Parallel processes: {CONFIG['num_parallel_processes']}")
    print("="*80)
    
    start_time = time.time()
    
    training_results = train_models_multiprocess_single_gpu(
        CONFIG['models'],
        CONFIG,
        OUTPUT_DIRS
    )
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"TOTAL TRAINING TIME: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Average time per model: {total_time/len(CONFIG['models'])/60:.1f} minutes")
    print(f"{'='*80}")
    
    # Extract successful results
    all_histories = {}
    all_checkpoints = {}
    all_best_val_accs = {}
    
    for model_name, result in training_results.items():
        if result.get('status') == 'success':
            all_histories[model_name] = result['history']
            all_checkpoints[model_name] = result['checkpoint']
            all_best_val_accs[model_name] = result['best_val_acc']
    
    print(f"\nSuccessfully trained {len(all_checkpoints)} models")
    
    # Save training summary
    summary_data = []
    for model_name, result in training_results.items():
        if result.get('status') == 'success':
            summary_data.append({
                'Model': model_name,
                'GPU': result['gpu_id'],
                'Process_ID': result['process_id'],
                'Best_Val_Acc': f"{result['best_val_acc']:.4f}",
                'Training_Time_min': f"{result['training_time']/60:.1f}",
                'Parameters_M': f"{result['total_params']/1e6:.2f}",
                'Status': 'Success'
            })
        else:
            summary_data.append({
                'Model': model_name,
                'GPU': result.get('gpu_id', '-'),
                'Process_ID': result.get('process_id', '-'),
                'Best_Val_Acc': '-',
                'Training_Time_min': '-',
                'Parameters_M': '-',
                'Status': f"Failed: {result.get('error', 'Unknown')[:50]}"
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(OUTPUT_DIRS["results"], "training_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    print(f"\n✓ Training summary saved: {summary_path}")
    print(f"✓ Output directory: {PATH_OUTPUT}")
