# %% [markdown]
# # Simplified Rice Disease Classification
# Direct training on full images without cropping

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
from typing import Dict, List
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
# ===== CONFIGURATION =====
CONFIG = {
    # Model
    'backbone': 'efficientnet_v2_s',  # 'mobilenetv3_small' | 'efficientnet_v2_s'
    'use_cbam': True,
    'use_better_head': True,
    
    # Training
    'img_size': 224,
    'batch_size': 16,  # Giảm từ 32 xuống 16 để tránh memory issues
    'epochs': 50,
    'lr': 3e-4,
    'num_workers': 2,   # Giảm từ 4 xuống 2 để tiết kiệm RAM
    'pin_memory': True,
    
    # Optimizations
    'use_weighted_sampler': True,
    'use_sce_loss': True,
    'sce_alpha': 0.1,
    'sce_beta': 1.0,
    'use_ema': True,
    'ema_decay': 0.999,
    
    # Data splits
    'val_size': 0.15,
    'test_size': 0.15,
}

# %%
# ===== SETUP =====
def get_output_folder(parent_dir: str, env_name: str) -> str:
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = f"{env_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir = os.path.join(parent_dir, experiment_id)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

PATH_OUTPUT = get_output_folder("../output", "simple-classifier")

def create_output_structure(base_path):
    folders = ["weights", "results", "plots", "logs", "demo"]
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
    0: {"name": "brown_spot", "match_substrings": ["../data/new_data_field_rice_detected/yolo_no_overlay/brown_spot"]},
    1: {"name": "leaf_blast", "match_substrings": ["../data/new_data_field_rice_detected/yolo_no_overlay/leaf_blast"]},
    2: {"name": "leaf_blight", "match_substrings": ["../data/new_data_field_rice_detected/yolo_no_overlay/leaf_blight"]},
    3: {"name": "healthy", "match_substrings": ["../data/new_data_field_rice_detected/yolo_no_overlay/healthy"]}
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
            nn.LayerNorm(hidden),  # Sử dụng LayerNorm thay vì BatchNorm để tránh lỗi batch size = 1
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
def build_classifier(num_classes):
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
    
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    net = efficientnet_v2_s(weights=weights)
    features = net.features
    feat_channels = 1280
    
    logging.info(f"Backbone: {CONFIG['backbone']} with {feat_channels} channels")
    
    if CONFIG['use_cbam']:
        cbam = CBAM(feat_channels, reduction=16)
        features = nn.Sequential(*list(features.children()), cbam)
        logging.info("✓ CBAM attached")
    
    model = nn.Module()
    model.features = features
    
    if CONFIG['use_better_head']:
        model.head = EnhancedHead(feat_channels, num_classes, dropout=0.3)
        logging.info("✓ Enhanced head")
    else:
        model.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_channels, num_classes)
        )
    
    def _forward(self, x):
        x = self.features(x)
        return self.head(x)
    
    model.forward = _forward.__get__(model, type(model))
    return model

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
            drop_last=train  # Drop last incomplete batch khi training
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=train,
            num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'],
            drop_last=train  # Drop last incomplete batch khi training
        )
    
    return loader

# %% [markdown]
# ## TRAINING

# %%
def train_model(model, train_df, val_df, epochs):
    device = DEVICE
    model = model.to(device)
    
    # Setup AMP
    use_cuda = torch.cuda.is_available()
    if _NEW_AMP:
        amp_ctx = autocast(device_type="cuda", enabled=use_cuda)
        scaler = GradScaler(device="cuda" if use_cuda else "cpu", enabled=use_cuda)
    else:
        from contextlib import nullcontext
        amp_ctx = autocast(enabled=use_cuda) if use_cuda else nullcontext()
        scaler = GradScaler(enabled=use_cuda)
    
    num_classes = len(LABELS)
    
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
    
    # Prepare loaders
    train_transform, val_transform = get_transforms(CONFIG['img_size'])
    train_loader = make_loader(train_df, train_transform, CONFIG['batch_size'], train=True)
    val_loader = make_loader(val_df, val_transform, CONFIG['batch_size'], train=False)
    
    best_val_acc = 0.0
    patience = 10
    bad_epochs = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_path = os.path.join(OUTPUT_DIRS["weights"], "best_model.pth")
    
    logging.info(f"\nTraining {CONFIG['backbone']}...")
    
    for epoch in range(1, epochs + 1):
        # ===== TRAIN =====
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.long().to(device)
            
            optimizer.zero_grad()
            
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
        
        logging.info(f"Epoch {epoch}: TL={train_loss:.4f} TA={train_acc:.4f} | VL={val_loss:.4f} VA={val_acc:.4f}")
        
        # Save best
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
model = build_classifier(len(LABELS))
history, best_checkpoint = train_model(model, train_df, val_df, CONFIG['epochs'])

# %% [markdown]
# ## TESTING

# %%
def test_model(model, test_df, device):
    model = model.to(device)
    model.eval()
    
    _, val_transform = get_transforms(CONFIG['img_size'])
    test_loader = make_loader(test_df, val_transform, CONFIG['batch_size'], train=False)
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            all_preds.append(outputs.argmax(1).cpu().numpy())
            all_labels.append(labels.numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1,
            'confusion_matrix': cm, 'y_true': y_true, 'y_pred': y_pred}

# Test
model.load_state_dict(torch.load(best_checkpoint, map_location=DEVICE))
test_metrics = test_model(model, test_df, DEVICE)

logging.info("\n" + "="*60)
logging.info("TEST RESULTS")
logging.info("="*60)
logging.info(f"Accuracy:  {test_metrics['accuracy']:.4f}")
logging.info(f"Precision: {test_metrics['precision']:.4f}")
logging.info(f"Recall:    {test_metrics['recall']:.4f}")
logging.info(f"F1 Score:  {test_metrics['f1']:.4f}")

# %% [markdown]
# ## VISUALIZATION

# %%
def plot_results(history, test_metrics):
    class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
    
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Loss curves
    ax1 = plt.subplot(2, 3, 1)
    epochs_range = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs_range, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs_range, history['val_loss'], 'r-', label='Val', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy curves
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs_range, history['train_acc'], 'b-', label='Train', linewidth=2)
    ax2.plot(epochs_range, history['val_acc'], 'r-', label='Val', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    ax3 = plt.subplot(2, 3, 3)
    sns.heatmap(test_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax3)
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    ax3.set_title('Confusion Matrix')
    
    # 4. Per-class metrics
    ax4 = plt.subplot(2, 3, 4)
    p, r, f1, _ = precision_recall_fscore_support(
        test_metrics['y_true'], test_metrics['y_pred'], labels=range(len(class_names))
    )
    
    x = np.arange(len(class_names))
    width = 0.25
    ax4.bar(x - width, p, width, label='Precision', alpha=0.8)
    ax4.bar(x, r, width, label='Recall', alpha=0.8)
    ax4.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax4.set_xlabel('Class')
    ax4.set_ylabel('Score')
    ax4.set_title('Per-Class Performance')
    ax4.set_xticks(x)
    ax4.set_xticklabels(class_names, rotation=45, ha='right')
    ax4.set_ylim(0, 1.1)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Data distribution
    ax5 = plt.subplot(2, 3, 5)
    train_dist = train_df.groupby('label_name').size()
    val_dist = val_df.groupby('label_name').size()
    test_dist = test_df.groupby('label_name').size()
    
    x = np.arange(len(class_names))
    width = 0.25
    ax5.bar(x - width, [train_dist.get(c, 0) for c in class_names], width, label='Train')
    ax5.bar(x, [val_dist.get(c, 0) for c in class_names], width, label='Val')
    ax5.bar(x + width, [test_dist.get(c, 0) for c in class_names], width, label='Test')
    
    ax5.set_xlabel('Class')
    ax5.set_ylabel('Samples')
    ax5.set_title('Data Distribution')
    ax5.set_xticks(x)
    ax5.set_xticklabels(class_names, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    summary_text = f"""
    RESULTS SUMMARY
    ═══════════════════════════
    
    Test Accuracy:  {test_metrics['accuracy']:.4f}
    Test Precision: {test_metrics['precision']:.4f}
    Test Recall:    {test_metrics['recall']:.4f}
    Test F1-Score:  {test_metrics['f1']:.4f}
    
    Total Images: {len(collected_df):,}
      Train: {len(train_df):,}
      Val:   {len(val_df):,}
      Test:  {len(test_df):,}
    
    Best Val Acc: {max(history['val_acc']):.4f}
    Final Val Acc: {history['val_acc'][-1]:.4f}
    
    Model: {CONFIG['backbone']}
    CBAM: {CONFIG['use_cbam']}
    Enhanced Head: {CONFIG['use_better_head']}
    SCE Loss: {CONFIG['use_sce_loss']}
    EMA: {CONFIG['use_ema']}
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Training & Test Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIRS["plots"], "results.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    logging.info(f"✓ Results saved: {save_path}")

plot_results(history, test_metrics)

# Classification report
class_names = [LABELS[i]['name'] for i in sorted(LABELS.keys())]
report = classification_report(test_metrics['y_true'], test_metrics['y_pred'], 
                               target_names=class_names, digits=4)
report_path = os.path.join(OUTPUT_DIRS["results"], 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write(f"Test Results\n")
    f.write("="*60 + "\n\n")
    f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n\n")
    f.write(report)

logging.info(f"\n✓ Classification report saved: {report_path}")

# %%
logging.info("\n" + "="*80)
logging.info("TRAINING COMPLETE")
logging.info("="*80)
logging.info(f"Output directory: {PATH_OUTPUT}")
logging.info(f"Best model: {best_checkpoint}")
logging.info("="*80)