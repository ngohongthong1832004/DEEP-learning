# ===== RICE LEAF DISEASE PREDICTION WITH YOLO PREPROCESSING =====
import os
import csv
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from pathlib import Path
import gc
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# YOLO
from ultralytics import YOLO

# ===== CONFIGURATION =====
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')

# YOLO Config
YOLO_CONFIG = {
    'checkpoint': '/home/bbsw/thong/deep_learning/tk1/output/yolov8n_leaf_disease_2025-10-06_01-23-16/best.pt',
    'conf_thresh': 0.15,
    'iou_thresh': 0.35,
    'img_size': 640,
    'max_det': 300,
}

# Classifier Config
OUTPUT_DIRS = {
    "weights": "../output/multi-model-classifier-v2_2025-10-06_11-48-35/weights",
    "results": "../output/predict/results",
    "crops": "../output/predict/crops",  # Lưu crops để debug
}

CONFIG = {
    'img_size': 224,
    'batch_size': 32,
    'num_workers': 2,
    'save_crops': False,  # True = lưu crops để debug
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LABELS = {
    0: {'name': 'brown_spot'},
    1: {'name': 'blast'},
    2: {'name': 'bacterial_leaf_blight'},
    3: {'name': 'normal'},
}


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
def build_classifier(backbone_name: str, num_classes: int, use_cbam: bool = True, use_better_head: bool = True):
    """Build classifier - khớp với training code"""
    from torchvision import models
    
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
        features = nn.Sequential(*list(base_model.children())[:-1])
        feat_channels = 1024
        
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    if use_cbam:
        cbam = CBAM(feat_channels, reduction=16)
        features = nn.Sequential(*list(features.children()), cbam)
    
    model = nn.Module()
    model.features = features
    
    if use_better_head:
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
    
    return model


def get_val_transform(img_size: int = 224):
    """Transform cho inference."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ===== YOLO DETECTION & CROPPING =====
class YOLODetector:
    """YOLO detector để crop vùng bệnh"""
    
    def __init__(self, checkpoint_path: str, conf_thresh: float = 0.15, 
                 iou_thresh: float = 0.35, img_size: int = 640, max_det: int = 300):
        self.model = YOLO(checkpoint_path)
        self.model.to(DEVICE)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        self.max_det = max_det
        
        logging.info(f"✅ Loaded YOLO model: {checkpoint_path}")
    
    def detect_and_crop(self, image_path: str, min_area: int = 100) -> List[np.ndarray]:
        """
        Detect và crop các vùng bệnh từ ảnh.
        
        Returns:
            List[np.ndarray]: Danh sách các crop (RGB format)
        """
        # Load ảnh
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            logging.warning(f"Cannot read image: {image_path}")
            return []
        
        # YOLO detection
        try:
            results = self.model.predict(
                source=image_path,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                imgsz=self.img_size,
                device=DEVICE,
                max_det=self.max_det,
                verbose=False
            )
        except Exception as e:
            logging.error(f"YOLO prediction failed for {image_path}: {e}")
            return []
        
        # Extract boxes
        crops = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.data.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Filter small boxes
                area = (x2 - x1) * (y2 - y1)
                if area < min_area:
                    continue
                
                # Crop (clamp to image bounds)
                h, w = img_bgr.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    crop_bgr = img_bgr[y1:y2, x1:x2]
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    crops.append(crop_rgb)
        
        return crops


# ===== DATASET FOR CROPS =====
class CropDataset(Dataset):
    """Dataset cho các crops đã detect"""
    
    def __init__(self, crops: List[np.ndarray], transform):
        self.crops = crops
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.crops)
    
    def __getitem__(self, idx: int):
        crop = self.crops[idx]
        img = Image.fromarray(crop)
        if self.transform:
            img = self.transform(img)
        return img


# ===== HELPER FUNCTIONS =====
def collect_images(input_dir: str) -> List[str]:
    """Thu thập tất cả đường dẫn ảnh."""
    if not os.path.exists(input_dir):
        raise ValueError(f"Thư mục không tồn tại: {input_dir}")
    
    paths = []
    for name in sorted(os.listdir(input_dir)):
        if name.lower().endswith(IMAGE_EXTS):
            paths.append(os.path.join(input_dir, name))
    return paths


def make_label_mapper(mode: str = "binary_blast", custom_map: Dict[str, str] = None):
    """Tạo hàm mapping label."""
    id2name = {i: LABELS[i]['name'] for i in sorted(LABELS.keys())}

    if custom_map is not None:
        def mapper(cls_idx: int) -> str:
            return custom_map.get(id2name[cls_idx], id2name[cls_idx])
        return mapper

    if mode == "binary_blast":
        def mapper(cls_idx: int) -> str:
            name = id2name[cls_idx]
            return "blast" if name == "leaf_blast" else "normal"
        return mapper

    def mapper(cls_idx: int) -> str:
        return id2name[cls_idx]
    return mapper


def aggregate_predictions(pred_indices: List[int], pred_probs: np.ndarray, 
                         method: str = "max_confidence") -> Tuple[int, float]:
    """
    Aggregate predictions từ nhiều crops.
    
    Args:
        pred_indices: List of predicted class indices
        pred_probs: Probability matrix (N, num_classes)
        method: "max_confidence" or "majority_vote"
    
    Returns:
        (final_class_idx, confidence)
    """
    if len(pred_indices) == 0:
        return 3, 0.0  # Default: healthy with 0 confidence
    
    if method == "max_confidence":
        # Chọn prediction có confidence cao nhất
        max_idx = np.argmax(pred_probs.max(axis=1))
        final_class = pred_indices[max_idx]
        confidence = pred_probs[max_idx, final_class]
        return final_class, confidence
    
    elif method == "majority_vote":
        # Vote theo số lượng
        from collections import Counter
        votes = Counter(pred_indices)
        final_class = votes.most_common(1)[0][0]
        # Confidence = average của class đó
        mask = np.array(pred_indices) == final_class
        confidence = pred_probs[mask, final_class].mean()
        return final_class, confidence
    
    else:
        raise ValueError(f"Unknown method: {method}")


# ===== PREDICTION WITH YOLO =====
@torch.no_grad()
def predict_single_model_with_yolo(
    input_dir: str,
    output_csv: str,
    model_name: str,
    checkpoint_path: str,
    yolo_detector: YOLODetector,
    img_size: int = 224,
    batch_size: int = 32,
    label_mode: str = "binary_blast",
    custom_label_map: Dict[str, str] = None,
    aggregation_method: str = "max_confidence",
    save_crops: bool = False,
    timestamp: str = None,
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Dự đoán với YOLO preprocessing.
    """
    device = DEVICE
    
    # Thu thập ảnh
    image_paths = collect_images(input_dir)
    if len(image_paths) == 0:
        raise ValueError(f"Không tìm thấy ảnh trong: {input_dir}")
    
    logging.info(f"📸 Tìm thấy {len(image_paths)} ảnh trong {input_dir}")

    # Load classifier
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint không tồn tại: {checkpoint_path}")
    
    num_classes = len(LABELS)
    model = build_classifier(model_name, num_classes, use_cbam=True, use_better_head=True)
    
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.to(device).eval()
    
    logging.info(f"✅ Đã load classifier: {model_name}")

    # Transform
    transform = get_val_transform(img_size)
    mapper = make_label_mapper(mode=label_mode, custom_map=custom_label_map)

    # Predict từng ảnh
    results = []
    crops_dir = Path(OUTPUT_DIRS["crops"]) / model_name if save_crops else None
    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in tqdm(image_paths, desc=f"🔮 Predicting - {model_name}"):
        fname = os.path.basename(img_path)
        
        # 1. YOLO detection & crop
        crops = yolo_detector.detect_and_crop(img_path)
        
        if len(crops) == 0:
            # Không detect được -> healthy
            label_str = mapper(3)  # healthy
            results.append((fname, label_str))
            continue
        
        # 2. Classify từng crop
        dataset = CropDataset(crops, transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        all_probs = []
        
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        all_probs = np.vstack(all_probs)
        
        # 3. Aggregate predictions
        final_class_idx, confidence = aggregate_predictions(
            all_preds, all_probs, method=aggregation_method
        )
        
        label_str = mapper(int(final_class_idx))
        results.append((fname, label_str))
        
        # 4. Save crops (optional, for debugging)
        if save_crops and crops_dir:
            img_crop_dir = crops_dir / Path(fname).stem
            img_crop_dir.mkdir(exist_ok=True)
            for i, crop in enumerate(crops):
                crop_path = img_crop_dir / f"crop_{i:03d}.jpg"
                Image.fromarray(crop).save(crop_path)

    # Thêm timestamp vào tên file nếu có
    if timestamp:
        base_path, ext = os.path.splitext(output_csv)
        output_csv = f"{base_path}_{timestamp}{ext}"
    
    # Ghi CSV
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(output_csv, "w", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "label"])
        writer.writerows(results)

    logging.info(f"💾 Đã lưu predictions: {output_csv}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return output_csv, results


@torch.no_grad()
def predict_multi_models_with_yolo(
    input_dir: str,
    output_dir: str,
    model_configs: List[Tuple[str, str]],
    yolo_checkpoint: str,
    img_size: int = 224,
    batch_size: int = 32,
    label_mode: str = "binary_blast",
    custom_label_map: Dict[str, str] = None,
    aggregation_method: str = "max_confidence",
    save_crops: bool = False,
    add_timestamp: bool = True,
) -> List[Tuple[str, str]]:
    """
    Chạy prediction cho nhiều models với YOLO preprocessing.
    """
    # Tạo timestamp cho session này
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if add_timestamp else None
    
    # Tạo output directory với timestamp nếu được bật
    if timestamp:
        timestamped_output_dir = os.path.join(output_dir, f"predictions_{timestamp}")
        os.makedirs(timestamped_output_dir, exist_ok=True)
        final_output_dir = timestamped_output_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
        final_output_dir = output_dir
    
    # Initialize YOLO detector (dùng chung cho tất cả models)
    yolo_detector = YOLODetector(
        yolo_checkpoint,
        conf_thresh=YOLO_CONFIG['conf_thresh'],
        iou_thresh=YOLO_CONFIG['iou_thresh'],
        img_size=YOLO_CONFIG['img_size'],
        max_det=YOLO_CONFIG['max_det']
    )
    
    results = []
    total = len(model_configs)
    
    print("\n" + "="*70)
    print(f"🚀 BẮT ĐẦU PREDICTION VỚI {total} MODELS + YOLO PREPROCESSING")
    print("="*70)
    
    for idx, (model_name, checkpoint_path) in enumerate(model_configs, 1):
        print(f"\n[{idx}/{total}] Model: {model_name}")
        print("-"*70)
        
        try:
            safe_name = model_name.replace('/', '_').replace('\\', '_')
            csv_path = os.path.join(final_output_dir, f"predictions_{safe_name}.csv")
            
            csv_path, _ = predict_single_model_with_yolo(
                input_dir=input_dir,
                output_csv=csv_path,
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                yolo_detector=yolo_detector,
                img_size=img_size,
                batch_size=batch_size,
                label_mode=label_mode,
                custom_label_map=custom_label_map,
                aggregation_method=aggregation_method,
                save_crops=save_crops,
                timestamp=None,  # Timestamp đã được thêm vào folder
            )
            results.append((model_name, csv_path))
            print(f"✅ Hoàn thành: {model_name}")
            
        except Exception as e:
            logging.error(f"❌ Lỗi với {model_name}: {e}")
            print(f"❌ Lỗi: {e}")
            continue
    
    print("\n" + "="*70)
    print(f"🎉 HOÀN THÀNH! Đã tạo {len(results)}/{total} file CSV")
    if timestamp:
        print(f"📁 Thư mục kết quả: {final_output_dir}")
        print(f"⏰ Timestamp: {timestamp}")
    print("="*70)
    
    return results


# ===== MAIN =====
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*70)
    print("🌾 RICE LEAF DISEASE PREDICTION - WITH YOLO PREPROCESSING 🌾")
    print("="*70)
    print(f"📱 Device: {DEVICE}")
    print(f"🖼️  Classifier image size: {CONFIG['img_size']}")
    print(f"📦 Batch size: {CONFIG['batch_size']}")
    print(f"🎯 YOLO checkpoint: {YOLO_CONFIG['checkpoint']}")
    print("="*70)
    
    # ===== CẤU HÌNH =====
    INPUT_DIR = "../data/raw/paddy_disease_classification/test_images"
    OUTPUT_CSV_DIR = OUTPUT_DIRS["results"]
    
    MODEL_CONFIGS = [
        ('efficientnet_b0', os.path.join(OUTPUT_DIRS["weights"], 'efficientnet_b0_best.pth')),
        ('efficientnet_v2_s', os.path.join(OUTPUT_DIRS["weights"], 'efficientnet_v2_s_best.pth')),
        ('mobilenet_v3_large', os.path.join(OUTPUT_DIRS["weights"], 'mobilenet_v3_large_best.pth')),
        ('mobilenet_v3_small', os.path.join(OUTPUT_DIRS["weights"], 'mobilenet_v3_small_best.pth')),
        ('resnet18', os.path.join(OUTPUT_DIRS["weights"], 'resnet18_best.pth')),
        ('shufflenet_v2_x1_0', os.path.join(OUTPUT_DIRS["weights"], 'shufflenet_v2_x1_0_best.pth')),
    ]
    
    LABEL_MODE = "multiclass"  # Thay đổi từ "binary_blast" để hiển thị đầy đủ 4 class
    AGGREGATION_METHOD = "max_confidence"  # hoặc "majority_vote"
    
    # ===== CHẠY PREDICTION =====
    try:
        outputs = predict_multi_models_with_yolo(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_CSV_DIR,
            model_configs=MODEL_CONFIGS,
            yolo_checkpoint=YOLO_CONFIG['checkpoint'],
            img_size=CONFIG['img_size'],
            batch_size=CONFIG['batch_size'],
            label_mode=LABEL_MODE,
            aggregation_method=AGGREGATION_METHOD,
            save_crops=CONFIG['save_crops'],
        )
        
        print("\n" + "="*70)
        print("📊 DANH SÁCH FILE CSV ĐÃ TẠO:")
        print("="*70)
        for model_name, csv_path in outputs:
            print(f"✓ {model_name:30s} => {csv_path}")
        
        print(f"\n✨ Hoàn tất! Kiểm tra thư mục: {OUTPUT_CSV_DIR}")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ LỖI: {e}")
        logging.error(f"Lỗi chính: {e}", exc_info=True)