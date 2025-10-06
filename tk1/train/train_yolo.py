"""
Code train YOLOv8 # Model configuration
MODEL_SIZE = "yolov8n.pt"  # yolov8n/s/m/l/x
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 8   # Giảm xuống 8 do RAM thấp
DEVICE = 0  # 0 cho GPU, 'cpu' cho CPU
WORKERS = 2  # Giảm xuống 2 để tiết kiệm RAMct vùng bệnh trên lá lúa
Dataset format: YOLO (train/valid/test + data.yaml)
3 classes: bacterial_leaf_blight, blast, brown_spot
"""

from ultralytics import YOLO
import os
import yaml
import cv2
import glob
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# ================= CẤU HÌNH =================
# Đường dẫn
DATA_YAML = "../data/rice-leaf-disease-detection_new_yolo/data.yaml"  # Thay bằng đường dẫn thực tế
DATASET_ROOT = "../data/rice-leaf-disease-detection_new_yolo"          # Thư mục gốc chứa train/valid/test

# Model configuration
MODEL_SIZE = "yolov8n.pt"  # yolov8n/s/m/l/x
EPOCHS = 150
IMG_SIZE = 640
BATCH_SIZE = 64
DEVICE = 0  # 0 cho GPU, 'cpu' cho CPU
WORKERS = 8

# Project settings
PROJECT_NAME = "../output/yolo"         
EXPERIMENT_NAME = f"yolov8n_leaf_disease_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# Early stopping
PATIENCE = 20  # Dừng sớm nếu không cải thiện sau 20 epochs

# ================= KIỂM TRA DATASET =================
def check_dataset_structure(data_yaml_path):
    """Kiểm tra cấu trúc dataset và data.yaml"""
    print("=" * 60)
    print("KIỂM TRA DATASET")
    print("=" * 60)
    
    # Đọc data.yaml
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\n📋 Thông tin từ data.yaml:")
    print(f"   - Số classes: {data_config['nc']}")
    print(f"   - Tên classes: {data_config['names']}")
    print(f"   - Train path: {data_config['train']}")
    print(f"   - Valid path: {data_config['val']}")
    print(f"   - Test path: {data_config.get('test', 'Không có')}")
    
    # Kiểm tra từng split
    dataset_root = Path(data_yaml_path).parent
    
    splits = {
        'train': data_config['train'],
        'valid': data_config['val'],
        'test': data_config.get('test', None)
    }
    
    for split_name, split_path in splits.items():
        if split_path is None:
            continue
            
        # Xử lý đường dẫn tương đối
        if split_path.startswith('../'):
            img_dir = dataset_root / split_path.replace('../', '')
        else:
            img_dir = dataset_root / split_path
        
        # Với cấu trúc mới, chỉ có train folder
        if split_name in ['valid', 'test'] and split_path == 'train/images':
            print(f"\n✅ {split_name.upper()}: Sẽ được YOLO tự động chia từ train dataset")
            continue
        
        label_dir = img_dir.parent / 'labels'
        
        if img_dir.exists():
            images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
            labels = list(label_dir.glob('*.txt')) if label_dir.exists() else []
            
            print(f"\n✅ {split_name.upper()}:")
            print(f"   - Số ảnh: {len(images)}")
            print(f"   - Số labels: {len(labels)}")
            print(f"   - Thư mục images: {img_dir}")
            print(f"   - Thư mục labels: {label_dir}")
        else:
            print(f"\n❌ {split_name.upper()}: Không tìm thấy thư mục {img_dir}")
    
    print("\n" + "=" * 60)
    return data_config

def validate_labels(dataset_root, split='train', num_samples=5):
    """Kiểm tra chất lượng labels"""
    print(f"\n🔍 KIỂM TRA LABELS ({split.upper()}):")
    print("-" * 60)
    
    img_dir = Path(dataset_root) / split / 'images'
    label_dir = Path(dataset_root) / split / 'labels'
    
    if not img_dir.exists() or not label_dir.exists():
        print(f"❌ Không tìm thấy thư mục {split}")
        return
    
    image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    
    issues = []
    checked = 0
    
    for img_path in image_files[:num_samples]:
        label_path = label_dir / (img_path.stem + '.txt')
        
        if not label_path.exists():
            issues.append(f"❌ Thiếu label: {img_path.name}")
            continue
        
        # Đọc label
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        print(f"\n📄 {img_path.name}:")
        
        if len(lines) == 0:
            issues.append(f"⚠️  Label rỗng: {img_path.name}")
            print(f"   ⚠️  File label rỗng!")
            continue
        
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                issues.append(f"❌ Sai format: {img_path.name}, dòng {i+1}")
                print(f"   ❌ Dòng {i+1}: Sai format (cần 5 giá trị)")
                continue
            
            class_id, cx, cy, w, h = map(float, parts)
            
            # Kiểm tra giá trị
            if class_id not in [0, 1, 2]:
                issues.append(f"❌ Class ID sai: {img_path.name}, class={class_id}")
            
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                issues.append(f"❌ Tọa độ ngoài [0,1]: {img_path.name}")
                print(f"   ❌ Dòng {i+1}: Tọa độ ngoài phạm vi [0,1]")
            else:
                class_names = ['bacterial_leaf_blight', 'blast', 'brown_spot']
                print(f"   ✅ Bbox {i+1}: class={int(class_id)} ({class_names[int(class_id)]}), "
                      f"cx={cx:.3f}, cy={cy:.3f}, w={w:.3f}, h={h:.3f}")
        
        checked += 1
    
    print(f"\n📊 Kết quả kiểm tra {checked} file:")
    if issues:
        print(f"⚠️  Tìm thấy {len(issues)} vấn đề:")
        for issue in issues[:10]:  # Hiện 10 lỗi đầu
            print(f"   {issue}")
    else:
        print("✅ Tất cả labels đều hợp lệ!")
    
    print("-" * 60)

def visualize_sample(dataset_root, split='train', num_samples=3):
    """Visualize một số sample với bounding boxes"""
    print(f"\n🖼️  VISUALIZE SAMPLES ({split.upper()}):")
    print("-" * 60)
    
    img_dir = Path(dataset_root) / split / 'images'
    label_dir = Path(dataset_root) / split / 'labels'
    output_dir = Path('dataset_visualization') / split
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    class_names = ['bacterial_leaf_blight', 'blast', 'brown_spot']
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # BGR
    
    for img_path in image_files[:num_samples]:
        label_path = label_dir / (img_path.stem + '.txt')
        
        if not label_path.exists():
            continue
        
        # Đọc ảnh
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        # Đọc labels và vẽ bbox
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            class_id, cx, cy, bw, bh = map(float, parts)
            class_id = int(class_id)
            
            # Chuyển về pixel coordinates
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            
            # Vẽ bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[class_id], 2)
            
            # Vẽ label
            label_text = f"{class_names[class_id]}"
            cv2.putText(img, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)
        
        # Lưu ảnh
        output_path = output_dir / f"viz_{img_path.name}"
        cv2.imwrite(str(output_path), img)
        print(f"✅ Đã lưu: {output_path}")
    
    print(f"\n💾 Các ảnh visualization đã lưu tại: {output_dir}")
    print("-" * 60)

# ================= TRAIN MODEL =================
def train_yolo():
    """Train YOLO model với cấu hình tối ưu"""
    print("\n" + "=" * 60)
    print("BẮT ĐẦU TRAINING YOLO")
    print("=" * 60)
    
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(PROJECT_NAME, exist_ok=True)
    
    # Load pretrained model
    model = YOLO(MODEL_SIZE)
    
    print(f"\n📊 Cấu hình training:")
    print(f"   - Model: {MODEL_SIZE}")
    print(f"   - Epochs: {EPOCHS}")
    print(f"   - Image size: {IMG_SIZE}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Device: {DEVICE}")
    print(f"   - Patience: {PATIENCE}")
    
    # Train
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        
        # Project settings
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        exist_ok=True,  # Cho phép overwrite
        
        # Dataset split - YOLO sẽ tự động chia train/val với tỉ lệ 80/20
        split=0.2,  # 20% cho validation, 80% cho training
        
        # Early stopping
        patience=PATIENCE,
        
        # Save settings
        save=True,
        save_period=20,  # Lưu checkpoint mỗi 20 epochs thay vì 10
        
        # Memory optimization
        cache=False,  # Không cache images trong RAM
        rect=False,   # Tắt rectangular training để tiết kiệm memory
        
        # Validation
        val=True,
        
        # Augmentation parameters (giảm để tiết kiệm memory)
        hsv_h=0.01,       # Hue augmentation (giảm)
        hsv_s=0.5,        # Saturation augmentation (giảm)
        hsv_v=0.3,        # Value augmentation (giảm)
        degrees=0.0,      # Rotation
        translate=0.05,   # Translation (giảm)
        scale=0.3,        # Scaling (giảm)
        shear=0.0,        # Shearing
        perspective=0.0,  # Perspective
        flipud=0.0,       # Flip up-down (lá lúa không nên flip)
        fliplr=0.5,       # Flip left-right
        mosaic=0.8,       # Mosaic augmentation (giảm từ 1.0 xuống 0.8)
        mixup=0.05,       # Mixup augmentation (giảm từ 0.1 xuống 0.05)
        
        # Optimizer
        optimizer='auto',
        lr0=0.01,         # Initial learning rate
        lrf=0.01,         # Final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Other
        verbose=True,
        seed=0,
        deterministic=False,
        plots=True,       # Tạo training plots
        amp=True          # Automatic Mixed Precision
    )
    
    print("\n✅ TRAINING HOÀN TẤT!")
    print(f"📁 Kết quả lưu tại: {PROJECT_NAME}/{EXPERIMENT_NAME}/")
    
    return model, results

# ================= VALIDATE MODEL =================
def validate_model(weights_path=None):
    """Validate model trên validation set"""
    print("\n" + "=" * 60)
    print("VALIDATE MODEL")
    print("=" * 60)
    
    if weights_path is None:
        weights_path = f"{PROJECT_NAME}/{EXPERIMENT_NAME}/weights/best.pt"
    
    model = YOLO(weights_path)
    
    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        plots=True,
        save_json=True,
        save_hybrid=False,
        conf=0.001,
        iou=0.6,
        max_det=300,
        split='val'
    )
    
    print(f"\n📊 VALIDATION METRICS:")
    print(f"   - mAP50: {metrics.box.map50:.4f}")
    print(f"   - mAP50-95: {metrics.box.map:.4f}")
    
    # Handle numpy arrays - take mean if array, otherwise use scalar
    precision = metrics.box.p.mean() if hasattr(metrics.box.p, 'mean') else metrics.box.p
    recall = metrics.box.r.mean() if hasattr(metrics.box.r, 'mean') else metrics.box.r
    
    print(f"   - Precision: {precision:.4f}")
    print(f"   - Recall: {recall:.4f}")
    
    # Per-class metrics
    print(f"\n📋 Per-class mAP50:")
    class_names = ['bacterial_leaf_blight', 'blast', 'brown_spot']
    
    # Handle per-class maps safely
    if hasattr(metrics.box, 'maps') and metrics.box.maps is not None:
        maps = metrics.box.maps
        if hasattr(maps, '__len__') and len(maps) >= len(class_names):
            for i, name in enumerate(class_names):
                print(f"   - {name}: {maps[i]:.4f}")
        else:
            print(f"   - Overall mAP50: {metrics.box.map50:.4f}")
    else:
        print(f"   - Overall mAP50: {metrics.box.map50:.4f}")
    
    return metrics

# ================= OUTPUT STRUCTURE =================
def get_output_folder(parent_dir: str, env_name: str) -> str:
    """Tạo thư mục output với timestamp"""
    # Đảm bảo parent_dir tồn tại
    os.makedirs(parent_dir, exist_ok=True)
    
    # Tạo tên thư mục với timestamp
    experiment_id = f"{env_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir = os.path.join(parent_dir, experiment_id)
    
    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Đã tạo thư mục output: {output_dir}")
    return output_dir

def create_output_structure(base_path: str, class_names: list) -> dict:
    """Tạo cấu trúc thư mục output"""
    folders = {
        '01_originals': {},
        '02_vegetation_masks': {},
        '03_disease_masks': {},
        '04_detected_results': {},
        '05_cropped_diseases': {},  # Thêm folder cho vùng bệnh đã crop
        '06_statistics': base_path,
        '07_config': base_path
    }
    
    folder_paths = {}
    for folder_type, _ in folders.items():
        if folder_type in ['06_statistics', '07_config']:
            folder_path = os.path.join(base_path, folder_type)
            os.makedirs(folder_path, exist_ok=True)
            folder_paths[folder_type] = folder_path
        else:
            for class_name in class_names:
                folder_path = os.path.join(base_path, folder_type, class_name)
                os.makedirs(folder_path, exist_ok=True)
                folder_paths[f"{folder_type}_{class_name}"] = folder_path
    
    print(f"\n📁 Đã tạo cấu trúc thư mục tại: {base_path}")
    return folder_paths

# ================= CROP DISEASE REGIONS =================
def crop_disease_regions(image_path, label_path, output_folders, class_names, 
                        img_name, min_area=100, padding=10):
    """
    Crop các vùng bệnh từ ảnh và lưu vào thư mục theo class
    
    Args:
        image_path: Đường dẫn ảnh gốc
        label_path: Đường dẫn file label (.txt)
        output_folders: Dictionary chứa đường dẫn các thư mục output
        class_names: List tên các class
        img_name: Tên file ảnh (không có extension)
        min_area: Diện tích tối thiểu của bbox để crop (pixel^2)
        padding: Số pixel padding xung quanh bbox
    
    Returns:
        List các thông tin về cropped regions
    """
    # Đọc ảnh
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Không đọc được ảnh: {image_path}")
        return []
    
    h, w = img.shape[:2]
    
    # Đọc labels
    if not os.path.exists(label_path):
        return []
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    cropped_info = []
    
    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        class_id = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])
        conf = float(parts[5]) if len(parts) > 5 else 1.0
        
        # Chuyển về pixel coordinates
        x_center = int(cx * w)
        y_center = int(cy * h)
        box_w = int(bw * w)
        box_h = int(bh * h)
        
        # Tính tọa độ crop với padding
        x1 = max(0, x_center - box_w//2 - padding)
        y1 = max(0, y_center - box_h//2 - padding)
        x2 = min(w, x_center + box_w//2 + padding)
        y2 = min(h, y_center + box_h//2 + padding)
        
        # Kiểm tra diện tích
        crop_area = (x2 - x1) * (y2 - y1)
        if crop_area < min_area:
            continue
        
        # Crop vùng bệnh
        cropped = img[y1:y2, x1:x2]
        
        # Tên file crop
        class_name = class_names[class_id]
        crop_filename = f"{img_name}_crop{idx:03d}_{class_name}_conf{conf:.2f}.jpg"
        
        # Lưu vào thư mục tương ứng
        crop_path = os.path.join(
            output_folders[f'05_cropped_diseases_{class_name}'], 
            crop_filename
        )
        cv2.imwrite(crop_path, cropped)
        
        # Lưu thông tin
        cropped_info.append({
            'class_id': class_id,
            'class_name': class_name,
            'bbox': [x1, y1, x2, y2],
            'confidence': conf,
            'crop_path': crop_path,
            'crop_size': (x2-x1, y2-y1)
        })
    
    return cropped_info

# ================= PREDICT WITH FULL PIPELINE =================
def predict_and_crop(source_path, weights_path=None, output_parent='../output/yolo/inference', 
                     conf=0.25, min_crop_area=100, padding=10):
    """
    Pipeline đầy đủ: Detect -> Visualize -> Crop -> Statistics
    
    Args:
        source_path: Đường dẫn ảnh hoặc thư mục ảnh
        weights_path: Đường dẫn weights
        output_parent: Thư mục cha chứa output
        conf: Confidence threshold
        min_crop_area: Diện tích tối thiểu để crop
        padding: Padding xung quanh bbox khi crop
    """
    print("\n" + "=" * 60)
    print("DETECTION & CROPPING PIPELINE")
    print("=" * 60)
    
    # Load model
    if weights_path is None:
        weights_path = f"{PROJECT_NAME}/{EXPERIMENT_NAME}/weights/best.pt"
    
    model = YOLO(weights_path)
    
    # Class names
    class_names = ['bacterial_leaf_blight', 'blast', 'brown_spot']
    
    # Tạo thư mục output parent nếu chưa tồn tại
    os.makedirs(output_parent, exist_ok=True)
    
    # Tạo output folder
    output_dir = get_output_folder(output_parent, 'rice_disease_detection')
    folder_paths = create_output_structure(output_dir, class_names)
    
    print(f"\n🔮 Bắt đầu detection với:")
    print(f"   - Weights: {weights_path}")
    print(f"   - Source: {source_path}")
    print(f"   - Confidence: {conf}")
    print(f"   - Output: {output_dir}")
    
    # Lấy danh sách ảnh
    source_path = Path(source_path)
    if source_path.is_file():
        image_files = [source_path]
    else:
        image_files = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png'))
    
    print(f"\n📸 Tìm thấy {len(image_files)} ảnh")
    
    # Statistics
    stats = {
        'total_images': len(image_files),
        'images_with_disease': 0,
        'total_detections': 0,
        'detections_per_class': {name: 0 for name in class_names},
        'cropped_per_class': {name: 0 for name in class_names},
        'processing_time': 0
    }
    
    start_time = time.time()
    
    # Process từng ảnh
    for img_idx, img_path in enumerate(image_files, 1):
        print(f"\n[{img_idx}/{len(image_files)}] Processing: {img_path.name}")
        
        img_name = img_path.stem
        
        # 1. Copy ảnh gốc
        img = cv2.imread(str(img_path))
        for class_name in class_names:
            original_path = os.path.join(
                folder_paths[f'01_originals_{class_name}'],
                f"{img_name}.jpg"
            )
            cv2.imwrite(original_path, img)
        
        # 2. Detect
        results = model.predict(
            source=str(img_path),
            imgsz=IMG_SIZE,
            conf=conf,
            iou=0.45,
            device=DEVICE,
            verbose=False
        )
        
        result = results[0]
        boxes = result.boxes
        
        if len(boxes) == 0:
            print(f"   ℹ️  Không phát hiện bệnh")
            continue
        
        stats['images_with_disease'] += 1
        stats['total_detections'] += len(boxes)
        
        # 3. Vẽ bbox và lưu
        annotated = result.plot()
        
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            conf_score = float(box.conf[0])
            
            # Lưu vào folder detected_results
            detected_path = os.path.join(
                folder_paths[f'04_detected_results_{class_name}'],
                f"{img_name}.jpg"
            )
            cv2.imwrite(detected_path, annotated)
            
            stats['detections_per_class'][class_name] += 1
            
            print(f"   ✅ Phát hiện: {class_name} (conf: {conf_score:.2f})")
        
        # 4. Tạo label file tạm để crop
        temp_label_path = f"temp_{img_name}.txt"
        with open(temp_label_path, 'w') as f:
            for box in boxes:
                class_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                
                # YOLO format: class cx cy w h
                xywhn = box.xywhn[0].cpu().numpy()
                f.write(f"{class_id} {xywhn[0]} {xywhn[1]} {xywhn[2]} {xywhn[3]} {conf_score}\n")
        
        # 5. Crop disease regions
        cropped_info = crop_disease_regions(
            img_path, 
            temp_label_path,
            folder_paths,
            class_names,
            img_name,
            min_area=min_crop_area,
            padding=padding
        )
        
        # Xóa temp label
        if os.path.exists(temp_label_path):
            os.remove(temp_label_path)
        
        # Update stats
        for crop in cropped_info:
            stats['cropped_per_class'][crop['class_name']] += 1
        
        print(f"   📦 Đã crop {len(cropped_info)} vùng bệnh")
    
    # 6. Lưu statistics
    stats['processing_time'] = time.time() - start_time
    
    stats_path = os.path.join(folder_paths['06_statistics'], 'statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Tạo báo cáo text
    report_path = os.path.join(folder_paths['06_statistics'], 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("RICE LEAF DISEASE DETECTION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {weights_path}\n")
        f.write(f"Confidence threshold: {conf}\n\n")
        
        f.write(f"📊 TỔNG QUAN:\n")
        f.write(f"   - Tổng số ảnh: {stats['total_images']}\n")
        f.write(f"   - Ảnh có bệnh: {stats['images_with_disease']}\n")
        f.write(f"   - Tổng vùng bệnh phát hiện: {stats['total_detections']}\n")
        f.write(f"   - Thời gian xử lý: {stats['processing_time']:.2f}s\n\n")
        
        f.write(f"📋 PHÁT HIỆN THEO LOẠI BỆNH:\n")
        for class_name in class_names:
            f.write(f"   - {class_name}: {stats['detections_per_class'][class_name]} detections, "
                   f"{stats['cropped_per_class'][class_name]} cropped images\n")
    
    # 7. Lưu config
    config_path = os.path.join(folder_paths['07_config'], 'config.yaml')
    config_data = {
        'model_weights': weights_path,
        'confidence_threshold': conf,
        'image_size': IMG_SIZE,
        'min_crop_area': min_crop_area,
        'padding': padding,
        'class_names': class_names,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)
    
    # In kết quả
    print("\n" + "=" * 60)
    print("✅ HOÀN TẤT DETECTION & CROPPING!")
    print("=" * 60)
    print(f"\n📊 KẾT QUẢ:")
    print(f"   - Tổng ảnh: {stats['total_images']}")
    print(f"   - Ảnh có bệnh: {stats['images_with_disease']}")
    print(f"   - Tổng vùng bệnh: {stats['total_detections']}")
    print(f"\n📋 CROPPED THEO LOẠI BỆNH:")
    for class_name in class_names:
        count = stats['cropped_per_class'][class_name]
        print(f"   - {class_name}: {count} ảnh")
    print(f"\n📁 OUTPUT: {output_dir}")
    print(f"   - Ảnh gốc: 01_originals/")
    print(f"   - Ảnh đã detect: 04_detected_results/")
    print(f"   - Vùng bệnh đã crop: 05_cropped_diseases/")
    print(f"   - Statistics: 06_statistics/")
    print(f"\n⏱️  Thời gian: {stats['processing_time']:.2f}s")
    
    return output_dir, stats

# ================= EXPORT MODEL =================
def export_model(weights_path=None, formats=['onnx']):
    """Export model sang các format khác"""
    print("\n" + "=" * 60)
    print("EXPORT MODEL")
    print("=" * 60)
    
    if weights_path is None:
        weights_path = f"{PROJECT_NAME}/{EXPERIMENT_NAME}/weights/best.pt"
    
    model = YOLO(weights_path)
    
    for fmt in formats:
        print(f"\n📦 Export sang {fmt.upper()}...")
        exported_path = model.export(
            format=fmt,
            imgsz=IMG_SIZE,
            half=False,  # FP16 quantization
            dynamic=False,
            simplify=True  # Simplify ONNX
        )
        print(f"✅ Đã export: {exported_path}")
    
    print("\n" + "=" * 60)

# ================= RESUME TRAINING =================
def resume_training(checkpoint_path=None):
    """Resume training từ checkpoint"""
    if checkpoint_path is None:
        checkpoint_path = f"{PROJECT_NAME}/{EXPERIMENT_NAME}/weights/last.pt"
    
    print(f"\n🔄 Resume training từ: {checkpoint_path}")
    
    model = YOLO(checkpoint_path)
    
    results = model.train(
        resume=True
    )
    
    return model, results

# ================= MAIN WORKFLOW =================
def main():
    """Main workflow - Full training pipeline"""
    print("\n" + "=" * 60)
    print("YOLO TRAINING PIPELINE - RICE LEAF DISEASE DETECTION")
    print("=" * 60)
    
    # 1. Kiểm tra dataset
    check_dataset_structure(DATA_YAML)
    validate_labels(DATASET_ROOT, split='train', num_samples=5)
    # validate_labels(DATASET_ROOT, split='valid', num_samples=3)  # Không cần vì YOLO sẽ tự chia
    
    # 2. Visualize samples (optional)
    # visualize_sample(DATASET_ROOT, split='train', num_samples=3)
    
    # 3. Train model
    model, results = train_yolo()
    
    # 4. Validate model
    metrics = validate_model()
    
    # 5. Export model
    export_model(formats=['onnx'])
    
    print("\n" + "=" * 60)
    print("🎉 HOÀN THÀNH TRAINING!")
    print("=" * 60)
    print(f"\n📂 Kết quả:")
    print(f"   - Best weights: {PROJECT_NAME}/{EXPERIMENT_NAME}/weights/best.pt")
    print(f"   - Training plots: {PROJECT_NAME}/{EXPERIMENT_NAME}/results.png")
    print(f"   - Confusion matrix: {PROJECT_NAME}/{EXPERIMENT_NAME}/confusion_matrix.png")
    print(f"   - ONNX model: {PROJECT_NAME}/{EXPERIMENT_NAME}/weights/best.onnx")

def main_inference(source_path, weights_path=None, conf=0.25):
    """Main workflow - Inference với crop disease regions"""
    print("\n" + "=" * 60)
    print("INFERENCE PIPELINE - DETECT & CROP DISEASE REGIONS")
    print("=" * 60)
    
    # Run prediction và crop
    output_dir, stats = predict_and_crop(
        source_path=source_path,
        weights_path=weights_path,
        output_parent='../output/yolo/inference',
        conf=conf,
        min_crop_area=100,
        padding=10
    )
    
    print("\n" + "=" * 60)
    print("🎉 HOÀN THÀNH INFERENCE!")
    print("=" * 60)
    
    return output_dir, stats

if __name__ == "__main__":
    
    # ===== CHỌN MỘT TRONG CÁC WORKFLOW BẤN DƯỚI =====
    
    # ============================================================
    # WORKFLOW 1: TRAINING (chạy một lần đầu tiên)
    # ============================================================
    main()
    
    
    # ============================================================
    # WORKFLOW 2: INFERENCE - DETECT & CROP (sau khi đã train)
    # ============================================================
    # Chạy detection trên ảnh mới và tự động crop vùng bệnh
    
    # Ví dụ: detect trên thư mục test
    # output_dir, stats = main_inference(
    #     source_path="dataset/test/images",
    #     weights_path="runs_rice_leaf/yolov8n_leaf_disease_v1/weights/best.pt",
    #     conf=0.25
    # )
    
    # Ví dụ: detect trên thư mục ảnh mới
    # output_dir, stats = main_inference(
    #     source_path="path/to/new_images",
    #     conf=0.3  # Có thể điều chỉnh confidence
    # )
    
    
    # ============================================================
    # WORKFLOW 3: CHẠY TỪNG PHẦN (tùy chỉnh)
    # ============================================================
    
    # --- Chỉ kiểm tra dataset ---
    # check_dataset_structure(DATA_YAML)
    # validate_labels(DATASET_ROOT, 'train', num_samples=10)
    # visualize_sample(DATASET_ROOT, 'train', num_samples=5)
    
    # --- Chỉ train ---
    # model, results = train_yolo()
    
    # --- Chỉ validate ---
    # metrics = validate_model("runs_rice_leaf/yolov8n_leaf_disease_v1/weights/best.pt")
    
    # --- Chỉ detect và crop (tùy chỉnh chi tiết) ---
    # output_dir, stats = predict_and_crop(
    #     source_path="dataset/test/images",
    #     weights_path="runs_rice_leaf/yolov8n_leaf_disease_v1/weights/best.pt",
    #     output_parent="outputs",
    #     conf=0.25,
    #     min_crop_area=100,  # Diện tích tối thiểu (pixel^2)
    #     padding=10          # Padding xung quanh bbox
    # )
    
    # --- Resume training ---
    # model, results = resume_training()
    
    # --- Export sang nhiều format ---
    # export_model(formats=['onnx', 'torchscript', 'tflite'])
    
    pass