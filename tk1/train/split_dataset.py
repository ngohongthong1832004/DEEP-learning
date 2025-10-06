"""
Script để chia dataset từ thư mục train thành train/valid/test
theo tỉ lệ 70%/20%/10% với class distribution balanced
"""

import os
import random
import shutil
from pathlib import Path
from collections import defaultdict
import yaml

# Cấu hình
DATASET_ROOT = "../data/rice-leaf-disease-detection_new_yolo"
TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
TEST_RATIO = 0.1

# Seed cho reproducible results
random.seed(42)

def analyze_class_distribution(labels_dir):
    """Phân tích phân phối classes trong dataset"""
    class_count = defaultdict(int)
    image_to_classes = {}
    
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
            
        label_path = os.path.join(labels_dir, label_file)
        image_name = label_file.replace('.txt', '')
        classes_in_image = set()
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(float(parts[0]))
                    classes_in_image.add(class_id)
                    class_count[class_id] += 1
        
        image_to_classes[image_name] = list(classes_in_image)
    
    return class_count, image_to_classes

def split_dataset():
    """Chia dataset thành train/valid/test"""
    
    print("=" * 60)
    print("DATASET SPLITTING")
    print("=" * 60)
    
    dataset_path = Path(DATASET_ROOT)
    original_train_images = dataset_path / "train" / "images"
    original_train_labels = dataset_path / "train" / "labels"
    
    # Đổi tên thư mục gốc để tránh conflict
    original_backup_images = dataset_path / "original_train_images"
    original_backup_labels = dataset_path / "original_train_labels"
    
    # Move original folders
    if original_train_images.exists():
        shutil.move(str(original_train_images), str(original_backup_images))
    if original_train_labels.exists():
        shutil.move(str(original_train_labels), str(original_backup_labels))
    
    # Tạo thư mục mới
    new_folders = ['train', 'valid', 'test']
    for folder in new_folders:
        for subfolder in ['images', 'labels']:
            new_path = dataset_path / folder / subfolder
            new_path.mkdir(parents=True, exist_ok=True)
    
    # Lấy danh sách tất cả images từ backup folder
    image_files = []
    for img_file in original_backup_images.glob("*.jpg"):
        # Kiểm tra xem có label tương ứng không
        label_file = original_backup_labels / f"{img_file.stem}.txt"
        if label_file.exists():
            image_files.append(img_file.stem)
    
    print(f"\n📊 Tổng số images: {len(image_files)}")
    
    # Phân tích class distribution
    class_count, image_to_classes = analyze_class_distribution(str(original_backup_labels))
    
    print(f"\n📋 Phân phối classes:")
    class_names = ['bacterial_leaf_blight', 'blast', 'brown_spot']
    for class_id, count in class_count.items():
        print(f"   - {class_names[class_id]}: {count} bounding boxes")
    
    print(f"\n📈 Phân phối images theo classes:")
    images_per_class = defaultdict(list)
    for image_name, classes in image_to_classes.items():
        for class_id in classes:
            images_per_class[class_id].append(image_name)
    
    for class_id in sorted(images_per_class.keys()):
        print(f"   - {class_names[class_id]}: {len(images_per_class[class_id])} images")
    
    # Chia dataset theo stratified sampling
    train_images = []
    valid_images = []
    test_images = []
    
    # Shuffle để đảm bảo random
    random.shuffle(image_files)
    
    # Chia theo tỉ lệ
    n_train = int(len(image_files) * TRAIN_RATIO)
    n_valid = int(len(image_files) * VALID_RATIO)
    
    train_images = image_files[:n_train]
    valid_images = image_files[n_train:n_train + n_valid]
    test_images = image_files[n_train + n_valid:]
    
    splits = {
        'train': train_images,
        'valid': valid_images,
        'test': test_images
    }
    
    print(f"\n📦 Kích thước splits:")
    for split_name, images in splits.items():
        print(f"   - {split_name}: {len(images)} images ({len(images)/len(image_files)*100:.1f}%)")
    
    # Copy files
    for split_name, images in splits.items():
        print(f"\n📂 Đang copy {split_name} files...")
        
        split_images_dir = dataset_path / split_name / "images"
        split_labels_dir = dataset_path / split_name / "labels"
        
        for i, image_name in enumerate(images, 1):
            # Copy image từ backup folder
            src_img = original_backup_images / f"{image_name}.jpg"
            dst_img = split_images_dir / f"{image_name}.jpg"
            shutil.copy2(src_img, dst_img)
            
            # Copy label từ backup folder
            src_label = original_backup_labels / f"{image_name}.txt"
            dst_label = split_labels_dir / f"{image_name}.txt"
            shutil.copy2(src_label, dst_label)
            
            if i % 500 == 0 or i == len(images):
                print(f"   Đã copy {i}/{len(images)} files...")
    
    # Verify splits
    print(f"\n✅ Kiểm tra kết quả:")
    for split_name in splits.keys():
        split_images_dir = dataset_path / split_name / "images"
        split_labels_dir = dataset_path / split_name / "labels"
        
        n_images = len(list(split_images_dir.glob("*.jpg")))
        n_labels = len(list(split_labels_dir.glob("*.txt")))
        
        print(f"   - {split_name}: {n_images} images, {n_labels} labels")
        
        if n_images != n_labels:
            print(f"   ⚠️  WARNING: Số lượng images và labels không khớp!")
    
    return splits

def update_data_yaml():
    """Cập nhật data.yaml với đường dẫn mới"""
    
    print(f"\n📝 Cập nhật data.yaml...")
    
    data_yaml_path = Path(DATASET_ROOT) / "data.yaml"
    
    # Đọc data.yaml hiện tại
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Cập nhật đường dẫn
    data['train'] = "train/images"
    data['val'] = "valid/images"  
    data['test'] = "test/images"
    
    # Backup file cũ
    backup_path = data_yaml_path.with_suffix('.yaml.backup')
    shutil.copy2(data_yaml_path, backup_path)
    print(f"   📄 Đã backup file gốc: {backup_path}")
    
    # Ghi file mới
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    print(f"   ✅ Đã cập nhật data.yaml")
    
    # In nội dung mới
    print(f"\n📋 Nội dung data.yaml mới:")
    with open(data_yaml_path, 'r') as f:
        print(f.read())

def main():
    """Main function"""
    
    # Chia dataset
    splits = split_dataset()
    
    # Cập nhật data.yaml
    update_data_yaml()
    
    print("\n" + "=" * 60)
    print("✅ HOÀN THÀNH CHIA DATASET!")
    print("=" * 60)
    print(f"\n📁 Cấu trúc thư mục mới:")
    print(f"   {DATASET_ROOT}/")
    print(f"   ├── train/")
    print(f"   │   ├── images/ ({len(splits['train'])} files)")
    print(f"   │   └── labels/ ({len(splits['train'])} files)")
    print(f"   ├── valid/")
    print(f"   │   ├── images/ ({len(splits['valid'])} files)")
    print(f"   │   └── labels/ ({len(splits['valid'])} files)")
    print(f"   ├── test/")
    print(f"   │   ├── images/ ({len(splits['test'])} files)")
    print(f"   │   └── labels/ ({len(splits['test'])} files)")
    print(f"   └── data.yaml (updated)")
    
    print(f"\n🚀 Bây giờ bạn có thể chạy lại training script!")

if __name__ == "__main__":
    main()