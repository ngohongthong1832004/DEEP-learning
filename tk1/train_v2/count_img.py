import os
import csv
from collections import defaultdict

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')

def count_images_in_dir(path: str, recursive: bool = True) -> int:
    """Đếm số file ảnh trong path. Nếu recursive=True sẽ duyệt toàn bộ thư mục con."""
    if not os.path.exists(path):
        return 0
    if recursive:
        total = 0
        for root, _, files in os.walk(path):
            total += sum(1 for f in files if f.endswith(IMAGE_EXTS))
        return total
    else:
        try:
            return sum(1 for f in os.listdir(path) if f.endswith(IMAGE_EXTS))
        except Exception:
            return 0

def count_dataset(LABELS: dict, recursive: bool = True, save_csv: str = None):
    """
    Đếm số ảnh theo từng label và từng thư mục trong match_substrings.
    - LABELS: dict như bạn đã khai báo
    - recursive: duyệt đệ quy
    - save_csv: nếu truyền đường dẫn (vd: './counts.csv') sẽ ghi kết quả ra CSV
    """
    per_label_total = defaultdict(int)
    per_label_paths = defaultdict(list)  # list các tuple (path, count)

    # Đếm
    for label_id, info in LABELS.items():
        label_name = info['name']
        paths = info.get('match_substrings', [])
        for p in paths:
            c = count_images_in_dir(p, recursive=recursive)
            per_label_paths[label_name].append((p, c))
            per_label_total[label_name] += c

    # In ra màn hình (gọn)
    print("\n================= IMAGE COUNTS =================")
    grand_total = 0
    for label_name in sorted(per_label_total.keys()):
        print(f"\nLabel: {label_name}")
        for p, c in per_label_paths[label_name]:
            status = "" if os.path.exists(p) else " (not found)"
            print(f"  - {p}: {c}{status}")
        print(f"  => Total [{label_name}]: {per_label_total[label_name]}")
        grand_total += per_label_total[label_name]
    print("\n-----------------------------------------------")
    print(f"GRAND TOTAL (all labels): {grand_total}")
    print("===============================================\n")

    # Ghi CSV nếu cần
    if save_csv:
        os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
        with open(save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["label_name", "path", "count"])
            for label_name in sorted(per_label_paths.keys()):
                for p, c in per_label_paths[label_name]:
                    writer.writerow([label_name, p, c])
            # thêm tổng theo label
            writer.writerow([])
            writer.writerow(["label_name", "TOTAL_of_label", ""])
            for label_name in sorted(per_label_total.keys()):
                writer.writerow([label_name, per_label_total[label_name], ""])
            writer.writerow(["ALL_LABELS", grand_total, ""])
        print(f"✓ Đã lưu thống kê: {save_csv}")

    return {
        "per_label_paths": per_label_paths,
        "per_label_total": dict(per_label_total),
        "grand_total": grand_total
    }

# ===== Ví dụ dùng =====
if __name__ == "__main__":
    LABELS = {
        0: {"name": "brown_spot", "match_substrings": [
            # "../data_total/brown_spot",
            # "../data/yolo_detected_epoch_140/loki4514_train/bacterial_leaf_blight/crops",
            "../data/yolo_detected_epoch_140/paddy_disease_train/brown_spot/crops",
            "../data/yolo_detected_epoch_140/sikhaok_train/BrownSpot/crops",
            # "../data/yolo_detected_epoch_140/trumanrase_train/bacterial_leaf_blight/crops",
        ]},
        1: {"name": "leaf_blast", "match_substrings": [
            # "../data_total/blast",
            # "../data/yolo_detected_epoch_140/loki4514_train/leaf_blast/crops",
            "../data/yolo_detected_epoch_140/paddy_disease_train/blast/crops",
            # "../data/yolo_detected_epoch_140/sikhaok_train/LeafBlast/crops",
            # "../data/yolo_detected_epoch_140/trumanrase_train/blast/crops",
        ]},
        2: {"name": "leaf_blight", "match_substrings": [
            # "../data_total/bacterial_leaf_blight",
            # "../data/yolo_detected_epoch_140/loki4514_train/bacterial_leaf_blight/crops",
            "../data/yolo_detected_epoch_140/paddy_disease_train/bacterial_leaf_blight/crops",
            "../data/yolo_detected_epoch_140/sikhaok_train/Bacterialblight1/crops",
            "../data/yolo_detected_epoch_140/trumanrase_train/bacterial_leaf_blight/crops",
        ]},
        3: {"name": "healthy", "match_substrings": [
            "../data_total/normal",
            # "../data/yolo_detected_epoch_140/loki4514_train/healthy/crops",
            "../data/yolo_detected_epoch_140/paddy_disease_train/normal/crops",
            "../data/yolo_detected_epoch_140/sikhaok_train/Healthy/crops",
            # "../data/raw/paddy_disease_classification/train_images/normal",
        ]},
    }

    # Đếm và lưu ra CSV (tùy chọn)
    stats = count_dataset(LABELS, recursive=True, save_csv="./outputs/image_counts.csv")
