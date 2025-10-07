# compare_rice_disease_paths.py
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ==== ĐIỀN ĐƯỜNG DẪN TẠI ĐÂY ====
GT_PATH   = r"example_result.csv"   # ví dụ: r"D:\data\gt.csv"
PRED_PATH = r"/home/bbsw/thong/deep_learning/tk1/train_v2/trí_submission.csv"    # ví dụ: r"D:\data\pred.csv"
OUT_DIR   = r"/home/bbsw/thong/deep_learning/tk1/output/compare_results"              # ví dụ: r"D:\data\out"
# =================================

LABELS = {
    0: {'name': 'brown_spot'},
    1: {'name': 'blast'},
    2: {'name': 'bacterial_leaf_blight'},
    3: {'name': 'normal'},
}
ALLOWED = {v['name'] for v in LABELS.values()}
ORDERED_CLASSES = ['brown_spot', 'blast', 'bacterial_leaf_blight', 'normal']

def normalize_label(x):
    if pd.isna(x):
        return x
    return str(x).strip().lower()

def map_numeric_to_name_if_needed(series):
    """
    Nếu nhãn là số (0-3) -> map theo LABELS.
    Nếu là tên -> giữ nguyên.
    Giá trị lạ -> trả về như cũ (sẽ bị filter sau nếu không thuộc ALLOWED).
    """
    try:
        as_int = series.astype('Int64')  # pandas nullable int
        unique_vals = set([v for v in as_int.dropna().unique().tolist()])
        if unique_vals.issubset(set(LABELS.keys())) and len(unique_vals) > 0:
            return as_int.map(lambda v: LABELS[int(v)]['name'] if pd.notna(v) else np.nan)
    except Exception:
        pass
    return series

def ensure_out_dir(path):
    os.makedirs(path, exist_ok=True)
    return path.rstrip("/\\")  # bỏ dấu / hoặc \ cuối

def main():
    out_dir = ensure_out_dir(OUT_DIR)

    # Đọc CSV
    gt = pd.read_csv(GT_PATH)
    pred = pd.read_csv(PRED_PATH)

    # Chuẩn hoá cột tối thiểu
    for df in (gt, pred):
        if 'image_id' not in df.columns or 'label' not in df.columns:
            raise ValueError("Mỗi file CSV phải có cột 'image_id' và 'label'.")
        df['image_id'] = df['image_id'].astype(str).str.strip()
        df['label'] = df['label'].apply(normalize_label)

    # Map số -> tên nếu cần
    pred['label'] = map_numeric_to_name_if_needed(pred['label'])
    gt['label']   = map_numeric_to_name_if_needed(gt['label'])

    # Chỉ giữ 4 lớp
    gt = gt[gt['label'].isin(ALLOWED)].copy()
    pred = pred[pred['label'].isin(ALLOWED)].copy()

    # Thống kê ảnh chỉ có 1 phía
    gt_ids = set(gt['image_id'])
    pred_ids = set(pred['image_id'])
    only_in_gt = sorted(list(gt_ids - pred_ids))
    only_in_pred = sorted(list(pred_ids - gt_ids))

    # Ghép theo image_id
    merged = pd.merge(
        gt.rename(columns={'label': 'true_label'}),
        pred.rename(columns={'label': 'pred_label'}),
        on='image_id',
        how='inner'
    )

    if merged.empty:
        raise ValueError("Không có ảnh trùng nhau giữa 2 file sau khi lọc 4 lớp.")

    y_true = merged['true_label'].astype(str)
    y_pred = merged['pred_label'].astype(str)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=ORDERED_CLASSES)
    report_dict = classification_report(y_true, y_pred, labels=ORDERED_CLASSES, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()

    # Xuất file
    merged[['image_id', 'true_label', 'pred_label']].to_csv(f"{out_dir}/comparison_aligned.csv", index=False)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in ORDERED_CLASSES], columns=[f"pred_{c}" for c in ORDERED_CLASSES])
    cm_df.to_csv(f"{out_dir}/confusion_matrix.csv")
    report_df.to_csv(f"{out_dir}/classification_report.csv")

    # In tóm tắt
    print("==== TÓM TẮT ĐÁNH GIÁ (Chỉ 4 lớp) ====")
    print(f"Ground-truth (4 lớp): {len(gt)} hàng")
    print(f"Dự đoán (4 lớp):      {len(pred)} hàng")
    print(f"Số ảnh khớp so sánh:  {len(merged)}")
    print(f"Accuracy:             {acc:.4f}\n")
    print("Ma trận nhầm lẫn theo thứ tự lớp:", ORDERED_CLASSES)
    print(cm_df, "\n")
    print("Báo cáo chi tiết (precision/recall/f1):")
    print(report_df, "\n")

    if only_in_gt:
        print(f"Ảnh chỉ có ở ground-truth: {len(only_in_gt)} (ví dụ: {only_in_gt[:10]})")
    if only_in_pred:
        print(f"Ảnh chỉ có ở dự đoán:      {len(only_in_pred)} (ví dụ: {only_in_pred[:10]})")

if __name__ == "__main__":
    main()
