#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, shutil, sys
from pathlib import Path

# # ==== SỬA 2 DÒNG NÀY THEO MÁY BẠN ====
# IN_PATH  = r"../data/new_data_field_rice_detected/yolo_gpu1_fast"
# OUT_PATH = r"../data/new_data_field_rice_detected/yolo_no_overlay"
# # ======================================

IN_PATH  = r"./data/yolo_gpu1_fast_data_trumanrase"
OUT_PATH = r"./data/yolo_gpu1_fast_data_trumanrase/yolo_no_overlay"

# Tuỳ chọn
FLATTEN   = False    # PHẢI để False để giữ cấu trúc class
OVERWRITE = False    # True = ghi đè nếu file đã tồn tại ở đích

# Thư mục cần giữ NGUYÊN TRẠNG (copy full, không lọc)
EXCLUDE_DIR_NAMES = {"Healthy"}

# Bộ lọc cho phần còn lại (ngoài EXCLUDE_DIR_NAMES)
# Ở đây: chỉ bỏ ảnh *_overlay.png, còn lại copy PNG
def allowed_in_normal_dirs(filename: str) -> bool:
    n = filename.lower()
    return n.endswith(".png") and (not n.endswith("_overlay.png"))

def copy_file(src_file: str, dst_file: str, overwrite: bool) -> bool:
    if not os.path.exists(dst_file) or overwrite:
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        shutil.copy2(src_file, dst_file)
        return True
    return False

def mirror_copy_dir(src_dir: str, dst_dir: str, overwrite: bool) -> int:
    """Copy nguyên trạng toàn bộ src_dir -> dst_dir (không lọc). Trả về số file đã copy."""
    copied = 0
    for dirpath, _, files in os.walk(src_dir):
        rel = os.path.relpath(dirpath, src_dir)
        out_dir = dst_dir if rel == "." else os.path.join(dst_dir, rel)
        for f in files:
            src_f = os.path.join(dirpath, f)
            dst_f = os.path.join(out_dir, f)
            if copy_file(src_f, dst_f, overwrite):
                copied += 1
    return copied

def main():
    src_root = os.path.abspath(IN_PATH)
    dst_root = os.path.abspath(OUT_PATH)

    if not os.path.isdir(src_root):
        print(f"Không tìm thấy thư mục nguồn: {src_root}", file=sys.stderr)
        sys.exit(1)

    if FLATTEN and EXCLUDE_DIR_NAMES:
        print("Lỗi: FLATTEN=True không hỗ trợ khi bạn cần giữ nguyên các thư mục trong EXCLUDE_DIR_NAMES.",
              file=sys.stderr)
        sys.exit(1)

    copied = 0

    # 1) Pass 1: Duyệt toàn bộ cây nhưng PRUNE các thư mục trong EXCLUDE_DIR_NAMES
    #    -> áp dụng bộ lọc (bỏ *_overlay.png) cho phần còn lại
    for dirpath, dirnames, files in os.walk(src_root):
        # prune: không đi sâu vào các thư mục cần giữ nguyên trạng
        dirnames[:] = [d for d in dirnames if d.lower() not in EXCLUDE_DIR_NAMES]

        rel_dir = os.path.relpath(dirpath, src_root)
        out_dir = dst_root if rel_dir == "." else os.path.join(dst_root, rel_dir)

        for f in files:
            if allowed_in_normal_dirs(f):
                src_file = os.path.join(dirpath, f)
                dst_file = os.path.join(out_dir, f)
                if copy_file(src_file, dst_file, OVERWRITE):
                    copied += 1

    # 2) Pass 2: Riêng các thư mục trong EXCLUDE_DIR_NAMES -> copy NGUYÊN TRẠNG
    #    (có thể xuất hiện ở nhiều tầng)
    for dirpath, dirnames, _ in os.walk(src_root):
        for d in dirnames:
            if d.lower() in EXCLUDE_DIR_NAMES:
                src_dir = os.path.join(dirpath, d)
                rel_dir = os.path.relpath(src_dir, src_root)
                dst_dir = os.path.join(dst_root, rel_dir)
                copied += mirror_copy_dir(src_dir, dst_dir, OVERWRITE)

    print(f"Đã copy {copied} file vào: {dst_root}")
    print(f"- Giữ nguyên các thư mục: {sorted(EXCLUDE_DIR_NAMES)}")
    print("- Phần còn lại áp dụng lọc: bỏ *_overlay.png")

if __name__ == "__main__":
    main()
