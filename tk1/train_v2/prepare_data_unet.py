#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, shutil, sys

# ==== SỬA 2 DÒNG NÀY THEO MÁY BẠN ====
IN_PATH  = r"../data/new_data_field_rice_detected/yolo_gpu1_fast_data_trumanrase"   # ví dụ: r"D:\data\unet_gpu1_fast"
OUT_PATH = r"../data/new_data_field_rice_detected/unet_region_only"      # ví dụ: r"D:\data\region_only"
# ======================================


# Tuỳ chọn
FLATTEN   = False   # True = dồn hết ảnh vào OUT_PATH; False = giữ nguyên cấu trúc thư mục con
OVERWRITE = False   # True = ghi đè nếu trùng tên

def main():
    src = os.path.abspath(IN_PATH)
    dst = os.path.abspath(OUT_PATH)

    if not os.path.isdir(src):
        print(f"Không tìm thấy thư mục nguồn: {src}", file=sys.stderr)
        sys.exit(1)

    copied = 0
    for dirpath, _, files in os.walk(src):
        rel_dir = os.path.relpath(dirpath, src)
        for f in files:
            if not f.lower().endswith("_region.png"):
                continue

            src_file = os.path.join(dirpath, f)
            if FLATTEN or rel_dir == ".":
                dst_dir = dst
            else:
                dst_dir = os.path.join(dst, rel_dir)

            os.makedirs(dst_dir, exist_ok=True)
            dst_file = os.path.join(dst_dir, f)

            if not os.path.exists(dst_file) or OVERWRITE:
                shutil.copy2(src_file, dst_file)
                copied += 1

    print(f"Đã copy {copied} ảnh *_region.png vào: {dst}")

if __name__ == "__main__":
    main()
