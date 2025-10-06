import os
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
import cv2
import gc
import json
import torch
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from ultralytics import YOLO

# ===================== CONFIG =====================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

DATA_ROOT    = "/home/bbsw/thong/deep_learning/tk1/data/raw/trumanrase/rice_disease_val_test/val"
CKPT_PATH    = "/home/bbsw/thong/deep_learning/tk1/output/yolov8n_leaf_disease_2025-10-06_01-23-16/best.pt"
OUTPUT_ROOT  = "/home/bbsw/thong/deep_learning/tk1/data/yolo_detected/trumanrase_val"

# CONF_THRESH  = 0.4
# IOU_THRESH   = 0.5
# IMG_SIZE     = 640
BATCH_SIZE   = 16
MAX_DET      = 300
MAX_WORKERS  = 10    # số tiến trình song song tối đa


CONF_THRESH  = 0.15
IOU_THRESH   = 0.35
IMG_SIZE     = 640
BATCH_SIZE   = 16
MAX_DET      = 300
OVERLAY_EVERY = 1
ERROR_LOG    = "errors.jsonl"

# ===================== HELPERS =====================
def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def sanitize_text(text: str) -> str:
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\r', '\t']
    for b in bad:
        text = text.replace(b, '_')
    return text.strip() or "unk"

def chunk_list(items, n):
    return [items[i:i + n] for i in range(0, len(items), n)]

def save_image(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.ascontiguousarray(img))

# ===================== WORKER =====================
def process_disease_dir(
    disease_dir: Path, 
    model_path: str, 
    output_root: str, 
    conf: float, 
    iou: float, 
    img_size: int, 
    batch_size: int, 
    max_det: int, 
    use_cuda: bool, 
    device_id: int
) -> dict:
    """Xử lý 1 thư mục bệnh độc lập"""
    from ultralytics import YOLO
    import cv2, torch, gc, json, numpy as np
    from pathlib import Path

    model = YOLO(model_path)
    model.to(device_id if use_cuda else "cpu")
    names = model.names

    err_file = Path(output_root) / f"errors_{disease_dir.name}.jsonl"
    imgs = [p for p in disease_dir.iterdir() if p.is_file() and is_image_file(p)]
    out_dir = Path(output_root) / disease_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    done = 0

    for bi, batch_paths in enumerate(chunk_list(imgs, batch_size), 1):
        try:
            results = model.predict(
                source=[str(p) for p in batch_paths],
                conf=conf,
                iou=iou,
                imgsz=img_size,
                device=device_id if use_cuda else "cpu",
                max_det=max_det,
                verbose=False
            )
        except Exception as e:
            with open(err_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"stage": "batch_predict", "class": disease_dir.name, "error": str(e)}, ensure_ascii=False) + "\n")
            continue

        for p, res in zip(batch_paths, results):
            try:
                img = cv2.imread(str(p))
                if img is None:
                    raise ValueError("Unreadable image")
                det = res.boxes
                boxes = det.data.cpu().numpy() if det is not None and len(det) else np.empty((0, 6), dtype=float)
                overlay = img.copy()

                for *xyxy, conf_, cls in boxes:
                    label = names.get(int(cls), f"id{int(cls)}")
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(overlay, f"{label} {conf_:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                save_image(out_dir / f"{p.stem}_overlay.jpg", overlay)

                summary.append({
                    "image": str(p),
                    "detections": len(boxes)
                })
            except Exception as e2:
                with open(err_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "stage": "postprocess",
                        "class": disease_dir.name,
                        "image": str(p),
                        "error": str(e2)
                    }, ensure_ascii=False) + "\n")

        done += len(batch_paths)
        print(f"[{disease_dir.name}] ✅ {done}/{len(imgs)} images done")

        del results
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()

    return {"class": disease_dir.name, "processed": len(imgs), "errors": str(err_file)}

# ===================== MAIN =====================
def main():
    out_root = Path(OUTPUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 Using CUDA: {gpu_name}")
    else:
        print("🚀 Using CPU")

    disease_dirs = [p for p in Path(DATA_ROOT).iterdir() if p.is_dir()]
    print(f"🧩 Found {len(disease_dirs)} classes: {[d.name for d in disease_dirs]}")

    if not disease_dirs:
        print("❌ No subdirectories found in DATA_ROOT.")
        return

    summary = []
    futures = []
    n_workers = min(MAX_WORKERS, len(disease_dirs))

    print(f"⚡ Starting {n_workers} parallel workers...\n")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for i, ddir in enumerate(disease_dirs):
            gpu_id = i % torch.cuda.device_count() if use_cuda else 0
            fut = executor.submit(
                process_disease_dir,
                ddir, CKPT_PATH, OUTPUT_ROOT,
                CONF_THRESH, IOU_THRESH, IMG_SIZE, BATCH_SIZE, MAX_DET,
                use_cuda, gpu_id
            )
            futures.append(fut)

        for fut in as_completed(futures):
            try:
                res = fut.result()
                summary.append(res)
                print(f"✅ Done {res['class']} ({res['processed']} images)")
            except Exception as e:
                print(f"❌ Worker crashed: {e}")

    summary_path = Path(OUTPUT_ROOT) / "summary_all.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n🎯 ALL DONE!")
    print(f"📁 Results: {OUTPUT_ROOT}")
    print(f"📄 Summary: {summary_path}")

# ===================== ENTRY =====================
if __name__ == "__main__":
    main()
