import os
import json
from glob import glob
import cv2
import numpy as np


INPUT_DIR = "data/fr2_dishes/rgb"
OUTPUT_DIR = "outputs/preprocessed_frames"

# Keep 1 frame every STRIDE frames
STRIDE = 3

# Target output size
OUT_W, OUT_H = 640, 480

# Flag controlling whether preprocessing metadata is persisted to disk
SAVE_META = True

def list_images(input_dir: str):
    exts = ("*.png", "*.jpg", "*.jpeg")
    files = []
    for e in exts:
        files += glob(os.path.join(input_dir, e))
    files = sorted(files)
    return files


def parse_timestamp(path: str):
    """
       Parse a floating-point timestamp from an image filename.
       Assumes filenames follow the TUM RGB-D convention:
           <timestamp>.png  (e.g., 1305031102.175304.png)

       Returns:
           float: Parsed timestamp in seconds if successful.
           None:  If the filename does not encode a valid timestamp.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        return float(name)
    except ValueError:
        return None


def resize_with_letterbox(img_bgr, out_w, out_h):
    """
    Resize while preserving aspect ratio, then pad to (out_h, out_w).
    Returns: resized_padded_img, scale, pad_left, pad_top
    """
    h, w = img_bgr.shape[:2]
    scale = min(out_w / w, out_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    pad_left = (out_w - new_w) // 2
    pad_top = (out_h - new_h) // 2

    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas, scale, pad_left, pad_top


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = list_images(INPUT_DIR)
    if not files:
        raise RuntimeError(f"No images found in: {INPUT_DIR}")

    ts_path = os.path.join(OUTPUT_DIR, "timestamps.txt")
    meta_path = os.path.join(OUTPUT_DIR, "meta.json")

    saved = 0
    with open(ts_path, "w", encoding="utf-8") as fts:
        for i, path in enumerate(files):
            if i % STRIDE != 0:
                continue

            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] Cannot read: {path}")
                continue

            # timestamp
            t = parse_timestamp(path)
            if t is None:
                t = float(saved)

            out_img, scale, pad_left, pad_top = resize_with_letterbox(img, OUT_W, OUT_H)

            out_name = f"frame_{saved:05d}.png"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, out_img)

            fts.write(f"{out_name} {t:.6f}\n")

            if saved == 0:
                orig_h, orig_w = img.shape[:2]
                first_info = (orig_w, orig_h, scale, pad_left, pad_top)

            print(f"[INFO] {saved:05d} <- {os.path.basename(path)}  t={t:.6f}")

            saved += 1

    if SAVE_META:
        orig_w, orig_h, scale, pad_left, pad_top = first_info
        meta = {
            "input_dir": INPUT_DIR,
            "output_dir": OUTPUT_DIR,
            "stride": STRIDE,
            "orig_size": [orig_w, orig_h],
            "out_size": [OUT_W, OUT_H],
            "resize_mode": "letterbox_keep_aspect",
            "first_frame_letterbox": {
                "scale": scale,
                "pad_left": pad_left,
                "pad_top": pad_top
            },
            "notes": "Ensure camera intrinsics are consistent with the resized/letterboxed images for downstream geometry."
        }
        with open(meta_path, "w", encoding="utf-8") as fm:
            json.dump(meta, fm, indent=2)

    print(f"[DONE] Saved {saved} frames to {OUTPUT_DIR}")
    print(f"[DONE] timestamps: {ts_path}")
    if SAVE_META:
        print(f"[DONE] meta: {meta_path}")


if __name__ == "__main__":
    main()
