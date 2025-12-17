import os
import numpy as np
import cv2
from PIL import Image

import torch
from transformers import AutoFeatureExtractor, AutoModelForDepthEstimation


# Directory containing preprocessed RGB frames
INPUT_DIR = "outputs/preprocessed_frames"

# Directories for saving depth outputs
DEPTH_IMG_DIR = "outputs/depth_maps_png"
DEPTH_NPY_DIR = "outputs/depth_maps_npy"

os.makedirs(DEPTH_IMG_DIR, exist_ok=True)
os.makedirs(DEPTH_NPY_DIR, exist_ok=True)

# A monocular depth model
MODEL_NAME = "LiheYoung/depth-anything-small-hf"
# later try:
#   "LiheYoung/depth-anything-base-hf"
#   "LiheYoung/depth-anything-large-hf"

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# process only first N frames for debugging
MAX_FRAMES = None


def load_model_and_extractor(model_name: str, device: str):
    """
    Load the depth estimation model and its corresponding feature extractor.
    """
    print(f"[INFO] Loading depth model: {model_name} on device: {device}")
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return extractor, model


def compute_depth_for_image(image_rgb: Image.Image,
                            extractor: AutoFeatureExtractor,
                            model: AutoModelForDepthEstimation,
                            device: str) -> np.ndarray:
    """
    Given a PIL RGB image, run the depth model and return a 2D numpy array
    of predicted depth values (float32).
    """
    # Preprocess image for the model
    inputs = extractor(images=image_rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # outputs.predicted_depth: (1, 1, H, W)
        depth = outputs.predicted_depth.squeeze(0).squeeze(0)  # (H, W)
        depth_np = depth.cpu().numpy().astype(np.float32)

    return depth_np


def normalize_depth_for_visualization(depth: np.ndarray) -> np.ndarray:
    """
    Normalize a depth map to [0, 255] uint8 for saving as a grayscale image.
    This is for visualization only; the raw depth should be kept separately.
    """
    d = depth.copy()
    # Avoid issues if depth has constant values
    d_min = np.min(d)
    d_max = np.max(d)

    if d_max - d_min < 1e-6:
        # Flat depth, return a mid-gray image
        return np.full_like(d, 128, dtype=np.uint8)

    d = (d - d_min) / (d_max - d_min)  # scale to [0, 1]
    d_uint8 = (d * 255.0).clip(0, 255).astype(np.uint8)
    return d_uint8


# Main pipeline
def main():
    # 1) Load model + extractor
    extractor, model = load_model_and_extractor(MODEL_NAME, DEVICE)

    # 2) Collect frame file names
    frame_files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    if MAX_FRAMES is not None:
        frame_files = frame_files[:MAX_FRAMES]

    print(f"[INFO] Found {len(frame_files)} frame(s) in {INPUT_DIR}")

    # 3) Loop over frames
    for idx, fname in enumerate(frame_files):
        img_path = os.path.join(INPUT_DIR, fname)

        # Load image as RGB (PIL)
        image = Image.open(img_path).convert("RGB")

        # Run depth model
        depth = compute_depth_for_image(image, extractor, model, DEVICE)
        # depth shape: (H, W), float32

        # Save raw depth as .npy (for later point cloud / 3D stages)
        base_name = os.path.splitext(fname)[0]
        npy_path = os.path.join(DEPTH_NPY_DIR, base_name + "_depth.npy")
        np.save(npy_path, depth)

        # Create visualization image (grayscale PNG)
        depth_vis = normalize_depth_for_visualization(depth)
        png_path = os.path.join(DEPTH_IMG_DIR, base_name + "_depth.png")

        # OpenCV expects BGR, but for single-channel we can save directly
        cv2.imwrite(png_path, depth_vis)

        print(f"[INFO] [{idx+1}/{len(frame_files)}] Saved depth: "
              f"npy={npy_path}, png={png_path}")

    print("[DONE] Depth estimation completed for all frames.")


if __name__ == "__main__":
    main()
