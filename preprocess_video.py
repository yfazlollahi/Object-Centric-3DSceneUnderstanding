import cv2
import numpy as np
import os

# Path to the input video recorded by the phone
VIDEO_PATH = "data/room_walk.mp4"

# Directory where preprocessed frames will be saved
OUTPUT_DIR = "outputs/preprocessed_frames"

# Keep 1 frame every SAMPLE_STRIDE frames
SAMPLE_STRIDE = 3

# Target frame size (width, height) after resizing
TARGET_WIDTH = 640
TARGET_HEIGHT = 384  # 384 = 32 * 12


USE_UNDISTORT = False

# Replace with real calibration results if undistortion / metric geometry is needed
CAMERA_MATRIX = np.array([
    [700.0,   0.0, 640.0],   # fx,  0, cx
    [  0.0, 700.0, 360.0],   # 0,  fy, cy
    [  0.0,   0.0,   1.0]
], dtype=np.float32)

DIST_COEFFS = np.array([0.1, -0.05, 0.0, 0.0, 0.0], dtype=np.float32)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    frame_idx = 0   # index of the current frame in the original video
    saved_idx = 0   # index of the saved preprocessed frame

    while True:
        # Read next frame from the video
        ret, frame_bgr = cap.read()
        if not ret:
            # End of video (no more frames)
            break

        # Frame is in BGR format: shape (H, W, 3)

        # Frame sampling
        if frame_idx % SAMPLE_STRIDE != 0:
            frame_idx += 1
            continue


        if USE_UNDISTORT:
            # Remove lens distortion using camera intrinsics
            frame_bgr = cv2.undistort(frame_bgr, CAMERA_MATRIX, DIST_COEFFS)

        # Resize frame
        frame_bgr_resized = cv2.resize(
            frame_bgr,
            (TARGET_WIDTH, TARGET_HEIGHT),
            interpolation=cv2.INTER_LINEAR
        )

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr_resized, cv2.COLOR_BGR2RGB)


        # Normalize to [0, 1] for later deep models
        # frame_rgb_float = frame_rgb.astype(np.float32) / 255.0

        # Save preprocessed frame as PNG image
        save_path = os.path.join(OUTPUT_DIR, f"frame_{saved_idx:05d}.png")

        # cv2.imwrite expects BGR order, so convert back RGB to BGR
        output_bgr_for_save = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, output_bgr_for_save)

        print(f"[INFO] Saved frame {saved_idx} (orig idx {frame_idx}) → {save_path}")

        # Update counters
        saved_idx += 1
        frame_idx += 1

    # Release video capture
    cap.release()
    print("[DONE] Total saved frames:", saved_idx)


if __name__ == "__main__":
    main()
