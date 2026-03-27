"""
Stage 1 — Raw landmark extraction
Reads an mp4 video frame by frame, runs MediaPipe PoseLandmarker,
and saves:
    landmarks.npy          (T, 33, 3)  x, y, visibility per landmark
    timestamps.npy         (T,)        milliseconds per frame
    detection_quality.npy  (T,)        fraction of squat landmarks visible
    metadata.json          session info, fps, model version, resolution
"""

import json
import urllib.request
import warnings
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from squat_analysis.config import (
    MODEL_URL,
    MODEL_PATH,
    MODEL_NAME,
    PROCESSED_DIR,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MIN_PRESENCE_CONFIDENCE,
    VISIBILITY_NAN_THRESHOLD,
    MIN_FRAME_DETECTION_QUALITY,
    N_LANDMARKS,
    SQUAT_LANDMARKS,
)


# ── Model management ──────────────────────────────────────────────────────────

def _ensure_model(model_path: Optional[Path] = None) -> str:
    """Return path to the .task model file, downloading if needed.

    Args:
        model_path: Optional override. Uses MODEL_PATH cache if None.

    Returns:
        Absolute path string to the model file.
    """
    target = Path(model_path) if model_path else MODEL_PATH

    if target.exists() and target.stat().st_size > 1_000_000:
        return str(target)

    print(f"  Downloading MediaPipe pose model (~30MB) → {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, target)
        print(f"  Downloaded ({target.stat().st_size / 1e6:.1f} MB)")
    except Exception as exc:
        raise RuntimeError(
            f"Model download failed: {exc}\n"
            f"Download manually from:\n  {MODEL_URL}\n"
            f"Save to: {target}"
        ) from exc

    return str(target)


# ── Core extraction ───────────────────────────────────────────────────────────

def extract(
    video_path: str,
    session_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    max_frames:  Optional[int] = None,
) -> Path:
    """Run Stage 1 on a single video file.

    Extracts (x, y, visibility) per landmark per frame. Drops Z.
    Saves output files to data/processed/{session_id}/.

    Args:
        video_path:  Path to input .mp4 (or .mov / .avi).
        session_id:  Identifier for this recording. Defaults to video stem.
        output_dir:  Override output directory. Defaults to PROCESSED_DIR.
        model_path:  Optional path to pose_landmarker_heavy.task.
        max_frames:  Process only first N frames (useful for quick tests).

    Returns:
        Path to the session output directory.

    Output files:
        {output_dir}/{session_id}/landmarks.npy          (T, 33, 3)
        {output_dir}/{session_id}/timestamps.npy         (T,)
        {output_dir}/{session_id}/detection_quality.npy  (T,)
        {output_dir}/{session_id}/metadata.json
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if session_id is None:
        session_id = video_path.stem

    out_dir = Path(output_dir) / session_id if output_dir else PROCESSED_DIR / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    model_file = _ensure_model(Path(model_path) if model_path else None)

    print(f"\n[Stage 1] Extracting landmarks")
    print(f"  video      : {video_path}")
    print(f"  session    : {session_id}")
    print(f"  output     : {out_dir}")

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    fps           = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Configure MediaPipe PoseLandmarker (VIDEO mode) ───────────────────────
    # VIDEO mode gives frame-consistent tracking without callback complexity.
    base_opts = mp_tasks.BaseOptions(model_asset_path=model_file)
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        output_segmentation_masks=False,
    )

    # ── Collect per-frame data ────────────────────────────────────────────────
    all_landmarks:        list[np.ndarray] = []
    all_timestamps_ms:    list[float]      = []
    all_detection_quality:list[float]      = []

    frames_read      = 0
    frames_detected  = 0

    with mp_vision.PoseLandmarker.create_from_options(pose_opts) as detector:
        while cap.isOpened():
            ret, bgr = cap.read()
            if not ret:
                break

            # Exact timestamp from OpenCV (handles variable frame rate)
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # MediaPipe requires monotonically increasing timestamps
            ts_int = max(int(timestamp_ms), frames_read)
            result = detector.detect_for_video(mp_img, ts_int)

            if result.pose_world_landmarks and result.pose_landmarks:
                frames_detected += 1

                world_lms = result.pose_world_landmarks[0]   # 3D world coords
                image_lms = result.pose_landmarks[0]          # image coords (for visibility)

                # x, y from world landmarks (metric, camera-independent)
                # visibility from image landmarks (more reliable signal)
                coords = np.array(
                    [[lm.x, lm.y] for lm in world_lms],
                    dtype=np.float32,
                )                                              # (33, 2)
                vis = np.array(
                    [lm.visibility for lm in image_lms],
                    dtype=np.float32,
                )                                              # (33,)

                # Build (33, 3): x, y, visibility
                frame_data = np.column_stack([coords, vis])   # (33, 3)

                # NaN out low-visibility landmarks (but visibility value kept)
                low_vis = vis < VISIBILITY_NAN_THRESHOLD
                frame_data[low_vis, :2] = np.nan              # x,y → NaN; vis stays

                # Detection quality: fraction of key squat landmarks visible
                squat_vis = vis[SQUAT_LANDMARKS]
                quality   = float(np.mean(squat_vis >= VISIBILITY_NAN_THRESHOLD))

            else:
                # No detection this frame — full NaN row
                frame_data = np.full((N_LANDMARKS, 3), np.nan, dtype=np.float32)
                quality    = 0.0

            all_landmarks.append(frame_data)
            all_timestamps_ms.append(float(timestamp_ms))
            all_detection_quality.append(quality)

            frames_read += 1
            if max_frames and frames_read >= max_frames:
                break

    cap.release()

    if frames_read == 0:
        raise RuntimeError(f"No frames could be read from {video_path}")

    # ── Stack into arrays ─────────────────────────────────────────────────────
    landmarks         = np.stack(all_landmarks, axis=0)          # (T, 33, 3)
    timestamps_ms     = np.array(all_timestamps_ms,    dtype=np.float64)  # (T,)
    detection_quality = np.array(all_detection_quality, dtype=np.float32) # (T,)

    detection_rate = frames_detected / frames_read
    low_quality_frames = int(np.sum(detection_quality < MIN_FRAME_DETECTION_QUALITY))

    if detection_rate < 0.5:
        warnings.warn(
            f"Low detection rate: {detection_rate:.0%} of frames. "
            "Check that the full body (hips to feet) is visible."
        )

    # ── Save outputs ──────────────────────────────────────────────────────────
    np.save(out_dir / "landmarks.npy",         landmarks)
    np.save(out_dir / "timestamps.npy",        timestamps_ms)
    np.save(out_dir / "detection_quality.npy", detection_quality)

    metadata = {
        "session_id":         session_id,
        "source_video":       str(video_path),
        "fps":                fps,
        "total_frames":       frames_read,
        "width":              width,
        "height":             height,
        "detection_rate":     round(detection_rate, 4),
        "low_quality_frames": low_quality_frames,
        "mediapipe_version":  mp.__version__,
        "model_name":         MODEL_NAME,
        "camera_view":        "unknown",
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"  frames read    : {frames_read}")
    print(f"  frames detected: {frames_detected} ({detection_rate:.0%})")
    print(f"  low quality    : {low_quality_frames} frames")
    print(f"  output shape   : {landmarks.shape}  (T, 33, 3)")
    print(f"  saved to       : {out_dir}")

    return out_dir


# ── Loader helper (used by Stage 2) ──────────────────────────────────────────

def load_extraction(session_dir: str) -> dict:
    """Load Stage 1 outputs back into memory.

    Args:
        session_dir: Path to a session directory produced by extract().

    Returns:
        dict with keys:
            landmarks         (T, 33, 3) float32
            timestamps_ms     (T,)       float64
            detection_quality (T,)       float32
            metadata          dict
    """
    d = Path(session_dir)
    with open(d / "metadata.json") as f:
        metadata = json.load(f)

    return {
        "landmarks":         np.load(d / "landmarks.npy"),
        "timestamps_ms":     np.load(d / "timestamps.npy"),
        "detection_quality": np.load(d / "detection_quality.npy"),
        "metadata":          metadata,
    }
