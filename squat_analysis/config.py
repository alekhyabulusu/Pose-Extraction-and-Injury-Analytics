"""
Central configuration for the squat analysis pipeline.
"""

from pathlib import Path

# ── Project paths ─────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT_DIR / "data"
RAW_VIDEO_DIR = DATA_DIR / "raw_videos"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR   = ROOT_DIR / "outputs"

# ── MediaPipe model ───────────────────────────────────────────────────────────
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)
MODEL_PATH = Path.home() / ".cache" / "mediapipe" / "pose_landmarker_heavy.task"
MODEL_NAME = "pose_landmarker_heavy"

# ── Stage 1: Extraction ───────────────────────────────────────────────────────
MIN_DETECTION_CONFIDENCE  = 0.7
MIN_TRACKING_CONFIDENCE   = 0.6
MIN_PRESENCE_CONFIDENCE   = 0.7

# Landmarks below this visibility are set to NaN (raw value still stored)
VISIBILITY_NAN_THRESHOLD      = 0.65
# Frames where fewer than this fraction of squat landmarks are visible
# are flagged as low quality
MIN_FRAME_DETECTION_QUALITY   = 0.5

# ── MediaPipe landmark indices ────────────────────────────────────────────────
L_SHOULDER   = 11
R_SHOULDER   = 12
L_HIP        = 23
R_HIP        = 24
L_KNEE       = 25
R_KNEE       = 26
L_ANKLE      = 27
R_ANKLE      = 28
L_HEEL       = 29
R_HEEL       = 30
L_FOOT_INDEX = 31
R_FOOT_INDEX = 32

N_LANDMARKS = 33

# Landmarks that matter for squats — used for detection quality scoring
SQUAT_LANDMARKS = [
    L_SHOULDER, R_SHOULDER,
    L_HIP, R_HIP,
    L_KNEE, R_KNEE,
    L_ANKLE, R_ANKLE,
    L_HEEL, R_HEEL,
    L_FOOT_INDEX, R_FOOT_INDEX,
]

# ── Stage 2: Preprocessing ────────────────────────────────────────────────────
SG_WINDOW              = 7     # Savitzky-Golay window (must be odd)
SG_POLY                = 2     # polynomial order
JOLT_THRESHOLD         = 0.15  # frame-to-frame hip displacement (body units)
TORSO_ALIGNMENT_THRESHOLD = 30.0  # degrees off-vertical before low_confidence flag

# ── Stage 3: Features + Rep segmentation ─────────────────────────────────────
N_FRAMES           = 20    # all reps resampled to this length
REP_PROMINENCE     = 0.08  # min valley depth in normalised hip-Y units
REP_DISTANCE       = 20    # min frames between rep bottoms
MIN_REP_FRAMES     = 15
MAX_REP_FRAMES     = 180
MIN_DEPTH_FRACTION = 0.60  # partial-rep rejection threshold

# ── Stage 4: Mining ───────────────────────────────────────────────────────────
DTW_MAX_K              = 6
PCA_VARIANCE_THRESHOLD = 0.95
ARM_MIN_SUPPORT        = 0.10
ARM_MIN_CONFIDENCE     = 0.60

DISCRETIZATION_THRESHOLDS = {
    "knee_flexion_at_bottom":       (90.0, "lt", "shallow_squat"),
    "trunk_lean_max":               (45.0, "gt", "excessive_lean"),
    "symmetry_knee_at_bottom":      (10.0, "gt", "asymmetric_depth"),
    "ankle_dorsiflexion_at_bottom": (15.0, "lt", "ankle_restricted"),
    "descent_ascent_ratio":         (0.5,  "lt", "rushed_descent"),
    "butt_wink_delta":              (15.0, "gt", "butt_wink"),
}
