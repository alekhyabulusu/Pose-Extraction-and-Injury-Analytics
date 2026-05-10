# Automated Exercise Form Assessment & Injury Risk Prediction

A data mining pipeline that takes raw squat videos as input and outputs injury risk predictions with specific form error feedback. Built with MediaPipe, scikit-learn, and mlxtend.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)

**Automated Exercise Form Assessment** solves the problem of identifying unsafe exercise form in unsupervised workout settings like home gyms, online programs, solo training where no coach is present to catch risky movement patterns, by:
1. Extracting pose landmarks from raw squat video using MediaPipe
2. Engineering 15 biomechanical features from joint angles, timing, and velocity
3. Classifying each rep into risk categories with a Random Forest model (88.7% accuracy)
4. Discovering form error chains via association rule mining to pinpoint root causes

## Features

- **End-to-end video processing**: Raw MP4 in → risk prediction + form errors out
- **Pose extraction**: MediaPipe Heavy model extracts 33 landmarks per frame
- **8-step preprocessing**: NaN interpolation, Y-flip, global scaling, origin shift, jolt detection, Savitzky-Golay smoothing, torso alignment, confidence scoring
- **Rep segmentation**: Valley detection on knee-Y with peak-detection fallback
- **15 engineered features**: 8 static joint angles + 7 temporal dynamics features
- **PCA anomaly detection**: Flags top 10% of reps by reconstruction error (7 components, 95.9% variance)
- **Association rule mining**: FP-Growth discovers co-occurring form error chains
- **Binary + 3-class classification**: Random Forest with balanced class weights

## Pipeline Architecture

```
MP4 squat video
  │
  ├─ Stage 1: Landmark Extraction
  │    MediaPipe Heavy model → 33 landmarks × (x, y, visibility)
  │
  ├─ Stage 2: Preprocessing
  │    NaN interpolation → Y-flip → scale → smooth → jolt detection → confidence
  │
  ├─ Stage 3: Feature Engineering
  │    Rep segmentation → 15 biomechanical features + 6 binary form-error flags
  │
  └─ Stage 4: Mining & Classification
       PCA anomaly detection → FP-Growth association rules → Random Forest classifier
       │
       └─▶ Risk prediction + form error feedback
```

## Dataset

1,525 MP4 squat videos from the Pose Estimation for Powerlifting dataset. Each video is 3–6 seconds (~114–180 frames at 30 fps) containing approximately one squat rep.

## Tech Stack

- **Pose Estimation**: MediaPipe PoseLandmarker (Heavy, float16)
- **Preprocessing**: NumPy, SciPy (Savitzky-Golay filter)
- **Dimensionality Reduction**: scikit-learn (PCA, RobustScaler)
- **Classification**: scikit-learn (Random Forest, Logistic Regression)
- **Association Rules**: mlxtend (FP-Growth)
- **Language**: Python
