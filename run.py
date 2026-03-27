"""
Single CLI entry point. Runs all pipeline stages in sequence.
Usage:
    python run.py --video data/raw_videos/session_01.mp4
    python run.py --video data/raw_videos/session_01.mp4 --session my_session
    python run.py --video data/raw_videos/session_01.mp4 --max-frames 300
"""

import argparse
from pathlib import Path

from squat_analysis.extraction import extract


def main():
    parser = argparse.ArgumentParser(
        description="Squat analysis pipeline: video → features → mining"
    )
    parser.add_argument("--video",      required=True, help="Path to input .mp4")
    parser.add_argument("--session",    default=None,  help="Session ID (default: video stem)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Process only first N frames (for testing)")
    args = parser.parse_args()

    # Stage 1
    session_dir = extract(
        video_path=args.video,
        session_id=args.session,
        max_frames=args.max_frames,
    )

    # Stage 2, 3, 4 — coming soon
    print("\nStages 2–4 not yet implemented. Stage 1 output at:", session_dir)


if __name__ == "__main__":
    main()
