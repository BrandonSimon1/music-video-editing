"""
allin1 algorithm — uses All-In-One Music Structure Analyzer.

Produces clips aligned to downbeats with beat density / regularity metrics,
enabling the downstream beat filter in process_video.py.

Cache: analysis/<date>-allin1.json (beats, downbeats, segments, BPM).
       Reused across runs; re-created only when --reanalyze is passed.
"""

import argparse
from datetime import date
from pathlib import Path

ALGORITHM_NAME = "allin1"
ALGORITHM_VERSION = "harmonix-all"


def add_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("allin1 options")
    g.add_argument("--min-duration", type=float, default=30.0,
                   help="Minimum clip duration in seconds (default: 30)")
    g.add_argument("--max-duration", type=float, default=60.0,
                   help="Maximum clip duration in seconds (default: 60)")
    g.add_argument("--reanalyze", action="store_true",
                   help="Re-run allin1 even if a cached analysis file exists")
    g.add_argument("--device", default="cpu",
                   help="Torch device for allin1 inference (default: cpu)")


def run(video_file: Path, cache_dir: Path, args: argparse.Namespace) -> list[dict]:
    # Lazy import so non-allin1 algorithms don't need madmom installed
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from allin1_clip_extractor import run_analysis, build_clips_from_file

    analysis_path = _find_or_create_analysis(
        video_file, cache_dir,
        device=args.device,
        reanalyze=getattr(args, "reanalyze", False),
        run_analysis_fn=run_analysis,
    )

    clips_data = build_clips_from_file(
        str(analysis_path),
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
    return clips_data["clips"]


def get_params(args: argparse.Namespace) -> dict:
    return {
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "device": args.device,
    }


def _find_or_create_analysis(video_file, cache_dir, device, reanalyze, run_analysis_fn):
    today = date.today().isoformat()
    new_path = cache_dir / f"{today}-allin1.json"

    if not reanalyze:
        candidates = sorted(cache_dir.glob("*-allin1.json"), reverse=True)
        if candidates:
            print(f"Using cached analysis: {candidates[0].name}")
            return candidates[0]

    print("Running allin1 (this takes ~6 hours on CPU)...")
    run_analysis_fn(str(video_file), str(new_path), device=device)
    return new_path
