#!/usr/bin/env python3
"""
One-off: reconstruct downbeats from the original allin1 clips JSON and
rebuild clips using the new 30-60s duration logic.

The original analysis didn't save a raw downbeats list. We recover them by
treating each clip's start_time as a downbeat (they were built by stepping
through downbeats 4 at a time), and pulling intermediate downbeats from each
clip's beats array at indices 0, 4, 8, 12, ... (every 4th beat = each measure
start in 4/4).

Usage:
    uv run python clip-splitting/rebuild_clips.py \
        clip-splitting/2025-10-30-mcs-practice_allin1_clips.json \
        [--min-duration 30] [--max-duration 60] [-o output_clips.json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from allin1_clip_extractor import build_clips, build_clips_from_file


def reconstruct_analysis(clips_json_path: str) -> dict:
    with open(clips_json_path) as f:
        data = json.load(f)

    # Reconstruct downbeats: each clip start_time is a downbeat; within each clip
    # beats[0::4] are the measure-start downbeats (assuming 4/4).
    downbeat_set = set()
    for clip in data["clips"]:
        beats = clip.get("beats", [])
        # Every 4th beat starting from 0 is a downbeat (measure 1, 2, 3, 4...)
        for i in range(0, len(beats), 4):
            downbeat_set.add(round(beats[i], 6))
        # Clip start_time is always a downbeat (belt-and-suspenders)
        downbeat_set.add(round(clip["start_time"], 6))

    downbeats = sorted(downbeat_set)

    # Collect all beats across all clips
    all_beats = set()
    for clip in data["clips"]:
        for b in clip.get("beats", []):
            all_beats.add(round(b, 6))
    beats = sorted(all_beats)

    # Segments are already stored as dicts in the original JSON
    segments = data.get("segments", [])

    print(f"Reconstructed {len(downbeats)} downbeats, {len(beats)} beats, "
          f"{len(segments)} segments from {len(data['clips'])} original clips")

    return {
        "audio_file": data["audio_file"],
        "total_duration": data.get("total_duration"),
        "bpm": data.get("bpm"),
        "total_beats": len(beats),
        "total_downbeats": len(downbeats),
        "total_segments": len(segments),
        "beats": beats,
        "downbeats": downbeats,
        "segments": segments,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips_json", help="Path to original allin1 clips JSON")
    parser.add_argument("-o", "--output", help="Output clips JSON (default: <input>_rebuilt.json)")
    parser.add_argument("--min-duration", type=float, default=30.0)
    parser.add_argument("--max-duration", type=float, default=60.0)
    args = parser.parse_args()

    if not Path(args.clips_json).exists():
        print(f"Error: file not found: {args.clips_json}", file=sys.stderr)
        sys.exit(1)

    output_json = args.output or (
        str(Path(args.clips_json).with_suffix("")) + "_rebuilt.json"
    )

    analysis = reconstruct_analysis(args.clips_json)
    clips = build_clips(analysis, args.min_duration, args.max_duration)

    result = {
        **{k: v for k, v in analysis.items() if k not in ("beats", "downbeats")},
        "num_clips": len(clips),
        "clip_parameters": {
            "min_duration": args.min_duration,
            "max_duration": args.max_duration,
            "source": "reconstructed from original clips JSON",
        },
        "clips": clips,
    }

    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved rebuilt clips: {output_json}")


if __name__ == "__main__":
    main()
