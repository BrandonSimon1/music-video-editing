#!/usr/bin/env python3
"""
Exp 01: Silence/Gap Detection Baseline — automatic song boundary detection.

Loads pre-computed features from exp00 (percussive RMS + onset strength) and
finds inter-song gaps using a simple gap score threshold. No LLM involved.
Goal: establish what rule-based detection alone can achieve.

Algorithm:
  1. Smooth percussive RMS and onset strength with a rolling window
  2. Compute gap_score = (1 - norm_perc_rms) * (1 - norm_onset_strength)
     → high when both energy and rhythmic activity are low
  3. Threshold gap_score to find "silent" frames
  4. Keep only contiguous silent regions >= min_gap_duration seconds
  5. Merge nearby gaps (< merge_distance apart) into one boundary event
  6. Song segments = intervals between detected gaps

Usage:
  uv run experiments/exp01_gap_detection.py <features_json> [options]

  --threshold FLOAT       Gap score threshold (0-1, default: 0.4)
  --smooth-window INT     Smoothing window in seconds (default: 5)
  --min-gap FLOAT         Minimum gap duration in seconds (default: 5.0)
  --merge-distance FLOAT  Merge gaps closer than N seconds (default: 30.0)
  --min-song FLOAT        Discard segments shorter than N seconds (default: 60.0)
  -o OUTPUT.png           Save visualization
  --save-json PATH        Save detected segments as JSON
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_features(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    # Convert lists back to numpy arrays
    return {k: np.array(v) if isinstance(v, list) else v for k, v in data.items()}


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Simple uniform rolling mean."""
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def normalize(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)


def detect_gaps(
    features: dict,
    threshold: float = 0.4,
    smooth_window: int = 5,
    min_gap: float = 5.0,
    merge_distance: float = 30.0,
    min_song: float = 60.0,
) -> tuple[list[dict], list[dict], np.ndarray]:
    """
    Returns (gaps, songs, gap_score_array).
    gaps: list of {start, end, duration}
    songs: list of {start, end, duration, song_num}
    """
    times = features["times"]
    dt = float(times[1] - times[0]) if len(times) > 1 else 1.0

    perc_rms = smooth(features["perc_rms"], smooth_window)
    onset = smooth(features["onset_strength"], smooth_window)

    gap_score = normalize(1 - normalize(perc_rms)) * normalize(1 - normalize(onset))

    # Find silent frames
    silent = gap_score > threshold

    # Find contiguous silent regions
    raw_gaps = []
    in_gap = False
    gap_start = 0
    for i, s in enumerate(silent):
        if s and not in_gap:
            in_gap = True
            gap_start = i
        elif not s and in_gap:
            in_gap = False
            dur = (i - gap_start) * dt
            if dur >= min_gap:
                raw_gaps.append({
                    "start": float(times[gap_start]),
                    "end": float(times[i - 1]),
                    "duration": dur,
                })
    if in_gap:
        dur = (len(times) - gap_start) * dt
        if dur >= min_gap:
            raw_gaps.append({
                "start": float(times[gap_start]),
                "end": float(times[-1]),
                "duration": dur,
            })

    # Merge nearby gaps
    merged = []
    for gap in raw_gaps:
        if merged and (gap["start"] - merged[-1]["end"]) < merge_distance:
            merged[-1]["end"] = gap["end"]
            merged[-1]["duration"] = merged[-1]["end"] - merged[-1]["start"]
        else:
            merged.append(dict(gap))

    # Build song segments from the spaces between gaps
    boundaries = [0.0] + [g["start"] + g["duration"] / 2 for g in merged] + [float(times[-1])]
    songs = []
    song_num = 1
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        dur = end - start
        if dur >= min_song:
            songs.append({
                "song_num": song_num,
                "start_time": start,
                "end_time": end,
                "duration": dur,
            })
            song_num += 1

    return merged, songs, gap_score


def plot_results(
    features: dict,
    gaps: list[dict],
    songs: list[dict],
    gap_score: np.ndarray,
    threshold: float,
    output: str | None,
):
    times = features["times"]
    fig, axes = plt.subplots(4, 1, figsize=(22, 10), sharex=True)
    fig.suptitle(f"Exp 01: Gap Detection  (threshold={threshold}, {len(songs)} songs detected)", fontsize=12)

    def shade_gaps(ax):
        for gap in gaps:
            ax.axvspan(gap["start"], gap["end"], color="red", alpha=0.25, zorder=0)
        for song in songs:
            ax.axvspan(song["start_time"], song["end_time"],
                       color=plt.cm.tab20(song["song_num"] % 20), alpha=0.08, zorder=0)

    # Panel 1: Percussive RMS (smoothed)
    perc_sm = np.convolve(features["perc_rms"], np.ones(5) / 5, mode="same")
    axes[0].plot(times, normalize(perc_sm), color="darkorange", linewidth=0.7)
    shade_gaps(axes[0])
    axes[0].set_ylabel("Percussive RMS\n(smoothed, norm)", fontsize=8)

    # Panel 2: Onset strength (smoothed)
    onset_sm = np.convolve(features["onset_strength"], np.ones(5) / 5, mode="same")
    axes[1].plot(times, normalize(onset_sm), color="green", linewidth=0.7)
    shade_gaps(axes[1])
    axes[1].set_ylabel("Onset Strength\n(smoothed, norm)", fontsize=8)

    # Panel 3: Gap score with threshold line
    axes[2].plot(times, gap_score, color="steelblue", linewidth=0.7, alpha=0.9)
    axes[2].axhline(threshold, color="red", linewidth=1.0, linestyle="--", label=f"threshold={threshold}")
    axes[2].fill_between(times, gap_score, threshold,
                         where=gap_score > threshold, color="red", alpha=0.3)
    shade_gaps(axes[2])
    axes[2].set_ylabel("Gap Score", fontsize=8)
    axes[2].legend(fontsize=7, loc="upper right")

    # Panel 4: Detected songs as labeled spans
    axes[3].set_ylim(0, 1)
    for song in songs:
        color = plt.cm.tab20(song["song_num"] % 20)
        axes[3].axvspan(song["start_time"], song["end_time"], color=color, alpha=0.5)
        mid = (song["start_time"] + song["end_time"]) / 2
        dur_min = song["duration"] / 60
        axes[3].text(mid, 0.5, f"Song {song['song_num']}\n{dur_min:.1f}m",
                     ha="center", va="center", fontsize=7,
                     transform=axes[3].get_xaxis_transform())
    for gap in gaps:
        axes[3].axvspan(gap["start"], gap["end"], color="red", alpha=0.4)
    axes[3].set_ylabel("Detected Songs", fontsize=8)
    axes[3].set_yticks([])

    axes[-1].set_xlabel("Time (seconds)", fontsize=9)
    for ax in axes:
        ax.tick_params(labelsize=7)

    legend_handles = [
        mpatches.Patch(color="red", alpha=0.4, label=f"Detected gap ({len(gaps)})"),
        mpatches.Patch(color="steelblue", alpha=0.4, label=f"Detected songs ({len(songs)})"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Exp 01: Silence/gap detection for song boundary finding"
    )
    parser.add_argument("features_json", help="Features JSON from exp00")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Gap score threshold 0-1 (default: 0.4)")
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="Smoothing window in seconds (default: 5)")
    parser.add_argument("--min-gap", type=float, default=5.0,
                        help="Minimum gap duration in seconds (default: 5.0)")
    parser.add_argument("--merge-distance", type=float, default=30.0,
                        help="Merge gaps closer than N seconds (default: 30.0)")
    parser.add_argument("--min-song", type=float, default=60.0,
                        help="Discard segments shorter than N seconds (default: 60.0)")
    parser.add_argument("-o", "--output", help="Output PNG path")
    parser.add_argument("--save-json", help="Save detected segments as JSON")
    args = parser.parse_args()

    if not Path(args.features_json).exists():
        sys.exit(f"Error: not found: {args.features_json}")

    print(f"Loading features: {args.features_json}")
    features = load_features(args.features_json)
    total_duration = float(features["times"][-1])
    print(f"  Total duration: {total_duration/60:.1f} minutes, {len(features['times'])} frames")

    print(f"\nDetecting gaps (threshold={args.threshold}, smooth={args.smooth_window}s, "
          f"min_gap={args.min_gap}s, merge={args.merge_distance}s)...")
    gaps, songs, gap_score = detect_gaps(
        features,
        threshold=args.threshold,
        smooth_window=args.smooth_window,
        min_gap=args.min_gap,
        merge_distance=args.merge_distance,
        min_song=args.min_song,
    )

    print(f"\n--- Detected {len(gaps)} gap(s) ---")
    for g in gaps:
        t = g["start"]
        print(f"  {t/60:5.1f}m  ({t:.0f}s)  duration={g['duration']:.1f}s")

    print(f"\n--- Detected {len(songs)} song(s) ---")
    for s in songs:
        print(f"  Song {s['song_num']:2d}: {s['start_time']/60:5.1f}m – {s['end_time']/60:5.1f}m  "
              f"({s['duration']/60:.1f} min)")

    if args.save_json:
        out = {
            "params": {
                "threshold": args.threshold,
                "smooth_window": args.smooth_window,
                "min_gap": args.min_gap,
                "merge_distance": args.merge_distance,
                "min_song": args.min_song,
            },
            "gaps": gaps,
            "songs": songs,
        }
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved segments: {args.save_json}")

    plot_results(features, gaps, songs, gap_score, args.threshold, args.output)


if __name__ == "__main__":
    main()
