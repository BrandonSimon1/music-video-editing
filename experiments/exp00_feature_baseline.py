#!/usr/bin/env python3
"""
Exp 00: Audio Feature Baseline Visualization

Extracts cheap librosa features at 1-second resolution and plots them against
known song/clip boundaries from an allin1 analysis JSON or song-split segments
JSON. Goal: identify which features cleanly discriminate boundaries before
involving any LLM.

Features extracted:
  - RMS energy (overall loudness)
  - Percussive RMS (after harmonic-percussive source separation)
  - Onset strength envelope (rhythmic activity)
  - Spectral centroid (brightness — drops during silence)
  - Spectral flux (rate of change — high during transitions)
  - Zero crossing rate (correlates with noisiness / high-frequency content)

Usage:
  uv run experiments/exp00_feature_baseline.py <audio_or_video> [options]

  --analysis-json PATH   allin1 _analysis.json (for segment/downbeat overlays)
  --segments-json PATH   song-splitting segments JSON (for song boundary overlays)
  -o OUTPUT.png          save plot (default: show interactively)
  --duration SECONDS     limit analysis to first N seconds (default: full file)
  --hop-seconds FLOAT    feature time resolution in seconds (default: 1.0)
"""

import argparse
import json
import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_audio(path: str, duration: float | None = None) -> tuple[np.ndarray, int]:
    print(f"Loading audio: {path}")
    y, sr = librosa.load(path, sr=22050, mono=True, duration=duration)
    print(f"  Duration: {len(y)/sr:.1f}s  SR: {sr}Hz")
    return y, sr


def extract_features(y: np.ndarray, sr: int, hop_seconds: float = 1.0) -> dict:
    hop = int(sr * hop_seconds)
    print(f"Extracting features (hop={hop_seconds}s)...")

    # RMS energy
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]

    # Harmonic-percussive separation
    print("  HPSS...")
    y_harm, y_perc = librosa.effects.hpss(y)
    perc_rms = librosa.feature.rms(y=y_perc, hop_length=hop)[0]

    # Onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    # Spectral centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]

    # Spectral flux (frame-to-frame spectral difference)
    S = np.abs(librosa.stft(y, hop_length=hop))
    flux = np.sqrt(np.mean(np.diff(S, axis=1) ** 2, axis=0))
    flux = np.concatenate([[0], flux])  # prepend 0 to align with other features

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop)[0]

    # Common time axis
    n_frames = min(len(rms), len(perc_rms), len(onset_env),
                   len(spec_centroid), len(flux), len(zcr))
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop)

    print(f"  {n_frames} frames over {times[-1]:.1f}s")
    return {
        "times": times,
        "rms": rms[:n_frames],
        "perc_rms": perc_rms[:n_frames],
        "onset_strength": onset_env[:n_frames],
        "spectral_centroid": spec_centroid[:n_frames],
        "spectral_flux": flux[:n_frames],
        "zcr": zcr[:n_frames],
    }


def normalize(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)


def load_boundaries(analysis_json: str | None, segments_json: str | None) -> dict:
    """Load boundary times from allin1 analysis or song-split segments JSON."""
    result = {
        "song_boundaries": [],   # between-song gaps
        "segment_boundaries": [], # allin1 structural segments (verse/chorus etc.)
        "segment_labels": [],
        "downbeat_times": [],     # allin1 downbeats (sampled — every 4th)
    }

    if analysis_json and Path(analysis_json).exists():
        with open(analysis_json) as f:
            data = json.load(f)
        segs = data.get("segments", [])
        result["segment_boundaries"] = [s["start"] for s in segs]
        result["segment_labels"] = [s["label"] for s in segs]
        # Sample every 16th downbeat to avoid visual clutter
        dbs = data.get("downbeats", [])
        result["downbeat_times"] = dbs[::16]
        print(f"Loaded {len(segs)} segments, {len(dbs)} downbeats from {analysis_json}")

    if segments_json and Path(segments_json).exists():
        with open(segments_json) as f:
            data = json.load(f)
        # Support both {segments: [...]} and {clips: [...]} formats
        items = data.get("segments", data.get("clips", []))
        result["song_boundaries"] = [s["start_time"] for s in items if "start_time" in s]
        print(f"Loaded {len(result['song_boundaries'])} song boundaries from {segments_json}")

    return result


def plot_features(features: dict, boundaries: dict, output: str | None):
    times = features["times"]
    fig, axes = plt.subplots(6, 1, figsize=(20, 14), sharex=True)
    fig.suptitle("Exp 00: Audio Feature Baseline", fontsize=13)

    panels = [
        ("RMS Energy",         features["rms"],               "steelblue"),
        ("Percussive RMS",     features["perc_rms"],          "darkorange"),
        ("Onset Strength",     features["onset_strength"],    "green"),
        ("Spectral Centroid",  features["spectral_centroid"], "purple"),
        ("Spectral Flux",      features["spectral_flux"],     "firebrick"),
        ("Zero Crossing Rate", features["zcr"],               "teal"),
    ]

    def draw_boundaries(ax):
        for t in boundaries["song_boundaries"]:
            ax.axvline(t, color="red", linewidth=1.5, alpha=0.8, linestyle="-")
        for t in boundaries["segment_boundaries"]:
            ax.axvline(t, color="gold", linewidth=0.8, alpha=0.6, linestyle="--")
        for t in boundaries["downbeat_times"]:
            ax.axvline(t, color="lightblue", linewidth=0.4, alpha=0.4, linestyle=":")

    for ax, (label, data, color) in zip(axes, panels):
        ax.plot(times, normalize(data), color=color, linewidth=0.7, alpha=0.9)
        draw_boundaries(ax)
        ax.set_ylabel(label, fontsize=8)
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Time (seconds)", fontsize=9)

    # Legend
    legend_handles = [
        mpatches.Patch(color="red",       label="Song boundary"),
        mpatches.Patch(color="gold",      label="allin1 segment"),
        mpatches.Patch(color="lightblue", label="Downbeat (1 of 16)"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()

    plt.close(fig)


def save_features_json(features: dict, output_path: str):
    """Save features as JSON for use in later experiments."""
    data = {k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in features.items()}
    with open(output_path, "w") as f:
        json.dump(data, f)
    print(f"Saved features JSON: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Exp 00: Audio feature baseline visualization"
    )
    parser.add_argument("audio_file", help="Audio or video file to analyze")
    parser.add_argument("--analysis-json", help="allin1 _analysis.json for overlay")
    parser.add_argument("--segments-json", help="Song-split segments JSON for overlay")
    parser.add_argument("-o", "--output", help="Output PNG path (default: show interactively)")
    parser.add_argument("--save-json", help="Also save extracted features as JSON")
    parser.add_argument("--duration", type=float, help="Limit to first N seconds")
    parser.add_argument("--hop-seconds", type=float, default=1.0,
                        help="Feature time resolution in seconds (default: 1.0)")
    args = parser.parse_args()

    if not Path(args.audio_file).exists():
        sys.exit(f"Error: file not found: {args.audio_file}")

    y, sr = load_audio(args.audio_file, duration=args.duration)
    features = extract_features(y, sr, hop_seconds=args.hop_seconds)
    boundaries = load_boundaries(args.analysis_json, args.segments_json)

    if args.save_json:
        save_features_json(features, args.save_json)

    plot_features(features, boundaries, args.output)

    # Print summary stats at boundary vs non-boundary frames
    print("\n--- Feature summary at segment boundaries ---")
    segs = boundaries["segment_boundaries"]
    if segs:
        times = features["times"]
        for feat_name in ("rms", "perc_rms", "onset_strength"):
            vals = features[feat_name]
            norm_vals = normalize(vals)
            boundary_idxs = [np.argmin(np.abs(times - t)) for t in segs]
            non_boundary_idxs = [i for i in range(len(times)) if i not in set(boundary_idxs)]
            at_boundary = np.mean(norm_vals[boundary_idxs])
            not_boundary = np.mean(norm_vals[non_boundary_idxs])
            print(f"  {feat_name:20s}: at boundary={at_boundary:.3f}  not at boundary={not_boundary:.3f}  "
                  f"ratio={at_boundary/not_boundary:.2f}")


if __name__ == "__main__":
    main()
