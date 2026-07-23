#!/usr/bin/env python3
"""
Exp 02: LLM reasoning over audio feature text — song boundary detection.

Sends a compact text representation of the gap score time series to Claude
and asks it to identify genuine song boundaries. No threshold tuning needed —
the LLM reasons about which low-energy valleys represent actual song ends.

The gap score (from exp01) is high when both percussive RMS and onset strength
are low. We downsample to 10-second intervals and format as an ASCII bar chart
so Claude can see the structure at a glance.

Approach:
  1. Load features from exp00 JSON
  2. Downsample gap_score to 10-second intervals
  3. Format as compact ASCII table
  4. Send to Claude via `claude -p --input-format stream-json`
  5. Parse JSON response for boundary timestamps
  6. Visualize and save results

Usage:
  uv run experiments/exp02_text_features_llm.py <features_json> [options]

  --model MODEL       Claude model (default: claude-sonnet-4-5)
  --interval INT      Downsample interval in seconds (default: 10)
  -o OUTPUT.png       Save visualization
  --save-json PATH    Save detected segments as JSON
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


BAR_WIDTH = 12  # chars for ASCII bar


def load_features(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {k: np.array(v) if isinstance(v, list) else v for k, v in data.items()}


def normalize(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(x, np.ones(window) / window, mode="same")


def compute_gap_score(features: dict, smooth_window: int = 8) -> np.ndarray:
    perc = smooth(features["perc_rms"], smooth_window)
    onset = smooth(features["onset_strength"], smooth_window)
    return normalize(1 - normalize(perc)) * normalize(1 - normalize(onset))


def build_text_table(times: np.ndarray, gap_score: np.ndarray, interval: int) -> str:
    """
    Format gap_score as a compact ASCII bar chart at `interval`-second resolution.

    Example row:
      15:30 |████████░░░░| 0.67
    """
    step = interval
    rows = []
    i = 0
    while i < len(times):
        t = float(times[i])
        score = float(gap_score[i])
        mm = int(t // 60)
        ss = int(t % 60)
        filled = round(score * BAR_WIDTH)
        bar = "█" * filled + "░" * (BAR_WIDTH - filled)
        rows.append(f"{mm:3d}:{ss:02d} |{bar}| {score:.2f}")
        # Advance by `step` seconds
        target = t + step
        while i < len(times) and float(times[i]) < target:
            i += 1

    return "\n".join(rows)


def build_prompt(table: str, total_minutes: float) -> str:
    return f"""\
You are analyzing a {total_minutes:.0f}-minute music practice session recording.

The table below shows a "gap score" sampled every 10 seconds across the full recording.
- gap_score near 1.0 (full bar ████████████) = very quiet, no drumming or rhythmic activity
- gap_score near 0.0 (empty bar ░░░░░░░░░░░░) = loud, active playing

Format: MM:SS |bar| score

{table}

---

Your task: identify the timestamps where one song ENDED and the next song STARTED.
Look for sustained high-gap-score regions (multiple consecutive rows with high bars)
that separate periods of active playing. Ignore brief pauses within songs.

Think step by step:
1. Identify all sustained high-gap-score regions (lasting >20 seconds)
2. For each, judge whether it represents a genuine between-song gap or just a
   long pause/breakdown within a song (those tend to be shorter and sandwiched
   between active sections)
3. Pick the most likely song boundaries

Respond with ONLY a JSON object in this exact format (no markdown, no explanation):
{{
  "reasoning": "brief summary of your approach",
  "boundaries": [
    {{"time_seconds": 180, "gap_start": 170, "gap_end": 210, "confidence": "high", "reason": "..."}},
    ...
  ],
  "songs": [
    {{"song_num": 1, "start_seconds": 0, "end_seconds": 180, "duration_minutes": 3.0}},
    ...
  ]
}}"""


def call_claude(prompt: str, model: str) -> str:
    """Send prompt to Claude via `claude -p --input-format stream-json`."""
    msg_json = json.dumps({
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        },
    })
    cmd = [
        "claude", "--model", model, "-p",
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--verbose",
    ]
    result = subprocess.run(cmd, input=msg_json, capture_output=True, text=True, timeout=300)

    # Extract text from assistant message event
    for line in result.stdout.splitlines():
        try:
            obj = json.loads(line)
            if obj.get("type") == "assistant":
                for block in obj.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        return block["text"].strip()
        except (json.JSONDecodeError, KeyError):
            continue

    if result.stderr:
        print(f"stderr: {result.stderr[:500]}", file=sys.stderr)
    return ""


def parse_response(text: str) -> dict | None:
    """Extract JSON from Claude's response (handles both raw and code-block JSON)."""
    # Try raw first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from ```json ... ``` block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding a bare {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def plot_results(features: dict, gap_score: np.ndarray, parsed: dict, output: str | None):
    times = features["times"]
    songs = parsed.get("songs", [])
    boundaries = parsed.get("boundaries", [])

    fig, axes = plt.subplots(3, 1, figsize=(22, 9), sharex=True)
    fig.suptitle(f"Exp 02: LLM Song Boundary Detection — {len(songs)} songs", fontsize=12)

    def shade(ax):
        for b in boundaries:
            ax.axvspan(b.get("gap_start", b["time_seconds"]),
                       b.get("gap_end", b["time_seconds"] + 30),
                       color="red", alpha=0.3, zorder=0)
        for s in songs:
            color = plt.cm.tab20(s["song_num"] % 20)
            ax.axvspan(s["start_seconds"], s["end_seconds"], color=color, alpha=0.1, zorder=0)

    # Panel 1: percussive RMS
    perc_sm = smooth(features["perc_rms"], 8)
    axes[0].plot(times, normalize(perc_sm), color="darkorange", linewidth=0.7)
    shade(axes[0])
    axes[0].set_ylabel("Percussive RMS\n(norm)", fontsize=8)

    # Panel 2: gap score
    axes[1].plot(times, gap_score, color="steelblue", linewidth=0.7)
    shade(axes[1])
    axes[1].set_ylabel("Gap Score", fontsize=8)

    # Panel 3: song map
    axes[2].set_ylim(0, 1)
    for s in songs:
        color = plt.cm.tab20(s["song_num"] % 20)
        axes[2].axvspan(s["start_seconds"], s["end_seconds"], color=color, alpha=0.6)
        mid = (s["start_seconds"] + s["end_seconds"]) / 2
        axes[2].text(mid, 0.5,
                     f"Song {s['song_num']}\n{s['duration_minutes']:.1f}m",
                     ha="center", va="center", fontsize=7,
                     transform=axes[2].get_xaxis_transform())
    for b in boundaries:
        axes[2].axvspan(b.get("gap_start", b["time_seconds"]),
                        b.get("gap_end", b["time_seconds"] + 30),
                        color="red", alpha=0.5)
    axes[2].set_ylabel("LLM Song Map", fontsize=8)
    axes[2].set_yticks([])

    axes[-1].set_xlabel("Time (seconds)", fontsize=9)
    for ax in axes:
        ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Exp 02: LLM song boundary detection from text features"
    )
    parser.add_argument("features_json", help="Features JSON from exp00")
    parser.add_argument("--model", default="claude-sonnet-4-5",
                        help="Claude model (default: claude-sonnet-4-5)")
    parser.add_argument("--interval", type=int, default=10,
                        help="Downsample interval in seconds (default: 10)")
    parser.add_argument("--smooth-window", type=int, default=8,
                        help="Smoothing window in seconds (default: 8)")
    parser.add_argument("-o", "--output", help="Output PNG path")
    parser.add_argument("--save-json", help="Save detected segments as JSON")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompt and exit without calling Claude")
    args = parser.parse_args()

    if not Path(args.features_json).exists():
        sys.exit(f"Error: not found: {args.features_json}")

    print(f"Loading features: {args.features_json}")
    features = load_features(args.features_json)
    times = features["times"]
    total_minutes = float(times[-1]) / 60
    print(f"  Duration: {total_minutes:.1f} min, {len(times)} frames")

    print("Computing gap score...")
    gap_score = compute_gap_score(features, smooth_window=args.smooth_window)

    print(f"Building text table ({args.interval}s intervals → {len(times)//args.interval} rows)...")
    table = build_text_table(times, gap_score, args.interval)
    prompt = build_prompt(table, total_minutes)

    print(f"\nPrompt length: {len(prompt)} chars / ~{len(prompt)//4} tokens")

    if args.dry_run:
        print("\n--- PROMPT (first 3000 chars) ---")
        print(prompt[:3000])
        print("...")
        return

    print(f"\nCalling Claude ({args.model})...")
    raw_response = call_claude(prompt, args.model)

    print("\n--- Raw response ---")
    print(raw_response[:2000])

    parsed = parse_response(raw_response)
    if not parsed:
        print("\nWarning: could not parse JSON from response. Saving raw response.")
        if args.save_json:
            with open(args.save_json, "w") as f:
                json.dump({"raw_response": raw_response}, f, indent=2)
        return

    songs = parsed.get("songs", [])
    boundaries = parsed.get("boundaries", [])

    print(f"\n--- {len(boundaries)} boundaries detected ---")
    for b in boundaries:
        t = b["time_seconds"]
        print(f"  {t//60:.0f}:{t%60:02.0f}  confidence={b.get('confidence','?')}  {b.get('reason','')}")

    print(f"\n--- {len(songs)} songs ---")
    for s in songs:
        print(f"  Song {s['song_num']:2d}: {s['start_seconds']//60:.0f}:{s['start_seconds']%60:02.0f}"
              f" – {s['end_seconds']//60:.0f}:{s['end_seconds']%60:02.0f}"
              f"  ({s['duration_minutes']:.1f} min)")

    print(f"\nReasoning: {parsed.get('reasoning', '—')}")

    if args.save_json:
        out = {
            "model": args.model,
            "params": {"interval": args.interval, "smooth_window": args.smooth_window},
            "raw_response": raw_response,
            "parsed": parsed,
        }
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved: {args.save_json}")

    if args.output:
        plot_results(features, gap_score, parsed, args.output)


if __name__ == "__main__":
    main()
