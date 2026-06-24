#!/usr/bin/env python3
"""
Analyze tempo variation in detected beats.
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_tempo_variation(clips_json: str, output_file: str = None):
    """Analyze tempo variation from beat intervals."""

    # Load clips metadata
    print(f"Loading clips metadata from: {clips_json}")
    with open(clips_json, 'r') as f:
        data = json.load(f)

    # Extract all beat times
    all_beats = []
    for clip in data['clips']:
        for measure in clip['measures']:
            for beat in measure:
                all_beats.append(beat)

    all_beats = np.array(all_beats)
    print(f"Total beats: {len(all_beats)}")

    # Compute beat intervals (time between consecutive beats)
    beat_intervals = np.diff(all_beats)

    # Convert to instantaneous tempo (BPM)
    # tempo = 60 / interval
    instantaneous_tempo = 60.0 / beat_intervals

    # Statistics
    print(f"\nTempo Statistics:")
    print(f"  Global tempo estimate: {data['tempo_bpm']:.2f} BPM")
    print(f"  Mean beat interval: {np.mean(beat_intervals):.3f} seconds")
    print(f"  Std beat interval: {np.std(beat_intervals):.3f} seconds")
    print(f"  Min beat interval: {np.min(beat_intervals):.3f} seconds ({60/np.min(beat_intervals):.1f} BPM)")
    print(f"  Max beat interval: {np.max(beat_intervals):.3f} seconds ({60/np.max(beat_intervals):.1f} BPM)")
    print(f"\nInstantaneous Tempo:")
    print(f"  Mean: {np.mean(instantaneous_tempo):.2f} BPM")
    print(f"  Std: {np.std(instantaneous_tempo):.2f} BPM")
    print(f"  Min: {np.min(instantaneous_tempo):.2f} BPM")
    print(f"  Max: {np.max(instantaneous_tempo):.2f} BPM")

    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Panel 1: Beat intervals over time
    ax1 = axes[0]
    beat_times = all_beats[:-1]  # Times for each interval
    ax1.plot(beat_times, beat_intervals, color='blue', alpha=0.5, linewidth=0.5)
    ax1.axhline(np.mean(beat_intervals), color='red', linestyle='--',
                label=f'Mean: {np.mean(beat_intervals):.3f}s')
    ax1.set_ylabel('Beat Interval (seconds)')
    ax1.set_title('Tempo Variation Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Instantaneous tempo over time
    ax2 = axes[1]
    ax2.plot(beat_times, instantaneous_tempo, color='purple', alpha=0.5, linewidth=0.5)
    ax2.axhline(data['tempo_bpm'], color='red', linestyle='--',
                label=f'Global estimate: {data["tempo_bpm"]:.1f} BPM')
    ax2.axhline(np.mean(instantaneous_tempo), color='orange', linestyle=':',
                label=f'Mean: {np.mean(instantaneous_tempo):.1f} BPM')
    ax2.set_ylabel('Instantaneous Tempo (BPM)')
    ax2.set_xlabel('Time (seconds)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Histogram of beat intervals
    ax3 = axes[2]
    ax3.hist(beat_intervals, bins=100, color='green', alpha=0.7)
    ax3.axvline(np.mean(beat_intervals), color='red', linestyle='--',
                label=f'Mean: {np.mean(beat_intervals):.3f}s')
    ax3.set_xlabel('Beat Interval (seconds)')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of Beat Intervals')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization: {output_file}")
    else:
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: analyze_tempo_variation.py <clips_json> [output_file]")
        sys.exit(1)

    clips_json = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'tempo_variation.png'

    if not Path(clips_json).exists():
        print(f"Error: Clips JSON not found: {clips_json}")
        sys.exit(1)

    analyze_tempo_variation(clips_json, output_file)
