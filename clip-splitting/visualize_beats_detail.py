#!/usr/bin/env python3
"""
Create a detailed visualization of beat detection for the first minute.
This allows us to visually validate beat alignment.
"""

import json
import sys
import librosa
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def visualize_beat_detail(audio_file: str, clips_json: str, duration: float = 60.0, output_file: str = None):
    """
    Create detailed visualization of beat detection for the first N seconds.

    Args:
        audio_file: Path to audio file
        clips_json: Path to clips JSON file
        duration: Duration to visualize in seconds
        output_file: Output file path
    """
    # Load clips metadata
    print(f"Loading clips metadata from: {clips_json}")
    with open(clips_json, 'r') as f:
        data = json.load(f)

    # Load audio
    print(f"Loading audio: {audio_file}")
    y, sr = librosa.load(audio_file, sr=None, duration=duration)

    print(f"Sample rate: {sr} Hz, Loaded duration: {len(y)/sr:.1f} seconds")

    # Extract all beat times from clips that fall within our duration
    all_beats = []
    clip_boundaries = []

    for clip in data['clips']:
        if clip['start_time'] > duration:
            break

        clip_boundaries.append(clip['start_time'])

        # Add all beats from all measures in this clip
        for measure in clip['measures']:
            for beat in measure:
                if beat <= duration:
                    all_beats.append(beat)

    all_beats = np.array(all_beats)
    clip_boundaries = np.array(clip_boundaries)

    print(f"Visualizing {len(all_beats)} beats and {len(clip_boundaries)} clip boundaries")

    # Compute features for visualization
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)

    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))

    # Panel 1: Waveform with beats
    ax1 = axes[0]
    times = np.arange(len(y)) / sr
    ax1.plot(times, y, color='gray', alpha=0.5, linewidth=0.5)
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Beat Detection Detail - First {duration:.0f} seconds (Tempo: {data["tempo_bpm"]:.1f} BPM)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, duration)

    # Mark beats
    for beat in all_beats:
        ax1.axvline(beat, color='blue', alpha=0.5, linewidth=1)

    # Mark clip boundaries
    for boundary in clip_boundaries:
        ax1.axvline(boundary, color='red', linestyle='--', alpha=0.9, linewidth=2)

    # Panel 2: Onset strength with beats
    ax2 = axes[1]
    ax2.plot(onset_times, onset_env, color='purple', linewidth=1.5)
    ax2.set_ylabel('Onset Strength')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, duration)

    # Mark beats
    for beat in all_beats:
        ax2.axvline(beat, color='blue', alpha=0.5, linewidth=1,
                   label='Beat' if beat == all_beats[0] else '')

    # Mark clip boundaries
    for i, boundary in enumerate(clip_boundaries):
        label = 'Clip Boundary (4 measures)' if i == 0 else ''
        ax2.axvline(boundary, color='red', linestyle='--', alpha=0.9, linewidth=2, label=label)

    ax2.legend(loc='upper right')

    # Panel 3: RMS energy with beats and measure groupings
    ax3 = axes[2]
    ax3.plot(rms_times, rms, color='orange', linewidth=1.5)
    ax3.set_ylabel('RMS Energy')
    ax3.set_xlabel('Time (seconds)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, duration)

    # Mark beats
    for beat in all_beats:
        ax3.axvline(beat, color='blue', alpha=0.5, linewidth=1)

    # Mark clip boundaries
    for boundary in clip_boundaries:
        ax3.axvline(boundary, color='red', linestyle='--', alpha=0.9, linewidth=2)

    # Add beat numbers to show grouping into measures
    # Show every 4th beat number (measure boundaries)
    for i in range(0, len(all_beats), 4):
        if i < len(all_beats) and all_beats[i] < duration:
            measure_num = i // 4 + 1
            ax3.text(all_beats[i], ax3.get_ylim()[1] * 0.95, f'M{measure_num}',
                    fontsize=7, ha='left', va='top', color='blue', alpha=0.7)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {output_file}")
    else:
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: visualize_beats_detail.py <audio_file> <clips_json> [duration] [output_file]")
        sys.exit(1)

    audio_file = sys.argv[1]
    clips_json = sys.argv[2]
    duration = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
    output_file = sys.argv[4] if len(sys.argv) > 4 else 'beat_detail_visualization.png'

    if not Path(audio_file).exists():
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)

    if not Path(clips_json).exists():
        print(f"Error: Clips JSON not found: {clips_json}")
        sys.exit(1)

    visualize_beat_detail(audio_file, clips_json, duration, output_file)
