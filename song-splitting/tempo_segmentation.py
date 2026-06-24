#!/usr/bin/env python3
"""
Concert Audio Segmentation via Percussive Energy & Onset Rate Analysis

Automatically segments 1-2 hour concert recordings into individual songs
by detecting drops in percussive energy and onset rate during transitions.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, medfilt


# =============================================================================
# Stage 1: Feature Extraction
# =============================================================================

def extract_onset_strength(audio_file: str) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Extract onset strength envelope from audio file.

    Args:
        audio_file: Path to audio file

    Returns:
        Tuple of (onset_env, sample_rate, y_percussive)
    """
    print(f"Loading audio: {audio_file}")
    y, sr = librosa.load(audio_file, sr=None)  # Load at native sample rate

    print(f"Sample rate: {sr} Hz, Duration: {len(y)/sr:.1f} seconds")

    # Separate percussive elements to get cleaner rhythm signal
    print("Separating percussive elements...")
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    print("Computing onset strength envelope from percussive component...")
    onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr, aggregate=np.median)

    return onset_env, sr, y_percussive


# =============================================================================
# Stage 2: Feature Analysis (Percussive Energy & Onset Rate)
# =============================================================================

def compute_features(
    onset_env: np.ndarray,
    y_percussive: np.ndarray,
    sr: float,
    window_size: float = 10.0,
    hop_size: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute percussive energy and onset rate over time.

    Args:
        onset_env: Onset strength envelope
        y_percussive: Percussive component of audio
        sr: Sample rate
        window_size: Window size in seconds
        hop_size: Hop size in seconds

    Returns:
        Tuple of (percussive_energy, onset_rate, time_points)
    """
    print(f"Computing percussive energy and onset rate (window={window_size}s, hop={hop_size}s)...")

    hop_length = 512
    window_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)

    percussive_energy = []
    onset_rate = []
    time_points = []

    # Detect onsets for onset rate calculation
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=False
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # Slide window across audio
    num_windows = int((len(y_percussive) - window_samples) / hop_samples) + 1

    for i in range(num_windows):
        start_sample = i * hop_samples
        end_sample = start_sample + window_samples

        if end_sample > len(y_percussive):
            break

        # 1. Percussive RMS energy in this window
        window_audio = y_percussive[start_sample:end_sample]
        rms_energy = np.sqrt(np.mean(window_audio**2))
        percussive_energy.append(rms_energy)

        # 2. Onset rate (onsets per second) in this window
        window_start_time = start_sample / sr
        window_end_time = end_sample / sr

        # Count onsets in this time window
        onsets_in_window = np.sum(
            (onset_times >= window_start_time) & (onset_times < window_end_time)
        )
        rate = onsets_in_window / window_size  # onsets per second
        onset_rate.append(rate)

        # Time point is center of window
        time_points.append((window_start_time + window_end_time) / 2)

    print(f"Generated {len(percussive_energy)} feature windows")

    return (
        np.array(percussive_energy),
        np.array(onset_rate),
        np.array(time_points)
    )


# =============================================================================
# Stage 3: Gap Detection Score
# =============================================================================

def compute_gap_score(
    percussive_energy: np.ndarray,
    onset_rate: np.ndarray,
    time_points: np.ndarray,
    smoothing_window: int = 3
) -> np.ndarray:
    """
    Compute gap detection score from percussive energy and onset rate.
    High score = likely gap between songs.

    Args:
        percussive_energy: RMS energy of percussive component
        onset_rate: Onset rate (onsets per second)
        time_points: Time points for features
        smoothing_window: Window size for smoothing (samples)

    Returns:
        Gap score time series (high = likely gap)
    """
    print("Computing gap detection score...")

    # Smooth features to reduce noise
    if smoothing_window > 1:
        print(f"Smoothing features with median filter (window={smoothing_window})...")
        percussive_energy = medfilt(percussive_energy, kernel_size=smoothing_window)
        onset_rate = medfilt(onset_rate, kernel_size=smoothing_window)

    # Invert and normalize: low energy/onset rate = high gap score
    energy_gap_score = 1 - normalize(percussive_energy)
    onset_gap_score = 1 - normalize(onset_rate)

    # Combined gap score (equal weights)
    gap_score = 0.5 * energy_gap_score + 0.5 * onset_gap_score

    print(f"Gap score range: [{np.min(gap_score):.3f}, {np.max(gap_score):.3f}]")

    return gap_score


def normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)


# =============================================================================
# Stage 4: Boundary Detection
# =============================================================================

def detect_boundaries(
    gap_score: np.ndarray,
    time_points: np.ndarray,
    threshold: float = 0.65,
    min_gap_duration: float = 10.0,
    min_song_duration: float = 300.0,
    total_duration: float = None
) -> List[float]:
    """
    Detect song boundaries from gap scores.

    Args:
        gap_score: Gap score time series (high = likely gap)
        time_points: Time points for gap scores
        threshold: Gap score threshold for detection
        min_gap_duration: Minimum gap duration in seconds
        min_song_duration: Minimum song duration in seconds
        total_duration: Total recording duration

    Returns:
        List of boundary timestamps in seconds
    """
    print(f"Detecting boundaries (threshold={threshold}, min_gap={min_gap_duration}s)...")

    # Use peak detection approach
    time_step = time_points[1] - time_points[0]
    min_gap_samples = int(min_gap_duration / time_step)
    min_song_samples = int(min_song_duration / time_step)

    peaks, properties = find_peaks(
        gap_score,
        height=threshold,              # Minimum gap score
        distance=min_song_samples,     # Minimum song length
        width=min_gap_samples          # Minimum gap width
    )

    # Convert peak indices to time
    boundaries = [time_points[p] for p in peaks]

    # Add start and end
    boundaries = [0.0] + sorted(boundaries)
    if total_duration is not None:
        boundaries.append(total_duration)
    elif len(time_points) > 0:
        boundaries.append(time_points[-1])

    print(f"Detected {len(boundaries) - 1} segments")

    return boundaries


# =============================================================================
# Stage 5: Segment Characterization
# =============================================================================

def characterize_segments(
    boundaries: List[float],
    percussive_energy: np.ndarray,
    onset_rate: np.ndarray,
    time_points: np.ndarray,
    gap_score: np.ndarray
) -> List[Dict]:
    """
    Compute metadata for each detected segment.

    Args:
        boundaries: List of boundary timestamps
        percussive_energy: Percussive energy over time
        onset_rate: Onset rate over time
        time_points: Time points for features
        gap_score: Gap scores

    Returns:
        List of segment metadata dictionaries
    """
    print("Characterizing segments...")

    segments = []

    for i in range(len(boundaries) - 1):
        start_time = boundaries[i]
        end_time = boundaries[i + 1]

        # Find feature values in this segment
        mask = (time_points >= start_time) & (time_points < end_time)
        segment_energy = percussive_energy[mask]
        segment_onset_rate = onset_rate[mask]

        if len(segment_energy) == 0:
            # Handle edge case
            energy_mean = 0.0
            onset_rate_mean = 0.0
        else:
            energy_mean = float(np.mean(segment_energy))
            onset_rate_mean = float(np.mean(segment_onset_rate))

        # Analyze boundary
        next_boundary_info = None
        if i < len(boundaries) - 2:  # Not the last segment
            boundary_time = end_time
            # Find gap score at boundary
            boundary_idx = np.argmin(np.abs(time_points - boundary_time))
            window = 5  # Look at +/- 5 samples
            start_idx = max(0, boundary_idx - window)
            end_idx = min(len(gap_score), boundary_idx + window)

            max_gap_score = float(np.max(gap_score[start_idx:end_idx]))

            # Next segment features
            next_mask = (time_points >= end_time) & (time_points < boundaries[i + 2])
            next_energy = percussive_energy[next_mask]
            next_onset = onset_rate[next_mask]
            next_energy_mean = float(np.mean(next_energy)) if len(next_energy) > 0 else None
            next_onset_mean = float(np.mean(next_onset)) if len(next_onset) > 0 else None

            next_boundary_info = {
                "time": boundary_time,
                "gap_score": max_gap_score,
                "energy_from": energy_mean,
                "energy_to": next_energy_mean,
                "onset_rate_from": onset_rate_mean,
                "onset_rate_to": next_onset_mean
            }

        segment = {
            "segment_id": i + 1,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "percussive_energy_mean": energy_mean,
            "onset_rate_mean": onset_rate_mean,
            "next_boundary": next_boundary_info
        }

        segments.append(segment)

    return segments


# =============================================================================
# Visualization
# =============================================================================

def visualize_analysis(
    onset_env: np.ndarray,
    percussive_energy: np.ndarray,
    onset_rate: np.ndarray,
    time_points: np.ndarray,
    gap_score: np.ndarray,
    boundaries: List[float],
    sr: float,
    output_file: Optional[str] = None
):
    """
    Create visualization of the analysis.

    Args:
        onset_env: Onset strength envelope
        percussive_energy: Percussive energy over time
        onset_rate: Onset rate over time
        time_points: Time points for features
        gap_score: Gap scores
        boundaries: Detected boundaries
        sr: Sample rate
        output_file: Optional path to save figure
    """
    print("Creating visualization...")

    # Create time axis for onset envelope
    hop_length = 512
    onset_times = librosa.frames_to_time(
        np.arange(len(onset_env)),
        sr=sr,
        hop_length=hop_length
    )

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Panel 1: Onset strength
    ax1 = axes[0]
    ax1.plot(onset_times, onset_env, color='gray', alpha=0.7)
    ax1.set_ylabel('Onset Strength')
    ax1.set_title('Concert Audio Segmentation via Percussive Energy & Onset Rate Analysis')
    ax1.grid(True, alpha=0.3)

    # Mark boundaries
    for b in boundaries[1:-1]:  # Skip start and end
        ax1.axvline(b, color='red', linestyle='--', alpha=0.6)

    # Panel 2: Percussive energy
    ax2 = axes[1]
    ax2.plot(time_points, percussive_energy, color='purple', linewidth=2, label='Percussive Energy')
    ax2.set_ylabel('Percussive Energy (RMS)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mark boundaries
    for b in boundaries[1:-1]:
        ax2.axvline(b, color='red', linestyle='--', alpha=0.6)

    # Panel 3: Onset rate
    ax3 = axes[2]
    ax3.plot(time_points, onset_rate, color='blue', linewidth=2, label='Onset Rate')
    ax3.set_ylabel('Onset Rate (onsets/sec)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Mark boundaries
    for b in boundaries[1:-1]:
        ax3.axvline(b, color='red', linestyle='--', alpha=0.6)

    # Panel 4: Gap score
    ax4 = axes[3]
    ax4.plot(time_points, gap_score, color='orange', linewidth=2)
    ax4.axhline(0.65, color='green', linestyle=':', label='Threshold')
    ax4.set_ylabel('Gap Score')
    ax4.set_xlabel('Time (seconds)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Mark boundaries
    for b in boundaries[1:-1]:
        ax4.axvline(b, color='red', linestyle='--', alpha=0.6, label='Boundary' if b == boundaries[1] else '')

    if len(boundaries) > 2:
        ax4.legend()

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {output_file}")
    else:
        plt.show()


# =============================================================================
# Main Pipeline
# =============================================================================

def segment_concert(
    audio_file: str,
    output_json: Optional[str] = None,
    output_viz: Optional[str] = None,
    window_size: float = 10.0,
    hop_size: float = 2.0,
    threshold: float = 0.65,
    min_gap_duration: float = 10.0,
    min_song_duration: float = 300.0,
    stability_window: float = 5.0
) -> Dict:
    """
    Main pipeline for concert segmentation.

    Args:
        audio_file: Path to audio file
        output_json: Optional path to save JSON output
        output_viz: Optional path to save visualization
        window_size: Tempo estimation window size (seconds)
        hop_size: Tempo estimation hop size (seconds)
        threshold: Instability threshold for boundary detection
        min_gap_duration: Minimum gap duration (seconds)
        min_song_duration: Minimum song duration (seconds)
        stability_window: Window for stability computation (seconds)

    Returns:
        Dictionary with segments and metadata
    """
    # Stage 1: Feature Extraction
    onset_env, sr, y_percussive = extract_onset_strength(audio_file)

    # Calculate total duration
    hop_length = 512
    total_duration = librosa.frames_to_time(len(onset_env), sr=sr, hop_length=hop_length)

    # Stage 2: Feature Analysis
    percussive_energy, onset_rate, time_points = compute_features(
        onset_env, y_percussive, sr, window_size, hop_size
    )

    # Stage 3: Gap Detection Score
    gap_score = compute_gap_score(
        percussive_energy, onset_rate, time_points, smoothing_window=5
    )

    # Stage 4: Boundary Detection
    boundaries = detect_boundaries(
        gap_score, time_points, threshold,
        min_gap_duration, min_song_duration, total_duration
    )

    # Stage 5: Segment Characterization
    segments = characterize_segments(
        boundaries, percussive_energy, onset_rate, time_points, gap_score
    )

    # Create output
    result = {
        "audio_file": audio_file,
        "total_duration": float(total_duration),
        "num_segments": len(segments),
        "parameters": {
            "window_size": window_size,
            "hop_size": hop_size,
            "threshold": threshold,
            "min_gap_duration": min_gap_duration,
            "min_song_duration": min_song_duration,
            "stability_window": stability_window
        },
        "segments": segments
    }

    # Save JSON output
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved segments: {output_json}")

    # Create visualization
    if output_viz:
        visualize_analysis(
            onset_env, percussive_energy, onset_rate, time_points,
            gap_score, boundaries, sr, output_viz
        )

    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Segment concert recordings into individual songs using tempo stability analysis'
    )

    parser.add_argument(
        'audio_file',
        help='Path to concert audio file'
    )

    parser.add_argument(
        '-o', '--output',
        help='Output JSON file for segments (default: <audio_file>_segments.json)'
    )

    parser.add_argument(
        '-v', '--visualize',
        help='Output visualization file (e.g., output.png)'
    )

    parser.add_argument(
        '--window-size',
        type=float,
        default=10.0,
        help='Tempo estimation window size in seconds (default: 10.0)'
    )

    parser.add_argument(
        '--hop-size',
        type=float,
        default=2.0,
        help='Tempo estimation hop size in seconds (default: 2.0)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.65,
        help='Instability threshold for boundary detection (default: 0.65)'
    )

    parser.add_argument(
        '--min-gap',
        type=float,
        default=10.0,
        help='Minimum gap duration in seconds (default: 10.0)'
    )

    parser.add_argument(
        '--min-song',
        type=float,
        default=300.0,
        help='Minimum song duration in seconds (default: 300.0)'
    )

    parser.add_argument(
        '--stability-window',
        type=float,
        default=5.0,
        help='Window for stability computation in seconds (default: 5.0)'
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.audio_file).exists():
        print(f"Error: Audio file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    # Set default output file
    output_json = args.output
    if output_json is None:
        output_json = str(Path(args.audio_file).with_suffix('')) + '_segments.json'

    # Run segmentation
    try:
        result = segment_concert(
            audio_file=args.audio_file,
            output_json=output_json,
            output_viz=args.visualize,
            window_size=args.window_size,
            hop_size=args.hop_size,
            threshold=args.threshold,
            min_gap_duration=args.min_gap,
            min_song_duration=args.min_song,
            stability_window=args.stability_window
        )

        # Print summary
        print("\n" + "=" * 60)
        print(f"Segmentation complete!")
        print(f"Total duration: {result['total_duration']:.1f} seconds")
        print(f"Detected segments: {result['num_segments']}")
        print("\nSegments:")
        for seg in result['segments']:
            print(f"  Segment {seg['segment_id']}: "
                  f"{seg['start_time']:.1f}s - {seg['end_time']:.1f}s "
                  f"({seg['duration']:.1f}s, "
                  f"energy={seg['percussive_energy_mean']:.4f}, "
                  f"onset_rate={seg['onset_rate_mean']:.2f}/s)")
        print("=" * 60)

    except Exception as e:
        print(f"Error during segmentation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()