#!/usr/bin/env python3
"""
Beat-Aligned Clip Extraction

Splits music videos into clips aligned with musical beats and measures.
Each clip spans a fixed number of measures (default: 4 measures in 4/4 time = 16 beats).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import librosa
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Stage 1: Beat Detection
# =============================================================================

def extract_beats(audio_file: str) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """
    Extract beat times from audio file.

    Args:
        audio_file: Path to audio file

    Returns:
        Tuple of (beat_times, tempo, sample_rate, y_audio)
    """
    print(f"Loading audio: {audio_file}")
    y, sr = librosa.load(audio_file, sr=None)

    print(f"Sample rate: {sr} Hz, Duration: {len(y)/sr:.1f} seconds")

    # Detect beats and estimate tempo
    print("Detecting beats and estimating tempo...")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')

    # Convert tempo to scalar if it's an array
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo.item()) if tempo.size == 1 else float(np.mean(tempo))
    else:
        tempo = float(tempo)

    # Convert beat frames to times
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    print(f"Estimated tempo: {tempo:.1f} BPM")
    print(f"Detected {len(beat_times)} beats")
    if len(beat_times) > 1:
        print(f"Average beat interval: {np.mean(np.diff(beat_times)):.3f} seconds")
    else:
        print("Warning: Only detected one or zero beats")

    return beat_times, tempo, sr, y


# =============================================================================
# Stage 2: Measure Grouping
# =============================================================================

def group_beats_into_measures(
    beat_times: np.ndarray,
    beats_per_measure: int = 4
) -> List[List[float]]:
    """
    Group beats into measures.

    Args:
        beat_times: Array of beat timestamps in seconds
        beats_per_measure: Number of beats per measure (default: 4 for 4/4 time)

    Returns:
        List of measures, where each measure is a list of beat times
    """
    print(f"Grouping beats into measures ({beats_per_measure} beats per measure)...")

    measures = []
    for i in range(0, len(beat_times), beats_per_measure):
        measure_beats = beat_times[i:i + beats_per_measure]
        if len(measure_beats) == beats_per_measure:
            measures.append(measure_beats.tolist())

    print(f"Created {len(measures)} complete measures")

    return measures


# =============================================================================
# Stage 3: Clip Generation
# =============================================================================

def create_clips(
    measures: List[List[float]],
    measures_per_clip: int = 4,
    total_duration: float = None
) -> List[Dict]:
    """
    Create clip boundaries from measures.

    Args:
        measures: List of measures (each measure is a list of beat times)
        measures_per_clip: Number of measures per clip
        total_duration: Total audio duration in seconds

    Returns:
        List of clip metadata dictionaries
    """
    print(f"Creating clips ({measures_per_clip} measures per clip)...")

    clips = []
    clip_id = 1

    for i in range(0, len(measures), measures_per_clip):
        clip_measures = measures[i:i + measures_per_clip]

        if len(clip_measures) < measures_per_clip:
            print(f"Skipping incomplete final clip with only {len(clip_measures)} measures")
            break

        # Start time is the first beat of the first measure
        start_time = clip_measures[0][0]

        # End time is the first beat of the next measure (if it exists)
        # Otherwise, use the last beat + average beat interval
        if i + measures_per_clip < len(measures):
            end_time = measures[i + measures_per_clip][0]
        else:
            # Use total duration if available, otherwise estimate
            if total_duration:
                end_time = total_duration
            else:
                # Estimate end time based on average beat interval
                all_beats = [beat for measure in clip_measures for beat in measure]
                avg_interval = np.mean(np.diff(all_beats))
                end_time = clip_measures[-1][-1] + avg_interval

        duration = end_time - start_time

        # Count total beats in this clip
        total_beats = len([beat for measure in clip_measures for beat in measure])

        clip = {
            "clip_id": clip_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "num_measures": len(clip_measures),
            "num_beats": total_beats,
            "measures": clip_measures
        }

        clips.append(clip)
        clip_id += 1

    print(f"Created {len(clips)} clips")

    return clips


# =============================================================================
# Visualization
# =============================================================================

def visualize_analysis(
    y: np.ndarray,
    sr: float,
    beat_times: np.ndarray,
    clips: List[Dict],
    tempo: float,
    output_file: Optional[str] = None
):
    """
    Create visualization of beat detection and clip boundaries.

    Args:
        y: Audio signal
        sr: Sample rate
        beat_times: Detected beat times
        clips: List of clip metadata
        tempo: Estimated tempo
        output_file: Optional path to save figure
    """
    print("Creating visualization...")

    # Compute onset strength for visualization
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)

    # Compute RMS energy
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Panel 1: Waveform with beat markers
    ax1 = axes[0]
    times = np.arange(len(y)) / sr
    ax1.plot(times, y, color='gray', alpha=0.3, linewidth=0.5)
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Beat-Aligned Clip Extraction (Tempo: {tempo:.1f} BPM, {len(clips)} clips)')
    ax1.grid(True, alpha=0.3)

    # Mark beats
    for beat in beat_times:
        ax1.axvline(beat, color='blue', alpha=0.3, linewidth=0.5)

    # Mark clip boundaries
    for clip in clips:
        ax1.axvline(clip['start_time'], color='red', linestyle='--', alpha=0.8, linewidth=1.5)

    # Panel 2: Onset strength with beats
    ax2 = axes[1]
    ax2.plot(onset_times, onset_env, color='purple', linewidth=1)
    ax2.set_ylabel('Onset Strength')
    ax2.grid(True, alpha=0.3)

    # Mark beats
    for beat in beat_times:
        ax2.axvline(beat, color='blue', alpha=0.3, linewidth=0.5,
                   label='Beat' if beat == beat_times[0] else '')

    # Mark clip boundaries
    for i, clip in enumerate(clips):
        label = 'Clip Boundary' if i == 0 else ''
        ax2.axvline(clip['start_time'], color='red', linestyle='--',
                   alpha=0.8, linewidth=1.5, label=label)

    ax2.legend()

    # Panel 3: RMS energy with beats
    ax3 = axes[2]
    ax3.plot(rms_times, rms, color='orange', linewidth=1)
    ax3.set_ylabel('RMS Energy')
    ax3.set_xlabel('Time (seconds)')
    ax3.grid(True, alpha=0.3)

    # Mark beats
    for beat in beat_times:
        ax3.axvline(beat, color='blue', alpha=0.3, linewidth=0.5)

    # Mark clip boundaries
    for clip in clips:
        ax3.axvline(clip['start_time'], color='red', linestyle='--',
                   alpha=0.8, linewidth=1.5)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {output_file}")
    else:
        plt.show()


# =============================================================================
# Main Pipeline
# =============================================================================

def extract_beat_clips(
    audio_file: str,
    output_json: Optional[str] = None,
    output_viz: Optional[str] = None,
    beats_per_measure: int = 4,
    measures_per_clip: int = 4
) -> Dict:
    """
    Main pipeline for beat-aligned clip extraction.

    Args:
        audio_file: Path to audio/video file
        output_json: Optional path to save JSON output
        output_viz: Optional path to save visualization
        beats_per_measure: Number of beats per measure (default: 4 for 4/4 time)
        measures_per_clip: Number of measures per clip (default: 4)

    Returns:
        Dictionary with clips and metadata
    """
    # Stage 1: Beat Detection
    beat_times, tempo, sr, y = extract_beats(audio_file)

    # Calculate total duration
    total_duration = len(y) / sr

    # Stage 2: Measure Grouping
    measures = group_beats_into_measures(beat_times, beats_per_measure)

    # Stage 3: Clip Generation
    clips = create_clips(measures, measures_per_clip, total_duration)

    # Create output
    result = {
        "audio_file": audio_file,
        "total_duration": float(total_duration),
        "tempo_bpm": tempo,
        "total_beats": len(beat_times),
        "total_measures": len(measures),
        "num_clips": len(clips),
        "parameters": {
            "beats_per_measure": beats_per_measure,
            "measures_per_clip": measures_per_clip
        },
        "clips": clips
    }

    # Save JSON output
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved clips metadata: {output_json}")

    # Create visualization
    if output_viz:
        visualize_analysis(y, sr, beat_times, clips, tempo, output_viz)

    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract beat-aligned clips from music videos'
    )

    parser.add_argument(
        'audio_file',
        help='Path to audio or video file'
    )

    parser.add_argument(
        '-o', '--output',
        help='Output JSON file for clips (default: <audio_file>_clips.json)'
    )

    parser.add_argument(
        '-v', '--visualize',
        help='Output visualization file (e.g., output.png)'
    )

    parser.add_argument(
        '--beats-per-measure',
        type=int,
        default=4,
        help='Number of beats per measure (default: 4 for 4/4 time)'
    )

    parser.add_argument(
        '--measures-per-clip',
        type=int,
        default=4,
        help='Number of measures per clip (default: 4)'
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.audio_file).exists():
        print(f"Error: Audio file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    # Set default output file
    output_json = args.output
    if output_json is None:
        output_json = str(Path(args.audio_file).with_suffix('')) + '_clips.json'

    # Run extraction
    try:
        result = extract_beat_clips(
            audio_file=args.audio_file,
            output_json=output_json,
            output_viz=args.visualize,
            beats_per_measure=args.beats_per_measure,
            measures_per_clip=args.measures_per_clip
        )

        # Print summary
        print("\n" + "=" * 60)
        print(f"Beat-aligned clip extraction complete!")
        print(f"Total duration: {result['total_duration']:.1f} seconds")
        print(f"Tempo: {result['tempo_bpm']:.1f} BPM")
        print(f"Total beats: {result['total_beats']}")
        print(f"Total measures: {result['total_measures']}")
        print(f"Created clips: {result['num_clips']}")
        print(f"\nClip details:")
        for clip in result['clips'][:10]:  # Show first 10 clips
            print(f"  Clip {clip['clip_id']:2d}: "
                  f"{clip['start_time']:7.2f}s - {clip['end_time']:7.2f}s "
                  f"({clip['duration']:5.2f}s, "
                  f"{clip['num_measures']} measures, "
                  f"{clip['num_beats']} beats)")
        if len(result['clips']) > 10:
            print(f"  ... and {len(result['clips']) - 10} more clips")
        print("=" * 60)

    except Exception as e:
        print(f"Error during extraction: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
