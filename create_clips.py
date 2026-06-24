#!/usr/bin/env python3
"""
Create audio clips from segmentation results using ffmpeg.

Reads a segments JSON file and extracts each segment as a separate audio file.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def create_clips(
    segments_json: str,
    audio_file: str,
    output_dir: str = None,
    prefix: str = "segment"
):
    """
    Create audio clips from segmentation results.

    Args:
        segments_json: Path to segments JSON file
        audio_file: Path to original audio file
        output_dir: Directory to save clips (default: same as audio file)
        prefix: Prefix for output files (default: "segment")
    """
    # Load segments
    print(f"Loading segments from: {segments_json}")
    with open(segments_json, 'r') as f:
        data = json.load(f)

    segments = data['segments']
    print(f"Found {len(segments)} segments")

    # Determine output directory
    if output_dir is None:
        output_dir = Path(audio_file).parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get audio file extension
    audio_ext = Path(audio_file).suffix

    # Create clips
    print(f"\nCreating clips in: {output_dir}")
    print("=" * 60)

    for segment in segments:
        seg_id = segment['segment_id']
        start_time = segment['start_time']
        end_time = segment['end_time']
        duration = segment['duration']

        # Format output filename
        output_file = output_dir / f"{prefix}_{seg_id:02d}{audio_ext}"

        print(f"Segment {seg_id:2d}: {start_time:7.1f}s - {end_time:7.1f}s ({duration:6.1f}s) -> {output_file.name}")

        # Build ffmpeg command
        # -ss: start time
        # -t: duration
        # -i: input file
        # -c copy: copy codec (fast, no re-encoding)
        # -avoid_negative_ts make_zero: handle potential timestamp issues
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),
            '-t', str(duration),
            '-i', audio_file,
            '-c', 'copy',
            '-avoid_negative_ts', 'make_zero',
            '-y',  # Overwrite output files
            str(output_file)
        ]

        # Run ffmpeg
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: ffmpeg failed for segment {seg_id}")
            print(f"  {e.stderr}")
            continue

    print("=" * 60)
    print(f"\nCreated {len(segments)} clips in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Create audio clips from segmentation results using ffmpeg'
    )

    parser.add_argument(
        'segments_json',
        help='Path to segments JSON file (e.g., audio_segments.json)'
    )

    parser.add_argument(
        'audio_file',
        help='Path to original audio file'
    )

    parser.add_argument(
        '-o', '--output-dir',
        help='Output directory for clips (default: same as audio file)'
    )

    parser.add_argument(
        '-p', '--prefix',
        default='segment',
        help='Prefix for output files (default: segment)'
    )

    args = parser.parse_args()

    # Validate input files
    if not Path(args.segments_json).exists():
        print(f"Error: Segments JSON file not found: {args.segments_json}", file=sys.stderr)
        sys.exit(1)

    if not Path(args.audio_file).exists():
        print(f"Error: Audio file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg not found. Please install ffmpeg first.", file=sys.stderr)
        sys.exit(1)

    # Create clips
    try:
        create_clips(
            segments_json=args.segments_json,
            audio_file=args.audio_file,
            output_dir=args.output_dir,
            prefix=args.prefix
        )
    except Exception as e:
        print(f"Error creating clips: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
