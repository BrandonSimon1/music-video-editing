# Song Splitting Project

Automatic segmentation of 1-2 hour concert recordings into individual songs (7-15 songs, 7-10 minutes each).

## Problem Statement

**Goal:** Segment long concert recordings into individual songs without using volume-based methods.

**Constraints:**
- High background noise prevents volume threshold approaches
- Song transitions contain 10-120 seconds of crowd noise (no music)
- Songs have relatively consistent tempo throughout
- Prefer over-splitting (false positives) over under-splitting (missed boundaries)

## Solution

We developed an approach based on **percussive energy and onset rate analysis**:

1. Extract percussive component using harmonic-percussive source separation (HPSS)
2. Compute percussive energy (RMS) and onset rate over sliding windows
3. Combine into a gap detection score (high when both features are low)
4. Detect boundaries at gap score peaks
5. Extract clips using ffmpeg

See [approach.md](approach.md) for technical details.

## Key Results

- Successfully detects 7-15 segments in typical concert recordings
- Most segments are 7-11 minutes (typical song length)
- Onset rate provides the strongest signal (drops to ~0 during gaps)

## Files

- `tempo_segmentation.py` - Main segmentation script
- `create_clips.py` - Extract clips from segmentation results
- Design document in `tempo_segmentation_design.md`

## Usage

```bash
# Segment a concert recording
./tempo_segmentation.py concert.mp3 -v visualization.png

# Create individual song clips
./create_clips.py concert_segments.json concert.mp3
```

## Learn More

- [Technical Approach](approach.md) - Implementation details
- [Failed Approaches](failed-approaches.md) - What didn't work and why
- [Lessons Learned](lessons-learned.md) - Key insights and takeaways
