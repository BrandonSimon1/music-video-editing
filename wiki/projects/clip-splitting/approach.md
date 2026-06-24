# Clip Splitting: Technical Approach (allin1)

This document describes the current implementation of structure-aware clip extraction using the All-In-One Music Structure Analyzer (`allin1`).

For the previous librosa-based approach and why it failed, see [failed-approaches.md](failed-approaches.md).

## Overview

The approach uses `allin1` — a deep learning model that jointly predicts beats, downbeats, segment boundaries, and segment labels. The analysis output is saved once as a raw JSON; clip building is a separate step so duration parameters can be changed without re-running the 6.5-hour analysis.

## What allin1 Provides

From a single audio/video file, `allin1.analyze()` returns:

| Field | Description |
|-------|-------------|
| `bpm` | Estimated tempo |
| `beats` | List of all beat times (seconds) |
| `downbeats` | List of downbeat times (beat 1 of each measure) |
| `beat_positions` | Beat position within measure (1, 2, 3, 4, 1, 2, 3, 4...) |
| `segments` | List of `Segment(start, end, label)` — structural sections |

Segment labels include: `intro`, `verse`, `chorus`, `bridge`, `outro`, `inst`, `solo`, etc.

### How It Works Internally

1. **Source separation** (Demucs) — separates audio into stems (drums, bass, vocals, other)
2. **Spectrogram extraction** — computes spectrograms for each stem
3. **Neural network inference** — a model with dilated neighborhood attention (NATTEN) jointly predicts all four outputs from the multi-stem spectrograms

Reference: Kim et al., "All-In-One Metrical And Functional Structure Analysis With Neighborhood Attentions on Demixed Audio" (ISMIR 2023)

## Pipeline Stages

### Stage 1: analyze

Runs allin1 and saves a raw `_analysis.json` containing beats, downbeats, segments, and BPM. No clip building happens here — this output is the stable artifact that can be reused with different clip parameters.

```bash
uv run python clip-splitting/allin1_clip_extractor.py analyze video.MOV
# → video_analysis.json
```

Processing time for a 95-minute recording on Intel Mac CPU:
- Source separation (Demucs): ~68 minutes
- Spectrogram extraction: ~3 minutes
- Neural network inference (NATTEN): ~5.5 hours
- Total: ~6.5 hours (dominated by NATTEN attention on CPU)

**Analysis JSON format:**
```json
{
  "audio_file": "video.MOV",
  "total_duration": 5700.0,
  "bpm": 95,
  "total_beats": 8881,
  "total_downbeats": 2220,
  "total_segments": 256,
  "beats": [0.5, 1.13, 1.76, ...],
  "downbeats": [0.5, 3.02, 5.54, ...],
  "segments": [
    {"start": 0.0, "end": 30.5, "label": "intro"},
    {"start": 30.5, "end": 90.2, "label": "verse"}
  ]
}
```

### Stage 2: build-clips

Takes the analysis JSON and groups downbeats into clips. Duration logic:

1. **Target**: first multiple-of-4 measure count whose duration is ≥ 30s and ≤ 60s
2. **Fallback**: any measure count in the [30, 60] window if no multiple-of-4 lands there
3. **Last resort**: measure count whose duration is closest to 45s (window center)

Each clip advances to the end of the previous one — no gaps, no overlaps.

```bash
uv run python clip-splitting/allin1_clip_extractor.py build-clips video_analysis.json
# → video_clips.json

# Adjust duration window
uv run python clip-splitting/allin1_clip_extractor.py build-clips \
  video_analysis.json --min-duration 20 --max-duration 45
```

**Clips JSON format:**
```json
{
  "audio_file": "video.MOV",
  "num_clips": 161,
  "clip_parameters": {"min_duration": 30.0, "max_duration": 60.0, ...},
  "segments": [...],
  "clips": [
    {
      "clip_id": 1,
      "start_time": 53.81,
      "end_time": 87.23,
      "duration": 33.42,
      "num_measures": 16,
      "num_beats": 64,
      "beat_density": 1.915,
      "beat_regularity_cv": 0.031,
      "segment_labels": ["verse"],
      "beats": [53.81, 54.44, ...]
    }
  ]
}
```

### Stage 3: Visualization

```bash
uv run python clip-splitting/allin1_clip_extractor.py build-clips \
  video_analysis.json -v viz.png
```

The visualization shows:
- **Panel 1**: Beats (blue), downbeats (green), segment boundaries (red dashed) with section labels
- **Panel 2**: Clip boundaries as colored spans with clip IDs

### Stage 4: Beat-Density Filter

Practice sessions contain talking, tuning, and silence between songs. Two per-clip metrics distinguish music from non-music:

| Metric | Music | Talking/Silence |
|--------|-------|-----------------|
| `beat_density` (beats/sec) | > 1.2 bps (typically 1.4-2.5) | < 1.1 bps |
| `beat_regularity_cv` (coeff. of variation) | < 0.15 (regular) | > 0.3 (erratic) |

Default filter: keep clips where `beat_density >= 1.2` AND `beat_cv <= 0.25`.

```bash
uv run python clip-splitting/allin1_clip_extractor.py filter video_clips.json
# → video_clips_filtered.json

# Preview without writing
uv run python clip-splitting/allin1_clip_extractor.py filter video_clips.json --dry-run

# Adjust thresholds
uv run python clip-splitting/allin1_clip_extractor.py filter video_clips.json \
  --min-beat-density 1.0 --max-beat-cv 0.3
```

### Stage 5: Visual Filter (Claude vision)

After beat-density filtering, an optional visual pass checks whether people are **visibly playing instruments** in each clip. This catches edge cases that the audio-only filter misses: clips where the band is on stage but not yet playing, someone walking to their instrument, etc.

**How it works:**
1. ffmpeg extracts a JPEG frame via pipe at the midpoint of each clip (resized to 640px wide, no temp files)
2. Frame is base64-encoded and sent to `claude -p` via `--input-format stream-json`
3. Prompt: *"Are people actively playing musical instruments in this image? Answer YES or NO, then one brief sentence."*
4. Clips where Claude answers NO are removed

**Key implementation detail:** `claude -p` does not accept image file paths as arguments. Images must be sent as base64 JSON piped to stdin with `--input-format stream-json --output-format stream-json --verbose`. The assistant response is parsed from the stream-json output.

**Resume / caching:** Results are saved to a sidecar `_visual_cache.json` so an interrupted run can pick up where it left off.

**Parallelism:** 8 concurrent `claude -p` subprocesses by default (`--workers`).

**Multiple frames (`--frames 3`):** Sample 3 frames per clip at 25%, 50%, 75% and use majority vote.

**Authentication:** Uses Claude Code's existing auth — no `ANTHROPIC_API_KEY` needed.

```bash
uv run python clip-splitting/allin1_clip_extractor.py visual-filter \
  video_clips_filtered.json video.MOV
# → video_clips_filtered_visual.json

# Dry run, resume, multi-frame
uv run python clip-splitting/allin1_clip_extractor.py visual-filter \
  video_clips_filtered.json video.MOV --dry-run
uv run python clip-splitting/allin1_clip_extractor.py visual-filter \
  video_clips_filtered.json video.MOV --no-cache
uv run python clip-splitting/allin1_clip_extractor.py visual-filter \
  video_clips_filtered.json video.MOV --frames 3 --workers 16
```

### Stage 6: Rendering

Clips are rendered with `ffmpeg -c copy` for fast extraction (no re-encoding).

Note: `ffmpeg -c copy` seeks to the nearest keyframe, which can cause clip starts to be slightly before the requested time. Clip endings tend to be more precisely aligned. Re-encoding with `-c:v libx264` would give frame-exact cuts but is much slower.

```bash
uv run python clip-splitting/allin1_clip_extractor.py render \
  video_clips_filtered_visual.json video.MOV -o rendered/

# Render N evenly-spaced clips for a listening test
uv run python clip-splitting/allin1_clip_extractor.py render \
  video_clips_filtered_visual.json video.MOV -o test/ --evenly-spaced 10
```

## Rebuilding Clips from Old Data

If you have an old clips JSON (built before the analysis/build-clips split) but no `_analysis.json`, use `rebuild_clips.py` to reconstruct downbeats and re-build clips with the current duration logic:

```bash
uv run python clip-splitting/rebuild_clips.py old_clips.json
# → old_clips_rebuilt.json
```

It recovers downbeats by treating each clip's `start_time` as a downbeat and extracting intermediate measure starts from `beats[0::4]` within each clip (valid for 4/4 time, 4-measure clips).

## Dependencies

- `allin1>=1.1.0` — structure analysis (lazy import — only needed for `analyze` subcommand)
- `torch==2.2.2` — ML framework (CPU, from pytorch-cpu index)
- `natten==0.17.3` — neighborhood attention (built from source with `--no-build-isolation`)
- `madmom` — beat processing (built from git HEAD)
- `claude` CLI — Claude Code CLI for visual filter stage (uses existing `claude` auth)

See `pyproject.toml` for full dependency config and installation notes.
