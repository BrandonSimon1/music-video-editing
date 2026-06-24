# Beat-Aligned Clip Splitting Project

Automatically split music videos into clips aligned with musical beats, downbeats, and structural sections.

## Problem Statement

**Goal:** Create short video clips from long music videos that are:
- Aligned with the musical beat (not arbitrary time boundaries)
- Starting on actual downbeats (beat 1 of a measure)
- Between 30 and 60 seconds long (first multiple of 4 measures after 30s)
- Suitable for editing and sharing

**Input:**
- Long music video files (e.g., 90+ minute practice sessions)
- May contain multiple songs with varying tempos

## Solution Approach

### Milestone 1: Structure-Aware Clip Extraction (CURRENT)

Uses `allin1` (All-In-One Music Structure Analyzer) for joint beat, downbeat, segment boundary, and segment label prediction.

**Pipeline:**
1. **Analysis** — allin1 predicts beats, downbeats, segments, and BPM; saved to `_analysis.json`
2. **Clip Building** — Group downbeats into clips targeting 30–60s (first multiple-of-4 measures ≥ 30s)
3. **Beat-Density Filtering** — Remove non-musical clips (talking, tuning, silence) using beat density and regularity metrics
4. **Visual Filter** — Ask Claude (vision) whether people are actively playing in each clip; remove clips where they're not
5. **Rendering** — Extract clips with ffmpeg

**Key design decisions:**
- `analyze` saves a raw `_analysis.json` (beats, downbeats, segments) — no clip building
- `build-clips` is a separate step so duration parameters can be changed without re-running the 6.5-hour analysis
- `rebuild_clips.py` reconstructs downbeats from old clips JSONs if no analysis JSON exists

**Key Improvements over previous approach:**
- Downbeats detected by deep learning (no phase ambiguity)
- No single global tempo assumption
- Analysis output is decoupled from clip-building parameters

### Previous Approach (Failed)

Used `librosa.beat.beat_track()` with blind measure grouping. Failed due to:
- No downbeat detection (phase ambiguity)
- Single global tempo across 95 minutes
- No structural awareness

See [failed-approaches.md](failed-approaches.md) for details.

## Files

- `allin1_clip_extractor.py` — Main script (analyze, build-clips, filter, visual-filter, render subcommands)
- `rebuild_clips.py` — One-off: reconstruct downbeats from old clips JSON and re-build with new duration logic
- `beat_clip_extractor.py` — Previous script: librosa beat-track approach (kept for reference)
- `visualize_beats_detail.py` — Detailed beat visualization
- `analyze_tempo_variation.py` — Tempo stability analysis

## Usage

```bash
# Step 1: Analyze video — saves _analysis.json (beats, downbeats, segments)
uv run python clip-splitting/allin1_clip_extractor.py analyze video.MOV

# Step 2: Build clips from analysis JSON (30-60s, multiples of 4 measures)
uv run python clip-splitting/allin1_clip_extractor.py build-clips video_analysis.json

# Step 3: Beat-density filter (removes talking/silence)
uv run python clip-splitting/allin1_clip_extractor.py filter video_clips.json

# Step 4: Visual filter (removes non-playing clips via Claude vision)
uv run python clip-splitting/allin1_clip_extractor.py visual-filter \
    video_clips_filtered.json video.MOV

# Step 5: Render
uv run python clip-splitting/allin1_clip_extractor.py render \
    video_clips_filtered_visual.json video.MOV -o rendered/

# Re-build clips with different duration (no re-analysis needed)
uv run python clip-splitting/allin1_clip_extractor.py build-clips \
    video_analysis.json --min-duration 20 --max-duration 45

# Reconstruct downbeats from an old clips JSON (if no analysis JSON exists)
uv run python clip-splitting/rebuild_clips.py old_clips.json
```

## Status

Milestone 1 is **complete and validated** on the 2025-10-30 practice recording.

## Results (2025-10-30 Practice Recording)

| Metric | Value |
|--------|-------|
| Recording length | 94.5 minutes |
| BPM (global estimate) | 95 |
| Total beats | 8,881 |
| Total downbeats | 2,220 |
| Structural segments | 256 |
| Processing time (CPU) | ~6.5 hours |

### Pipeline run (30–60s clips)

| Stage | Clips |
|-------|-------|
| build-clips (30–60s) | 161 |
| After beat-density filter | 127 |
| After visual filter | 123 |
| Rendered to `rendered-rebuilt/` | 123 |

Clip durations: 30–45s, median ~35s.

## Learn More

- [Technical Approach](approach.md) — Current allin1-based implementation
- [Failed Approaches](failed-approaches.md) — Why librosa beat_track didn't work, and why volume thresholds were rejected for filtering
- [Lessons Learned](lessons-learned.md) — Key takeaways from building this pipeline
