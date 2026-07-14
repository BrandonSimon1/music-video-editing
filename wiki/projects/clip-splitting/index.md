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

- `process_video.py` — **Primary entry point.** Full pipeline in one command; handles folder layout, caching, filtering, rendering
- `algorithms/allin1.py` — allin1 algorithm plugin (analysis + clip building)
- `algorithms/__init__.py` — Documents the plugin interface for adding new algorithms
- `allin1_clip_extractor.py` — Low-level subcommands (analyze, build-clips, filter, visual-filter, render) used internally by the allin1 plugin
- `rebuild_clips.py` — One-off: reconstruct downbeats from old clips JSON and re-build with new duration logic
- `beat_clip_extractor.py` — Previous script: librosa beat-track approach (kept for reference)

## Usage

### Run the full pipeline

```bash
uv run process_video.py /path/to/session-folder
```

This produces:
```
session-folder/
  video.MOV
  analysis/
    2026-07-14-allin1.json   ← cached; reused on subsequent runs
  clips/
    2026-07-14-allin1/
      clips.json             ← algorithm, params, summary, clip timings
      _visual_cache.json     ← resume cache (internal)
      clip-001.mp4
      clip-002.mp4
      ...
```

### Common options

```bash
# Skip the ~6-hour allin1 step if analysis already cached
uv run process_video.py /path/to/folder   # automatically reuses analysis/<date>-allin1.json

# Different clip length target
uv run process_video.py /path/to/folder --min-duration 20 --max-duration 45

# Skip visual filter (faster, useful for testing)
uv run process_video.py /path/to/folder --no-visual-filter

# Force re-run allin1 even if cache exists
uv run process_video.py /path/to/folder --reanalyze

# Custom run folder name
uv run process_video.py /path/to/folder --name "2026-07-14-tight-cuts"

# Multiple video files in the same folder
uv run process_video.py /path/to/folder --video session.MOV
```

### Low-level subcommands (manual / one-off use)

```bash
# Analyze only
uv run python clip-splitting/allin1_clip_extractor.py analyze video.MOV

# Re-build clips with different duration (no re-analysis needed)
uv run python clip-splitting/allin1_clip_extractor.py build-clips \
    video_analysis.json --min-duration 20 --max-duration 45

# Reconstruct downbeats from an old clips JSON (if no analysis JSON exists)
uv run python clip-splitting/rebuild_clips.py old_clips.json
```

### Adding a new algorithm

Drop a file in `algorithms/`:

```python
# algorithms/my_algo.py
ALGORITHM_NAME = "my-algo"
ALGORITHM_VERSION = "1.0"

def add_args(parser): ...         # register CLI flags
def run(video_file, cache_dir, args) -> list[dict]: ...  # return clips
def get_params(args) -> dict: ... # params to record in clips.json (optional)
```

Then: `uv run process_video.py /path/to/folder --algorithm my-algo`

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
