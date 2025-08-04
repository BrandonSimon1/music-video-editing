# Music Video Editing Project

## Overview
This project aims to automatically extract clips from music videos that work well as loops. The goal is to identify complete musical sections that can be seamlessly repeated, rather than just finding interesting transition points.

## Current Status

### What We've Built So Far

1. **Initial Implementation (`main.py`)**
   - Basic novelty detection using librosa
   - Identifies transition points in music using chromagram analysis
   - Good for finding where musical changes occur
   - Issue: Only finds single points, not complete sections suitable for loops

2. **Debug and Testing Tools**
   - `main_debug.py`: Enhanced version with comprehensive logging and memory tracking
   - `test_components.py`: Component testing to isolate issues
   - `test_print.py`: Diagnostic tool for output issues

3. **Improved Audio Processing (`main_sensitive.py`, `main_sensitive_fixed.py`)**
   - Direct ffmpeg audio extraction for better performance
   - Multi-feature analysis (spectral, energy, onset, harmonic)
   - Sensitivity controls for detection threshold
   - Fixed numpy array formatting and type compatibility issues
   - Still focused on transition points rather than complete sections

4. **Loop Detection System (`main_loop_detection.py`)** - CURRENT FOCUS
   - Complete paradigm shift from transition detection to section detection
   - Self-similarity matrix (SSM) to find repeating musical patterns
   - Beat and bar-aligned section boundaries
   - Filters sections by musical duration (2-16 bars)
   - Outputs complete sections with start/end times suitable for loops

### Key Technical Components

- **Audio Extraction**: Using ffmpeg directly for efficient segment processing
- **Feature Analysis**: 
  - Chroma (harmonic content)
  - MFCC (timbral characteristics)
  - Tonnetz (tonal relationships)
  - Beat tracking for tempo and downbeat detection
- **Section Detection**:
  - Self-similarity matrix computation
  - Checkerboard kernel convolution for boundary detection
  - Musical bar alignment for seamless loops

## Project Goals

### Primary Objective
Create clips from music videos that:
- Start at the beginning of a musical section
- End at the end of that section
- Can be seamlessly looped
- Are musically coherent (complete phrases/sections)

### Why This Matters
- Traditional video editing finds "interesting moments" but not complete loops
- Musical sections have natural start/end points that work for repetition
- Beat-aligned boundaries ensure smooth transitions
- Bar-aware sections maintain musical structure

## Usage

### For Loop Detection (Recommended)
```bash
uv run main_loop_detection.py
```

This will:
- Analyze the video's audio track
- Find complete musical sections
- Output section boundaries aligned to musical bars
- Generate ffmpeg commands to extract each loop
- Create visualizations of the analysis

### Output Files
- `loopable_sections.txt`: List of all detected sections with timestamps and bar counts
- `loop_analysis.png`: Visualization of self-similarity matrix and section boundaries
- `loopable_sections.png`: Timeline view of all detected sections

### For Transition Detection (Original Approach)
```bash
uv run main_sensitive_fixed.py
```

This finds transition points but not complete sections - useful for different use cases.

## Technical Requirements
- Python 3.12
- ffmpeg (must be installed separately)
- Dependencies managed via UV (see pyproject.toml)

## Next Steps
1. Test loop detection on various music genres
2. Add quality scoring for loops (how "loopable" is each section)
3. Implement actual video extraction for the detected sections
4. Consider adding visual analysis to align with video cuts
5. Add configuration options for different musical styles/preferences

## Known Issues
- Large video files require segmented processing due to memory constraints
- Currently processes first 3 segments (15 minutes) for testing
- Beat detection assumes 4/4 time signature

## Development Notes
- Always use `uv run` to execute scripts (ensures correct environment)
- The project uses Git LFS for video files
- Temporary audio files are created during processing but cleaned up after