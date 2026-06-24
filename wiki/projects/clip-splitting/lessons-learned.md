# Clip Splitting: Lessons Learned

Key takeaways from building the beat-aligned clip splitting pipeline.

## Beat Detection

### Downbeat detection is essential for measure alignment
`librosa.beat_track()` detects beats but cannot distinguish beat 1 from beats 2, 3, or 4 within a measure. This "phase ambiguity" means blind grouping of beats into measures will be shifted by an unknown number of beats. Deep learning models like `allin1` solve this by jointly predicting beats and downbeats.

### Global tempo fails on long recordings
A single BPM estimate across a 95-minute recording with multiple songs at different tempos is meaningless. Use models that handle tempo variation locally (allin1, or at minimum, librosa with windowed tempo estimation).

## Structural Awareness

### Fixed-measure grouping is musically incoherent
Grouping every N measures into a clip regardless of musical structure produces clips that start mid-chorus and end mid-verse. Respecting segment boundaries (intro/verse/chorus transitions) produces musically coherent clips even if they have fewer than N measures.

### Segment labels are useful but not perfectly reliable
allin1 provides labels like "intro", "verse", "chorus", etc. These are directionally correct but not 100% accurate, especially in live performance recordings. They're useful for metadata but shouldn't be the sole criterion for filtering.

## Non-Musical Clip Filtering

### Beat density and regularity beat volume thresholds
When the beat tracker runs on talking/silence sections, it hallucinates sparse, irregularly-spaced beats. Two metrics capture this:
- **Beat density** (beats/sec): music > 1.2, talking < 1.1
- **Beat regularity CV**: music < 0.15, talking > 0.3

These are volume-independent — they work regardless of mic placement, gain settings, or room acoustics. Volume/decibel thresholds would require per-session tuning.

### The beat tracker "hallucinates" during non-music
This is actually useful. Rather than trying to detect the absence of music, we detect that the hallucinated beats have distinctive statistical properties (low density, high variability) that cleanly separate them from real musical beats.

## ffmpeg Clip Extraction

### `-c copy` gives slightly loose clip starts
`ffmpeg -c copy` (stream copy, no re-encoding) seeks to the nearest keyframe before the requested start time. This means clip starts may be slightly early. Clip endings are more precise. Re-encoding with `-c:v libx264` would fix this but is much slower.

### Listening tests are the only real validation
Visualizations of beats and clip boundaries looked reasonable for both the librosa and allin1 approaches. Only by actually listening to rendered clips did we discover the librosa approach was badly misaligned. Always render a subset of clips and listen before declaring success.

## Performance

### NATTEN inference dominates CPU processing time
On Intel Mac CPU, a 95-minute recording takes ~6.5 hours total:
- Demucs source separation: ~68 minutes (10%)
- Spectrogram extraction: ~3 minutes (<1%)
- NATTEN attention inference: ~5.5 hours (85%)

GPU would dramatically speed up the NATTEN stage. Consider cloud GPU for batch processing.

## Methodology

### Validate assumptions early with small tests
Before running a 6+ hour analysis, test the approach on a short clip first. This saves hours of wasted computation when assumptions are wrong.

### Iterate: analyze, visualize, listen, refine
The successful workflow was:
1. Run analysis (allin1)
2. Visualize beats/segments/clips
3. Render a subset of clips and listen
4. Identify problems (non-musical clips)
5. Implement fix (beat density filtering)
6. Re-render and validate
