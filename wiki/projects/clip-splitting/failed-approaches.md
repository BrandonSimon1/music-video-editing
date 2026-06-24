# Clip Splitting: Failed Approaches

## Approach: librosa beat_track with blind measure grouping

### The Idea

Use `librosa.beat.beat_track()` to detect beats, group them into measures of 4 beats, and create clips every 4 measures (16 beats). Simple and direct.

### Implementation

See `clip-splitting/beat_clip_extractor.py` for full code.

```python
# Detect beats
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# Group into measures of 4 beats
for i in range(0, len(beat_times), 4):
    measure = beat_times[i:i+4]

# Group measures into clips of 4
for i in range(0, len(measures), 4):
    clip = measures[i:i+4]
```

### Results

On 95 minutes of practice session video (2025-10-30-mcs-practice.MOV):
- Detected 9,096 beats at estimated 95.7 BPM
- Created 568 clips of 4 measures each (~10 seconds per clip)

### Why It Failed

Listening tests on 10 evenly-spaced clips revealed clips were **poorly aligned to actual musical beats and measures**.

**Root Cause 1: No downbeat detection (phase ambiguity)**

`librosa.beat.beat_track()` detects beats but cannot distinguish beat 1 from beats 2, 3, or 4 within a measure. The algorithm blindly groups the first 4 detected beats as "measure 1", but the first detected beat may not be a downbeat. This means every measure boundary could be shifted by 1-3 beats, and there's no way to correct it without downbeat detection.

**Root Cause 2: Single global tempo**

The beat tracker estimates one global tempo (95.7 BPM) for the entire 95-minute recording. In practice, a concert recording contains:
- Multiple songs at different tempos
- Rubato and tempo variations within songs
- Gaps between songs with no rhythm

A single tempo assumption forces the beat grid to be wrong for most of the recording.

**Root Cause 3: No structural awareness**

The 4-measure grouping is completely blind to musical structure. Clips cut arbitrarily through verses, choruses, intros, and transitions. A clip might start in the middle of a chorus and end in the middle of a verse — musically incoherent.

### What We Considered

**librosa.segment.agglomerative()** for structural segmentation was considered but it operates on feature similarity (e.g., chroma), not musical structure per se. It cannot identify downbeats or label sections.

### What We Moved To

**allin1** (All-In-One Music Structure Analyzer) — a deep learning model that jointly predicts:
- **Beats** and **downbeats** (solving phase ambiguity)
- **Segment boundaries** and **segment labels** (intro, verse, chorus, etc.)
- **Per-beat tempo** information

See [approach.md](approach.md) for the current implementation.

### Lessons Learned

1. **Downbeat detection is essential** for measure alignment — beat detection alone has inherent phase ambiguity
2. **Global tempo is insufficient** for long recordings with multiple songs
3. **Musical structure matters** for creating coherent clips — fixed-measure grouping is too rigid
4. **Listening tests are crucial** — visualizations looked reasonable but the clips sounded wrong

---

## Approach: Volume/Decibel Threshold for Non-Musical Clip Filtering

### The Idea

After switching to allin1 and getting correctly beat-aligned clips, many clips still contained talking/tuning/silence between songs. Use volume (RMS energy or decibel level) to filter these out.

### Why It Was Rejected

Before implementing, this approach was considered and rejected because:

1. **Volume depends on recording setup** — mic placement, gain levels, and room acoustics vary between recordings. A threshold tuned for one session would fail on another.
2. **Talking can be louder than quiet music** — in a practice session, between-song conversation is often at a similar or higher volume than quiet musical passages (e.g., solo acoustic guitar).
3. **Not generalizable** — would require per-session threshold tuning.

### What We Used Instead

**Beat density and beat regularity** (coefficient of variation of inter-beat intervals). These metrics are volume-independent and capture the fundamental difference between music (regular, dense beats) and non-music (sparse, erratic hallucinated beats). See [approach.md](approach.md) Stage 4 for details.

### Lesson Learned

When discriminating music from non-music, prefer **rhythmic features** (beat density, regularity) over **energy features** (volume, RMS). Rhythmic features are invariant to recording conditions.
