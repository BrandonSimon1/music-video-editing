# Song Splitting: Failed Approaches

Documentation of approaches that didn't work and why. Understanding failures is critical for future projects.

## Approach 1: Windowed Tempo Estimation

### The Idea

Estimate tempo in overlapping windows. When music stops, tempo estimation should become unstable or fail, creating a detectable signal.

### Implementation

```python
window_size = 10.0  # seconds
hop_size = 2.0      # seconds

for window in windows:
    tempo = librosa.beat.tempo(
        onset_envelope=window_onset_env,
        sr=sr,
        hop_length=hop_length
    )

    # Compute confidence
    onset_var = np.var(window_onset_env)
    onset_mean = np.mean(window_onset_env)
    confidence = onset_var / (onset_mean + 1e-8)
```

Then compute tempo variance over time as instability signal.

### Why It Failed

**Problem 1: Extremely Noisy Tempo Estimates**
- Tempo estimates ranged wildly from 40-300 BPM even during songs
- Tempo curve was unstable everywhere, not just at gaps
- 10-second windows don't have enough context for reliable tempo estimation

**Problem 2: No Clear Signal**
- Tempo variance was high throughout the recording
- Couldn't distinguish "unstable tempo in a song" from "no tempo in a gap"
- Instability score hovered around 0.4-0.5 everywhere

**Visualization Evidence:**
- Middle panel showed wild tempo swings (40-300 BPM) throughout
- Bottom panel showed instability everywhere, no clear peaks at boundaries
- Only 2-3 boundaries detected instead of expected 7-15

### Root Cause

`librosa.beat.tempo()` is designed to estimate tempo for **entire songs**, not short windows:
- Needs sufficient rhythmic context (many beats)
- Short windows amplify noise and edge effects
- Returns single "best guess" even when signal is garbage

### Lessons Learned

1. Don't use algorithms outside their intended use case
2. Tempo estimation needs longer context than we can provide
3. Indirect measurements (tempo → instability → gaps) introduce too much noise

## Approach 2: Tempogram Analysis

### The Idea

Use `librosa.feature.tempogram()` instead of windowed tempo estimation. The tempogram is a time-frequency representation of tempo, potentially more robust.

### Implementation

```python
# Compute tempogram
tempogram = librosa.feature.tempogram(
    onset_envelope=onset_env,
    sr=sr,
    hop_length=hop_length,
    win_length=win_length
)

# Get tempo frequency axis
tempo_frequencies = librosa.core.tempo_frequencies(
    tempogram.shape[0],
    hop_length=hop_length,
    sr=sr
)

# Extract dominant tempo at each frame
tempo_curve = tempo_frequencies[np.argmax(tempogram, axis=0)]
```

### Why It Failed

**Complete Failure:**
- Produced tempo values of ~0 BPM
- JSON output showed Infinity and NaN values
- Instability score was completely flat at exactly 0.65

**Likely Causes:**
1. Bug in how we extracted dominant tempo from tempogram
2. Edge case in normalization producing divide-by-zero
3. Tempogram might have all-zero or near-zero values for some frames

### Visualization Evidence

- Tempo panel showed values near 0 (should be 60-180 BPM)
- Gap score completely flat at threshold value
- No boundaries detected (1 segment for entire 98-minute recording)

### Why We Didn't Debug Further

At this point, it became clear that **tempo-based analysis was fundamentally not working** for this audio. Rather than fix implementation bugs, we pivoted to a completely different approach.

### Lessons Learned

1. When an approach has multiple fundamental issues, sometimes it's better to pivot than debug
2. Don't get attached to a solution that isn't working

## Approach 3: Adding Percussive Separation (Still Tempo-Based)

### The Idea

Maybe tempo estimation is noisy because harmonic/melodic content interferes. Try extracting only percussive elements first:

```python
y_harmonic, y_percussive = librosa.effects.hpss(y)
onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr, aggregate=np.median)
```

Then continue with windowed tempo estimation.

### Results

**Marginal improvement but still failed:**
- Tempo curve still very noisy (40-300 BPM)
- Only 2-3 boundaries detected instead of 7-15
- Instability score still everywhere, not concentrated at gaps

**Why it helped a bit:**
- Cleaner onset detection from percussive component
- Slightly less extreme tempo swings

**Why it wasn't enough:**
- Core problem was tempo estimation itself, not the input signal
- Percussive separation was good (we kept it) but tempo estimation was still broken

### Lessons Learned

1. Good preprocessing (HPSS) can help, but won't fix a fundamentally broken approach
2. This preprocessing later became useful in the working solution

## Common Thread: Why Tempo-Based Approaches Failed

### Theoretical Problem

**Tempo requires periodicity:**
- Tempo estimation looks for repeating patterns at regular intervals
- During songs: clear periodic beats → good tempo estimates (in theory)
- During gaps: no periodic structure → tempo estimation fails or latches onto noise

**But in practice:**
- Short windows (10s) don't capture enough beats for robust estimation
- Long windows (30s+) lose temporal resolution (can't pinpoint boundaries)
- Concert recordings have complex rhythms, tempo changes, crowd noise
- Tempo estimators weren't designed for this use case

### Practical Problem

**Signal chain too long:**
```
Audio → Onset Detection → Tempo Estimation → Tempo Variance → Instability Score → Boundaries
```

Each step introduces noise and error. By the time we compute instability score, the signal is buried in noise.

### The Better Approach

**Direct measurement:**
```
Audio → Percussive Separation → Onset Rate → Gap Score → Boundaries
```

Shorter chain, simpler features, direct interpretation:
- "Are beats happening?" is easier to answer than "What is the tempo and how stable is it?"
- Onset rate is a simple count, not a complex estimation
- Clear physical meaning: 0 onsets/sec = no music

## General Lessons

1. **Start simple** - Try the most direct measurement first
2. **Visualize everything** - Visualizations revealed tempo was broken immediately
3. **Know when to pivot** - Don't debug broken approaches endlessly
4. **Understand algorithm assumptions** - Tempo estimators need full songs, not windows
5. **Short signal chains** - Fewer processing steps = less accumulated error
6. **Direct > Indirect** - Measure what you care about directly when possible

## What We Kept From Failed Approaches

Not everything was wasted:

1. **Harmonic-Percussive Separation (HPSS)** - Excellent preprocessing, kept it
2. **Visualization Framework** - 4-panel plots became our primary debugging tool
3. **Peak Detection for Boundaries** - This part always worked
4. **Parameter Structure** - window_size, hop_size, threshold, etc.
5. **Understanding of the problem** - Learned what makes gaps detectable

The journey from failure to success taught us what actually matters in the signal.
