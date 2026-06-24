# Concert Audio Segmentation via Tempo Stability Analysis

## Background & Problem Space

### Goal
Automatically segment 1-2 hour concert recordings into individual songs (7-15 songs, 7-10 min each) without using volume-based methods.

### Key Constraints
- High background noise prevents volume threshold approaches
- Song transitions contain 10-120 seconds of crowd noise (no music)
- Songs have relatively consistent tempo throughout
- Need to prefer over-splitting (false positives) over under-splitting (missed boundaries)

### Key Insight
When music stops, onset detection and tempo estimation will fail or become highly unstable, creating a detectable signal for segmentation.

---

## High-Level Design

The system will use a **multi-stage tempo stability analysis** approach:

1. **Extract rhythmic features** from the audio
2. **Compute local tempo estimates** over sliding windows
3. **Detect tempo instability regions** where music likely stops
4. **Identify song boundaries** at instability region edges
5. **Output timestamped segments** with tempo metadata

---

## Detailed Design

### Stage 1: Feature Extraction

**Purpose**: Extract onset strength envelope - the foundation for tempo analysis

**Implementation**:
```python
y, sr = librosa.load(audio_file, sr=None)  # Load at native sample rate
onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
```

**Parameters**:
- `sr`: Native sample rate (typically 44.1kHz for audio)
- `aggregate=np.median`: Robust to noise across frequency bands
- `hop_length`: Default 512 samples (~11.6ms at 44.1kHz)

**Output**: Onset strength time series representing rhythmic energy

---

### Stage 2: Local Tempo Analysis

**Purpose**: Estimate tempo in overlapping windows to capture stability

#### Approach A: Windowed Tempo Estimation (Primary)
```python
window_size = 10.0  # seconds - enough to capture multiple beats
hop_size = 2.0      # seconds - overlap windows for smooth detection

tempo_curve = []
tempo_confidence = []

for window_start in range(0, len(onset_env), hop_samples):
    window_env = onset_env[window_start:window_start+window_samples]

    # Estimate tempo in this window
    tempo, confidence = librosa.beat.tempo(
        onset_envelope=window_env,
        sr=sr,
        aggregate=None  # Get per-frame estimates
    )

    tempo_curve.append(tempo)
    tempo_confidence.append(confidence)
```

#### Approach B: Tempogram Analysis (Alternative/Complementary)
```python
tempogram = librosa.feature.tempogram(
    onset_envelope=onset_env,
    sr=sr,
    win_length=384,  # ~8.7 sec at 44.1kHz with default hop
    hop_length=512
)

# Compute dominant tempo in each frame
dominant_tempo = librosa.core.tempo_frequencies(
    tempogram.shape[0],
    sr=sr
)[np.argmax(tempogram, axis=0)]
```

**Parameters to tune**:
- `window_size`: 5-15 seconds (longer = more stable, but less responsive)
- `hop_size`: 1-5 seconds (smaller = more temporal resolution)

**Output**: Time series of tempo estimates and confidence scores

---

### Stage 3: Tempo Stability Metrics

**Purpose**: Quantify how stable/unstable tempo is in each region

#### Metrics to compute:

1. **Tempo Variance** (primary signal)
   ```python
   # For each position, compute variance over surrounding window
   stability_window = 5  # seconds
   tempo_variance = rolling_variance(tempo_curve, stability_window)
   ```

2. **Tempo Confidence** (quality gate)
   ```python
   # Low confidence indicates no clear beat/tempo
   is_low_confidence = tempo_confidence < threshold  # e.g., 0.3
   ```

3. **Onset Strength Consistency**
   ```python
   # Measure periodicity of onsets (strong = music, weak = noise)
   onset_autocorr = librosa.autocorrelate(onset_env)
   onset_periodicity = np.max(onset_autocorr[min_lag:max_lag])
   ```

#### Instability Score (combined metric):
```python
instability_score = (
    normalize(tempo_variance) * w1 +
    (1 - normalize(tempo_confidence)) * w2 +
    (1 - normalize(onset_periodicity)) * w3
)
# Suggested weights: w1=0.5, w2=0.3, w3=0.2
```

**Output**: Instability score time series (high = likely no music)

---

### Stage 4: Boundary Detection

**Purpose**: Convert instability scores to discrete song boundaries

#### Algorithm:
```python
# 1. Find instability regions above threshold
instability_regions = instability_score > threshold

# 2. Apply minimum duration filter (remove spurious spikes)
min_gap_duration = 8.0  # seconds (below your min 10s, to be safe)
filtered_regions = filter_short_regions(instability_regions, min_gap_duration)

# 3. Find edges of instability regions
boundaries = []
in_gap = False
gap_start = 0

for i, is_unstable in enumerate(filtered_regions):
    if is_unstable and not in_gap:
        # Start of gap
        gap_start = i
        in_gap = True
    elif not is_unstable and in_gap:
        # End of gap - middle point is the boundary
        gap_end = i
        boundary_time = frames_to_time((gap_start + gap_end) / 2)
        boundaries.append(boundary_time)
        in_gap = False

# 4. Add start and end of recording
boundaries = [0] + sorted(boundaries) + [total_duration]
```

#### Alternative: Use peak detection on instability score
```python
from scipy.signal import find_peaks

peaks, properties = find_peaks(
    instability_score,
    height=threshold,      # Minimum instability
    distance=min_song_samples,  # Minimum song length (e.g., 5 min)
    width=min_gap_samples      # Minimum gap width (e.g., 8 sec)
)
```

**Parameters to tune**:
- `threshold`: 0.5-0.8 of normalized instability (lower = more boundaries)
- `min_gap_duration`: 8-15 seconds
- `min_song_duration`: 300-420 seconds (5-7 min) to prevent over-splitting

**Output**: List of boundary timestamps in seconds

---

### Stage 5: Segment Characterization & Output

**Purpose**: Provide metadata about each detected segment

#### For each segment (between consecutive boundaries):

1. **Compute tempo statistics**:
   ```python
   segment_tempos = tempo_curve[start_idx:end_idx]
   tempo_mean = np.mean(segment_tempos)
   tempo_std = np.std(segment_tempos)
   tempo_range = (np.min(segment_tempos), np.max(segment_tempos))
   ```

2. **Identify boundary trigger** (what caused the split):
   ```python
   # Look at instability region around boundary
   boundary_idx = time_to_frame(boundary_time)
   region = slice(boundary_idx - window, boundary_idx + window)

   trigger_tempo_var = tempo_variance[region].max()
   trigger_confidence = tempo_confidence[region].min()

   # What tempo did it transition from/to?
   prev_segment_tempo = tempo_mean[previous_segment]
   next_segment_tempo = tempo_mean[next_segment]
   ```

#### Output Format (JSON or CSV):
```json
{
  "segments": [
    {
      "segment_id": 1,
      "start_time": 0.0,
      "end_time": 487.3,
      "duration": 487.3,
      "tempo_mean": 128.5,
      "tempo_std": 3.2,
      "tempo_range": [124, 135],
      "next_boundary": {
        "time": 487.3,
        "instability_score": 0.87,
        "tempo_from": 128.5,
        "tempo_to": null,
        "gap_duration": 15.2
      }
    },
    {
      "segment_id": 2,
      "start_time": 487.3,
      "end_time": 892.1,
      ...
    }
  ]
}
```

---

## Implementation Strategy

### Phase 1: Prototype (validate approach)
- Implement basic windowed tempo analysis
- Detect instability regions with simple threshold
- Output timestamps and visualizations
- Test on sample concert recording
- Tune parameters based on results

### Phase 2: Refinement
- Add combined instability metrics
- Implement robust boundary detection with filtering
- Add segment characterization
- Create visualization dashboard (tempo curve + boundaries)

### Phase 3: Robustness
- Handle edge cases (very long/short gaps, tempo changes within songs)
- Add confidence scoring for boundaries
- Optimize performance for long recordings

---

## Key Parameters to Tune

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `window_size` | 10s | 5-15s | Tempo estimation window - longer = more stable but less responsive |
| `hop_size` | 2s | 1-5s | Temporal resolution of analysis |
| `instability_threshold` | 0.65 | 0.5-0.8 | Sensitivity - lower = more boundaries detected |
| `min_gap_duration` | 10s | 8-15s | Ignore brief tempo instabilities within songs |
| `min_song_duration` | 300s | 240-420s | Prevent splitting songs internally |
| `stability_window` | 5s | 3-10s | Window for computing tempo variance |

**Tuning strategy**: Start with conservative values (detect fewer boundaries), then gradually increase sensitivity until over-splitting occurs, then back off slightly.

---

## Visualization Recommendations

For validation and tuning, create plots showing:
1. **Onset strength envelope** (top panel)
2. **Tempo curve with confidence** (middle panel, with shaded low-confidence regions)
3. **Instability score** (bottom panel, with threshold line and detected boundaries marked)

This will make it easy to see why boundaries were detected and adjust parameters.

---

## Expected Performance

**Precision**: High (>90%) - instability during crowd-only periods should be very distinctive
**Recall**: Medium-High (70-85%) - may miss boundaries if:
- Very short gaps (<10s)
- Songs have very similar tempos and smooth transitions
- Extreme noise drowns out onset detection

**Trade-off**: Since you prefer over-splitting, we'll tune for high recall at the cost of some false positives.