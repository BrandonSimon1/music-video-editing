# Audio Analysis Best Practices

General tips and techniques for audio analysis projects.

## Fundamental Tools

### allin1 (All-In-One Music Structure Analyzer)

Deep learning model for joint beat, downbeat, segment boundary, and segment label prediction. Recommended over librosa for beat/structure tasks on real music.

**Key capabilities:**
- Joint beat and **downbeat** detection (solves phase ambiguity that librosa can't)
- Structural segmentation with labels (intro, verse, chorus, bridge, solo, inst, break, outro)
- Per-beat position tracking (beat 1, 2, 3, 4 within measure)
- BPM estimation

**Why use instead of librosa for beats:**
- Librosa `beat_track()` detects beats but NOT downbeats — you can't tell which beat is beat 1 of a measure
- Librosa estimates a single global tempo, which fails on multi-song recordings
- allin1 uses deep learning (Demucs source separation + NATTEN attention) for much higher accuracy

**Installation:** Complex on macOS x86_64 — see [dependency notes](../../clip-splitting/) or project `pyproject.toml`

**Basic usage:**
```python
import allin1

result = allin1.analyze('audio.mp3', device='cpu')

result.beats        # All beat times (seconds)
result.downbeats    # Downbeat times (beat 1 of each measure)
result.beat_positions  # Beat position within measure (1, 2, 3, 4, ...)
result.segments     # List of Segment(start, end, label)
result.bpm          # Estimated tempo
```

**Performance:** Very slow on CPU (~6.5 hours for 95 minutes on Intel Mac). Much faster on GPU.

**Reference:** Kim et al., "All-In-One Metrical And Functional Structure Analysis With Neighborhood Attentions on Demixed Audio" (ISMIR 2023)

### Librosa

The go-to library for music and audio analysis in Python.

**Key capabilities:**
- Load audio: `librosa.load()`
- Onset detection: `librosa.onset.onset_detect()`, `librosa.onset.onset_strength()`
- Tempo/beat: `librosa.beat.tempo()`, `librosa.beat.beat_track()`
- Spectral features: `librosa.feature.*`
- Time/frequency conversions: `librosa.frames_to_time()`, etc.

**Installation:**
```bash
pip install librosa
```

**Basic usage:**
```python
import librosa
import numpy as np

# Load audio (resamples to 22050 Hz by default)
y, sr = librosa.load('audio.mp3')

# Load at native sample rate
y, sr = librosa.load('audio.mp3', sr=None)

# Compute onset strength
onset_env = librosa.onset.onset_strength(y=y, sr=sr)

# Detect onsets
onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)
```

### Scipy

Signal processing utilities.

**Key uses in audio:**
- Peak detection: `scipy.signal.find_peaks()`
- Filtering: `scipy.signal.medfilt()`, `scipy.signal.butter()`, etc.
- Spectral analysis: `scipy.signal.spectrogram()`

**Example - peak detection:**
```python
from scipy.signal import find_peaks

peaks, properties = find_peaks(
    signal,
    height=0.5,      # Minimum peak height
    distance=100,    # Minimum distance between peaks
    width=10         # Minimum peak width
)
```

## Common Preprocessing Techniques

### Harmonic-Percussive Source Separation (HPSS)

Separates audio into harmonic and percussive components.

```python
y_harmonic, y_percussive = librosa.effects.hpss(y)
```

**When to use:**
- Analyzing rhythm (use percussive component)
- Analyzing melody/harmony (use harmonic component)
- Reducing interference between musical elements

**Why it works:**
- Harmonic content: stable over time, varies in frequency
- Percussive content: brief in time, broad in frequency
- Median filtering in time/frequency separates them

**Example applications:**
- Beat detection from percussive component
- Pitch tracking from harmonic component
- Noise reduction

### Onset Detection

Find the start times of musical events (notes, beats, attacks).

```python
# Compute onset strength envelope
onset_env = librosa.onset.onset_strength(
    y=y,
    sr=sr,
    aggregate=np.median  # Combine frequency bands
)

# Detect onset times
onset_frames = librosa.onset.onset_detect(
    onset_envelope=onset_env,
    sr=sr,
    hop_length=512,
    backtrack=False  # Don't try to refine onset times
)

onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
```

**Parameters to tune:**
- `aggregate`: How to combine frequency bands (median is robust to noise)
- `hop_length`: Time resolution (smaller = finer resolution)
- `backtrack`: Whether to refine onset times (slower but more accurate)

### Normalization

Scale features to [0, 1] range for comparison.

```python
def normalize(arr):
    """Normalize array to [0, 1] range."""
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)
```

**When to use:**
- Combining features with different scales
- Comparing across different audio files
- Before applying thresholds

**Warning:** Normalization loses absolute scale information.

### Smoothing

Reduce noise in time-varying features.

```python
from scipy.signal import medfilt

# Median filter (good for preserving edges)
smoothed = medfilt(signal, kernel_size=5)  # Must be odd

# Moving average (simpler but blurs edges)
window = 5
smoothed = np.convolve(signal, np.ones(window)/window, mode='same')
```

**When to use:**
- Noisy features (e.g., tempo estimates)
- Before peak detection
- Visualizations (makes trends clearer)

**Median vs. mean:**
- Median: preserves sharp transitions, removes outliers
- Mean: smoother but blurs edges

## Feature Engineering

### Energy Features

**RMS Energy:**
```python
# Over entire signal
rms = np.sqrt(np.mean(y**2))

# Over windows
frame_length = 2048
hop_length = 512
rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
```

**When to use:**
- Loudness estimation
- Voice activity detection
- Energy-based segmentation

### Spectral Features

**Spectral Centroid** (brightness):
```python
spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
```

**Spectral Rolloff** (frequency content):
```python
spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
```

**Zero Crossing Rate** (noisiness):
```python
zcr = librosa.feature.zero_crossing_rate(y)
```

**When to use:**
- Timbre analysis
- Instrument recognition
- Speech vs. music discrimination

### Rate Features

**Onset Rate** (events per second):
```python
onset_times = librosa.onset.onset_detect(...)
window_duration = 10.0  # seconds

for window in windows:
    onsets_in_window = np.sum(
        (onset_times >= window_start) & (onset_times < window_end)
    )
    onset_rate = onsets_in_window / window_duration
```

**Why rate features work:**
- Dimensionless (not affected by volume)
- Intuitive interpretation
- Robust to certain types of noise

**Other rates:**
- Beat rate (local tempo)
- Chord change rate
- Spectral flux rate

### Music vs. Non-Music Discrimination

When processing long recordings (e.g., practice sessions), you need to distinguish music from talking, tuning, and silence. Two metrics work well together:

**Beat Density** (beats per second):
```python
beat_density = num_beats_in_clip / clip_duration
```
- Music: typically 1.4-2.5 bps (84-150 BPM)
- Talking/silence: below 1.1 bps (beat tracker hallucinates sparse beats)
- Threshold: >= 1.2 bps to keep

**Beat Regularity** (coefficient of variation of inter-beat intervals):
```python
import numpy as np
intervals = np.diff(beat_times)
beat_cv = np.std(intervals) / np.mean(intervals)
```
- Music: CV < 0.15 (regular, steady tempo)
- Talking/silence: CV > 0.3 (erratic, irregular spacing)
- Threshold: <= 0.25 to keep

**Why NOT volume/decibel thresholds:**
- Volume depends on mic placement and gain settings
- Talking can be louder than quiet music
- Beat density and regularity are independent of absolute volume

**Validated result:** On a 95-minute practice recording, filtering with density >= 1.2 AND CV <= 0.25 removed 82/573 clips (14%), accurately removing all talking/setup sections while keeping all musical content.

## Sliding Window Analysis

Standard approach for time-varying features.

```python
window_size = 10.0  # seconds
hop_size = 2.0      # seconds (overlap for smooth analysis)

sr = 44100
window_samples = int(window_size * sr)
hop_samples = int(hop_size * sr)

features = []
time_points = []

for i in range(0, len(y) - window_samples, hop_samples):
    window = y[i:i + window_samples]

    # Compute feature for this window
    feature_value = compute_feature(window)
    features.append(feature_value)

    # Time point is center of window
    time_sec = (i + window_samples / 2) / sr
    time_points.append(time_sec)

features = np.array(features)
time_points = np.array(time_points)
```

**Parameters:**
- `window_size`: Larger = more stable, less responsive
- `hop_size`: Smaller = finer temporal resolution, more computation

**Rule of thumb:** `hop_size = window_size / 5` gives good overlap

## Combining Features

### Weighted Combination

```python
# Normalize features first
feature1_norm = normalize(feature1)
feature2_norm = normalize(feature2)

# Combine with weights
w1, w2 = 0.6, 0.4
combined = w1 * feature1_norm + w2 * feature2_norm
```

**How to choose weights:**
- Start with equal weights (0.5, 0.5)
- Visualize individual features and combined score
- Adjust based on which feature seems more reliable

### Complementary Features

Best results from features that capture different aspects:

**Example - Song Splitting:**
- Percussive energy: captures presence of drums
- Onset rate: captures rhythmic activity
- Together: more robust than either alone

**Good combinations:**
- Energy + spectral features
- Time-domain + frequency-domain
- Multiple time scales (short-term + long-term)

## Peak Detection

Find significant peaks in a signal.

```python
from scipy.signal import find_peaks

peaks, properties = find_peaks(
    signal,
    height=threshold,    # Minimum height
    distance=min_dist,   # Minimum distance between peaks
    width=min_width,     # Minimum width at half prominence
    prominence=min_prom  # Minimum prominence
)

# Convert to time if needed
peak_times = time_points[peaks]
```

**Parameters:**
- `height`: Absolute threshold
- `distance`: Prevents detecting peaks too close together
- `width`: Ensures peaks are sustained, not noise spikes
- `prominence`: How much peak stands out from surroundings

**Tuning strategy:**
1. Start with just `height`
2. Add `distance` to prevent duplicate detections
3. Add `width` if you're getting noise spikes
4. Add `prominence` for very noisy signals

## Common Pitfalls

### ❌ Using Wrong Sample Rate

```python
# Bad - assumes 22050 Hz
y, sr = librosa.load('audio.mp3')  # Resamples!

# Good - use native rate
y, sr = librosa.load('audio.mp3', sr=None)
```

**Why it matters:**
- Resampling changes frequency content
- Can introduce artifacts
- Affects time calculations

### ❌ Frame/Time Confusion

Librosa works in frames, not samples or seconds.

```python
# Bad
onset_seconds = onset_frames / sr  # Wrong!

# Good
onset_seconds = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
```

**Always use:**
- `librosa.frames_to_time()` for frame → time
- `librosa.time_to_frames()` for time → frame

### ❌ Forgetting Hop Length

Most librosa features use `hop_length=512` by default.

```python
# Bad - assumes hop_length
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

# Better - explicit hop_length
hop_length = 512
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
onset_frames = librosa.onset.onset_detect(
    onset_envelope=onset_env,
    sr=sr,
    hop_length=hop_length
)
```

### ❌ Not Validating Assumptions

**Always test core assumptions before building full pipeline:**

```python
# Test: Can we detect onsets reliably?
onset_times = librosa.onset.onset_detect(...)
print(f"Detected {len(onset_times)} onsets in {duration}s")
print(f"Average rate: {len(onset_times)/duration:.2f} onsets/sec")

# Visualize to check
plt.figure()
librosa.display.waveshow(y, sr=sr, alpha=0.5)
plt.vlines(onset_times, -1, 1, color='r', alpha=0.5)
plt.show()
```

### ❌ Ignoring Edge Effects

Windows at the start/end of audio may have issues.

```python
# Handle edge case
if end_sample > len(y):
    break  # or pad with zeros
```

## Debugging Strategies

### 1. Visualize Everything

See [Visualization Best Practices](visualization.md).

### 2. Start Simple

```python
# Step 1: Load audio
y, sr = librosa.load('test.mp3', sr=None)
print(f"Loaded {len(y)} samples at {sr} Hz ({len(y)/sr:.1f} seconds)")

# Step 2: Compute onset envelope
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
print(f"Onset envelope shape: {onset_env.shape}")
print(f"Onset envelope range: [{np.min(onset_env):.2f}, {np.max(onset_env):.2f}]")

# Step 3: Visualize
plt.plot(onset_env)
plt.show()
```

Build complexity gradually, checking each step.

### 3. Use Test Audio

Create simple test cases:

```python
# Generate test tone with known properties
duration = 10.0
sr = 22050
t = np.linspace(0, duration, int(duration * sr))
frequency = 440  # Hz (A4)

y_test = np.sin(2 * np.pi * frequency * t)
```

Test algorithms on known inputs before real audio.

### 4. Print Statistics

```python
print(f"Feature range: [{np.min(feature):.3f}, {np.max(feature):.3f}]")
print(f"Feature mean: {np.mean(feature):.3f}")
print(f"Feature std: {np.std(feature):.3f}")
print(f"Non-zero values: {np.sum(feature > 0)} / {len(feature)}")
```

Sanity check that values are reasonable.

## Performance Considerations

### Memory Usage

Long audio files can be huge:

```python
# 2-hour audio at 44.1 kHz
samples = 2 * 60 * 60 * 44100  # ~317 million
bytes_per_sample = 4  # float32
memory_mb = samples * bytes_per_sample / (1024**2)  # ~1.2 GB
```

**Solutions:**
- Work with lower sample rate if high frequencies don't matter
- Process in chunks for very long files
- Use appropriate data types (float32 vs float64)

### Speed

**Fast operations:**
- HPSS: O(n log n)
- Onset detection: O(n)
- RMS energy: O(n)

**Slow operations:**
- Tempo estimation: can be slow for long audio
- Some spectral features: O(n log n) to O(n²)

**Speed up:**
- Use smaller hop lengths only when necessary
- Cache intermediate results
- Process in parallel for multiple files

## Recommended Workflow

1. **Design** - Understand the problem and domain
2. **Test assumptions** - Validate on small examples
3. **Implement** - Build pipeline stage by stage
4. **Visualize** - Create multi-panel plots
5. **Iterate** - Refine based on visualization
6. **Validate** - Test on real data
7. **Document** - Record what works and what doesn't

## Resources

**Libraries:**
- librosa: https://librosa.org/
- scipy.signal: https://docs.scipy.org/doc/scipy/reference/signal.html
- allin1: https://github.com/mir-aidj/all-in-one (deep learning beat/structure analysis)

**Learning:**
- librosa tutorial: https://librosa.org/doc/latest/tutorial.html
- MIR (Music Information Retrieval) resources
- allin1 paper: Kim et al., ISMIR 2023

**References:**
- See project-specific wikis for detailed examples
- [Song Splitting](../projects/song-splitting/index.md) for a complete case study
- [Clip Splitting](../projects/clip-splitting/index.md) for beat-aligned clip extraction with allin1
