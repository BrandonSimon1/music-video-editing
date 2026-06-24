# Song Splitting: Working Approach

The successful approach uses **percussive energy and onset rate analysis** to detect gaps between songs.

## Core Insight

Instead of detecting when tempo becomes unstable, we directly measure when **rhythmic musical activity stops**.

During songs:
- High percussive energy (drums playing)
- High onset rate (2-3 onsets/second)

During gaps:
- Low percussive energy (only crowd noise)
- Low onset rate (~0-0.5 onsets/second)

## Pipeline

### Stage 1: Feature Extraction

Extract percussive component and onset envelope:

```python
import librosa

# Load audio
y, sr = librosa.load(audio_file, sr=None)

# Separate percussive elements using HPSS
y_harmonic, y_percussive = librosa.effects.hpss(y)

# Compute onset strength from percussive component
onset_env = librosa.onset.onset_strength(
    y=y_percussive,
    sr=sr,
    aggregate=np.median
)
```

**Why percussive separation?**
- Removes harmonic/melodic content that adds noise
- Leaves only drums/rhythm (what we need for gap detection)
- Crowd noise has minimal percussive content

### Stage 2: Feature Analysis

Compute two features over sliding windows:

**1. Percussive RMS Energy**

```python
window_samples = int(window_size * sr)  # e.g., 10 seconds
hop_samples = int(hop_size * sr)        # e.g., 2 seconds

for i in range(num_windows):
    start = i * hop_samples
    end = start + window_samples

    window_audio = y_percussive[start:end]
    rms_energy = np.sqrt(np.mean(window_audio**2))
```

**2. Onset Rate (onsets per second)**

```python
# Detect all onsets first
onset_frames = librosa.onset.onset_detect(
    onset_envelope=onset_env,
    sr=sr,
    hop_length=512,
    backtrack=False
)
onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

# Count onsets in each window
for window in windows:
    onsets_in_window = np.sum(
        (onset_times >= window_start) & (onset_times < window_end)
    )
    onset_rate = onsets_in_window / window_size  # onsets per second
```

### Stage 3: Gap Detection Score

Combine features into a single gap detection score:

```python
from scipy.signal import medfilt

# Smooth features to reduce noise
percussive_energy = medfilt(percussive_energy, kernel_size=5)
onset_rate = medfilt(onset_rate, kernel_size=5)

# Invert and normalize: low energy/onset rate = high gap score
energy_gap_score = 1 - normalize(percussive_energy)
onset_gap_score = 1 - normalize(onset_rate)

# Combined gap score (equal weights)
gap_score = 0.5 * energy_gap_score + 0.5 * onset_gap_score
```

**Gap score interpretation:**
- High gap score (>0.65) = likely gap between songs
- Low gap score (<0.65) = likely music playing

### Stage 4: Boundary Detection

Use peak detection to find boundaries:

```python
from scipy.signal import find_peaks

time_step = time_points[1] - time_points[0]
min_gap_samples = int(min_gap_duration / time_step)      # e.g., 10 seconds
min_song_samples = int(min_song_duration / time_step)    # e.g., 300 seconds

peaks, properties = find_peaks(
    gap_score,
    height=threshold,              # e.g., 0.65
    distance=min_song_samples,     # Minimum song length
    width=min_gap_samples          # Minimum gap width
)

# Convert peak indices to time
boundaries = [time_points[p] for p in peaks]
boundaries = [0.0] + sorted(boundaries) + [total_duration]
```

### Stage 5: Clip Extraction

Use ffmpeg to extract clips:

```bash
ffmpeg -ss START_TIME -t DURATION -i INPUT.mp3 -c copy -avoid_negative_ts make_zero OUTPUT.mp3
```

**Why `-c copy`?**
- Fast (no re-encoding)
- Preserves original audio quality
- Extracts exact byte ranges

## Parameters

### Default Values

```python
window_size = 10.0        # Feature window size (seconds)
hop_size = 2.0            # Feature hop size (seconds)
threshold = 0.65          # Gap score threshold for detection
min_gap_duration = 10.0   # Minimum gap duration (seconds)
min_song_duration = 300.0 # Minimum song duration (seconds)
smoothing_window = 5      # Median filter kernel size
```

### Tuning Strategy

1. **threshold** (0.5-0.8)
   - Lower = more boundaries detected
   - Higher = fewer boundaries detected
   - Look at gap score visualization to set appropriately

2. **min_song_duration** (180-420s)
   - Prevents detecting boundaries too close together
   - Should be less than typical song length

3. **min_gap_duration** (8-15s)
   - Ignores brief drops in energy/onset rate within songs
   - Should be less than minimum expected gap

4. **window_size** (5-15s)
   - Longer = more stable but less responsive
   - 10s works well for most content

## Visualization

The script creates a 4-panel visualization:

1. **Onset Strength** - Raw onset envelope
2. **Percussive Energy** - RMS energy over time
3. **Onset Rate** - Onsets per second
4. **Gap Score** - Combined detection score with boundaries marked

**Critical:** Visualization allows inspection of results without listening to hours of audio. You can see:
- Where boundaries were detected (red dashed lines)
- Why boundaries were detected (peaks in gap score)
- Whether features make sense (onset rate drops at gaps)

See [Visualization Best Practices](../../best-practices/visualization.md) for more details.

## Performance

**Typical results on concert recordings:**
- Precision: ~85-95% (most detected boundaries are correct)
- Recall: ~80-90% (catches most song transitions)
- Segments: 7-15 for a 1-2 hour recording
- Duration: Most segments 7-11 minutes

**When it works well:**
- Clear gaps between songs (>10 seconds)
- Strong percussion in songs
- Distinct crowd noise vs music

**When it struggles:**
- Very short gaps (<10 seconds)
- Songs without strong percussion
- Transitions with music playing through the gap
- Extreme noise levels drowning out onsets

## Code Location

- Main script: `song-splitting/tempo_segmentation.py`
- Clip extraction: `song-splitting/create_clips.py`
- Original design: `song-splitting/tempo_segmentation_design.md`
