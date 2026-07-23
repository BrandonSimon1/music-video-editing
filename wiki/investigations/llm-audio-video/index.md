# Investigation: LLM-Augmented Audio/Video Analysis

## Goal

Replace (or substantially augment) the allin1-based clip-splitting pipeline with an approach that:

- Requires no heavy ML model dependencies (no Demucs, no NATTEN, no madmom)
- Runs in minutes, not hours, on a CPU
- Works on recordings with bad or inconsistent audio quality
- Handles both **song boundary detection** (splitting a session into songs) and **clip/phrase boundary detection** (splitting songs into usable clips)
- Is robust enough to offer as a service

The core hypothesis: **traditional audio feature extraction gives us cheap, fast signal; LLM vision/reasoning gives us flexible, context-aware interpretation**. Used together they can outperform heavy-model pipelines at a fraction of the cost and complexity.

---

## Approach

Three complementary signals for the LLM:

| Signal type | How extracted | What it reveals |
|---|---|---|
| Audio features as text | librosa (RMS, onset, spectral flux at 1Hz) | Energy, rhythmic activity, change points |
| Spectrogram images | librosa mel/chroma/tempo → PNG | Visual patterns at song/phrase boundaries |
| Video frames | ffmpeg frame extraction | Visual cues (band picking up instruments, walking off stage, applause) |

The LLM doesn't replace beat tracking — it reasons *about* the outputs of cheap DSP tools to make decisions that previously required domain-specific ML models.

---

## Experiments

### Phase 1 — Understand the Signal (Weeks 1–2)

**Exp 00: Feature Baseline Visualization** ← *starting now*
- Extract RMS, percussive RMS, onset strength, spectral centroid at 1-sec resolution using librosa
- Plot against known song and clip boundaries (from allin1 analysis of the 2025-10-30 recording)
- Goal: see which cheap features cleanly distinguish music vs. silence, and phrase boundaries vs. mid-phrase
- Output: `experiments/exp00_feature_baseline.py`

**Exp 01: Silence/Gap Detection Baseline**
- Threshold percussive RMS + onset strength to find inter-song gaps
- Compare against allin1 segment boundaries and song-splitting ground truth
- Goal: establish what rule-based detection alone can achieve before involving LLM
- Output: `experiments/exp01_gap_detection.py`

### Phase 2 — LLM with Audio Features as Text (Weeks 2–3)

**Exp 02: Text Feature Time Series → Claude**
- Format 1Hz feature vectors as a compact, timestamped text table
- Send to Claude in 10-minute chunks with overlap
- Prompt: "Here are audio features. Identify timestamps where you see evidence of a song ending or phrase boundary. Be conservative — only flag high-confidence boundaries."
- Evaluate: precision/recall vs. ground truth
- Output: `experiments/exp02_text_features_llm.py`

### Phase 3 — LLM with Spectrogram Images (Weeks 3–4)

**Exp 03: Mel Spectrogram Strips → Claude Vision**
- Generate mel spectrograms as wide PNG strips (5-minute window, scrolling with 1-minute overlap)
- Ask Claude: "This is a mel spectrogram. Identify timestamps of phrase and song boundaries. Describe what you see."
- Evaluate against ground truth; compare to Exp 02

**Exp 04: Chromagram and Tempogram → Claude Vision**
- Chromagram: reveals harmonic content changes (key changes, song transitions)
- Tempogram: reveals rhythmic stability (drops to noise during gaps, shifts at song starts)
- Test each type independently, then combined in a single image panel

Output: `experiments/exp03_spectrogram_llm.py`

### Phase 4 — LLM with Video Frames (Week 4)

**Exp 05: Video Frame Strips → Claude Vision**
- Sample 1 frame per 10 seconds; send a montage of 30 frames (~5 minutes)
- Ask: "These frames are from a music practice video. Identify which frames mark the beginning or end of a song. Describe visual cues."
- Compare to audio-only Exp 02/03
- Output: `experiments/exp05_video_frames_llm.py`

### Phase 5 — Hybrid Integration (Weeks 5–6)

**Exp 06: Hybrid Song Boundary Detection**
- Stage 1: Gap detection (Exp 01) narrows candidate windows to ±60s around likely boundaries
- Stage 2: Claude reviews mel spectrogram + video frames in the candidate window to confirm and refine
- Goal: fast (seconds of CPU + a few API calls), robust to bad audio
- Output: `experiments/exp06_hybrid_song_boundary.py`

**Exp 07: Hybrid Clip/Phrase Boundary Detection**
- Within a detected song, find phrase boundaries every 30–60s
- Approach: beat tracking (librosa) for beat times → chromagram showing phrase structure → Claude identifies which beat is the best cut point
- Goal: replace allin1's 6.5-hour downbeat detection with a fast equivalent
- Output: `experiments/exp07_hybrid_clip_boundary.py`

### Phase 6 — End-to-End Algorithm (Weeks 7–8)

**Exp 08: Full Pipeline as a New Algorithm Plugin**
- Implement `algorithms/llm_hybrid.py` for `process_video.py`
- Full flow: song detection → clip building → visual filter → render
- Benchmark vs. allin1: accuracy, runtime, API cost per hour of video
- Output: `algorithms/llm_hybrid.py`

### Phase 7 — Service (Weeks 9–14)

**Exp 09: Cost and Latency Profiling**
- Measure API cost and wall-clock time per hour of video across different Claude models (Haiku vs. Sonnet)
- Identify the cheapest model that maintains quality
- Estimate pricing for a public service

**Exp 10: Service Architecture Design**
- Async job queue (video upload → processing → download)
- Storage design for input videos and output clips
- Authentication and usage limits
- Simple web UI design

**Exp 11: Service Prototype**
- Minimal working service: upload video, get back clips.json + rendered clips
- Deploy to a cloud environment

---

## Success Criteria

| Metric | Target |
|---|---|
| Song boundary detection accuracy | ≥ 90% of boundaries within ±10s |
| Clip boundary quality | Subjectively comparable to allin1 output |
| Processing time (90-min video) | < 10 minutes end-to-end |
| API cost (90-min video) | < $0.50 |
| Dependencies | librosa, ffmpeg, claude CLI — nothing else |

---

## Ground Truth

The 2025-10-30 practice recording is the primary test case:
- allin1 analysis at `clip-splitting/` (downbeats, segments, BPM)
- 161 clips (30-60s) built from allin1 downbeats, filtered to 123
- Song-splitting ground truth in `song-splitting/`

---

## Log

| Date | Experiment | Result |
|------|------------|--------|
| 2026-07-14 | Exp 00: Feature baseline | Complete. Key finding: over the full 94-min recording, allin1's 256 segments are mostly intra-song structural boundaries (verse→chorus), which are *energetic* hits — percussive RMS and onset strength are higher at boundaries, not lower. Energy-drop-based gap detection needs song-level ground truth, not allin1 segment ground truth. Exp 01 will produce that. |
