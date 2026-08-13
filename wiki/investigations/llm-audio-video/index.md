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

### Phase 2b — MCP Audio Analysis (Week 3)

**Exp 03: audio-analyzer-rs MCP Tool (direct LLM hearing)**
- Install `audio-analyzer-rs` MCP server (Homebrew, pure Rust, no Python deps)
- Let Claude call the MCP tools directly on the 95-min recording: section boundary detection, HPSS, onset density, tempo, rhythm
- Compare section boundaries to Exp 01/02 results — does native DSP + direct LLM reasoning beat our hand-rolled gap score?
- Key question: does the MCP `section_boundaries` tool find inter-song gaps or only intra-song structure (same issue as allin1)?
- If boundaries look wrong, can we feed the raw time-series features to Claude and get better reasoning than Exp 02's ASCII table?
- Output: `experiments/exp03_mcp_audio_analyzer.py` (or just a documented session if MCP tools make scripting unnecessary)
- Repo: https://github.com/JuzzyDee/audio-analyzer-rs

### Phase 3 — Video Frame Confirmation (Weeks 3–4)

> **Note:** Spectrogram/chromagram image experiments (originally Exp 04/05) superseded by audio-analyzer-rs MCP, which provides richer harmonic/spectral analysis faster and more token-efficiently than image strips. Skipping directly to video frames, which is the one signal the MCP approach cannot provide.

**Exp 04: Video Frame Strips → Claude Vision (targeted confirmation)**
- For each boundary identified in Exp 03, extract a frame strip covering the ±3-minute window (1 frame/10s → ~36 frames)
- Build a labeled grid montage (timestamp overlay on each frame)
- Ask Claude: "These frames are from a music practice video. Does this window show a song ending and a new song starting, or an intra-song breakdown/pause? Describe what you see."
- Special focus: resolve the ambiguous 60:44–65:43 quiet zone (is it a song boundary or a long breakdown?)
- Confirm/deny each boundary; refine timestamps using visual cues (instruments put down, people talking, walking off, new song count-in)
- Output: `experiments/exp04_video_frames.py`

### Phase 4 — Hybrid Integration (Weeks 4–5)

**Exp 05: Full Hybrid Song Boundary Algorithm**
- Stage 1: MCP full_analysis → LLM clusters section boundaries into candidate song boundaries
- Stage 2: For medium-confidence candidates, extract video frame strip → Claude vision confirm/deny
- Stage 3: Final song list with confidence-weighted timestamps
- Goal: fast (seconds of Rust + ~10 API calls), robust to bad audio, works on any genre
- Output: `experiments/exp05_hybrid_song_boundary.py`

**Exp 06: Hybrid Clip/Phrase Boundary Detection**
- Within a detected song, find phrase boundaries every 30–60s
- Approach: MCP rhythm analysis for beat/downbeat times → LLM picks best cut points given phrase structure
- Goal: replace allin1's 6.5-hour downbeat detection with a fast equivalent
- Output: `experiments/exp06_hybrid_clip_boundary.py`

### Phase 5 — End-to-End Algorithm (Weeks 5–6)

**Exp 07: Full Pipeline as a New Algorithm Plugin**
- Implement `algorithms/llm_hybrid.py` for `process_video.py`
- Full flow: song detection → clip building → visual filter → render
- Benchmark vs. allin1: accuracy, runtime, API cost per hour of video
- Output: `algorithms/llm_hybrid.py`

### Phase 7 — Service (Weeks 10–15)

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
| 2026-07-14 | Exp 00: Feature baseline | Complete. Key finding: allin1's 256 segments are mostly intra-song structural transitions (verse→chorus hits), not silence gaps. Energy features are higher at those boundaries, not lower. Song-level gap detection needs coarser ground truth. |
| 2026-07-14 | Exp 01: Gap detection | Complete. Gap score detects silences in <1s from saved features. Best params → 12 segments (4–15 min). Core problem: no ground truth, so parameter sensitivity is opaque. Motivates Exp 02. |
| 2026-07-14 | Exp 02: LLM text features | Complete. Sent 570-row ASCII gap score table (~4200 tokens) to claude-sonnet-4-6. Found 2 high-confidence boundaries (11:30, 33:30) → 3 songs. Key insight: LLM correctly diagnosed that the second half (34:10–95:00) has consistently elevated gap scores not because of song breaks, but because the playing style changed — looser, less distinct stops. Rule-based gap detection would have over-split this section. **Gap score text alone is insufficient for the second half; video frames needed to confirm visual song boundaries.** |
| 2026-08-03 | Exp 04: Video frame confirmation | Complete. All 6 MCP-identified boundaries confirmed visually (6/6). The ambiguous 60:44–65:43 quiet zone confirmed as a genuine song boundary at 62:10. Final result: **7 songs** with timestamps refined by visual cues (e.g., 35:40→35:20, 45:59→45:39, 55:03→54:53, 69:54→69:34, new boundary at 62:10). Visual evidence: musicians shifting posture, lowering instruments, moving between songs. |
| 2026-08-03 | Exp 03: audio-analyzer-rs MCP | Complete. Installed audio-analyzer-rs MCP server (pure Rust, Symphonia decoder). Note: MOV unsupported — ffmpeg extraction to FLAC required first. Full analysis of 95 min completed in 295s. MCP section boundaries: 163 detected (intra-song, same class as allin1). **Approach: sent all 163 boundaries to LLM for song-level clustering.** Result: 5 boundaries → 6 songs. Key wins over Exp 02: (1) harmonic cascade signal (consecutive harmonic-only boundaries = key change = new song) pinned boundary at 10:55 vs 11:30; (2) **confidence-1.00 boundary at 45:59 (max in dataset) + longest quiet stretch (7 min) revealed a song boundary completely invisible to gap score alone**; (3) 55:03 and 69:54 boundaries found in second half where Exp 02 was blind. MCP onset strength time-series confirmed all 5 boundaries via quiet-zone analysis. Open question: 5-min quiet at 60:44–65:43 may be additional boundary or long breakdown. |
| 2026-08-05 | Exp 05: Noodling trim — MCP density analysis | Complete. Ran `full_analysis` (low res, 300s windows) around the start and end of all 6 songs to find true playing boundaries. **Key finding: MCP col 11 (`onset_density` /sec) is a near-perfect discriminator** — full-band playing = 8–18/sec (sustained), noodling/silence = 0–5/sec. librosa `onset_strength` is useless here (noodling and playing are identical, mean ~0.10–0.13 both). Applied 5s padding around the detected threshold crossing. Results: Song 1 saves 94s (end trim only); Song 2 saves 434s (14+ min silence gap removed); Songs 3–6 save 98–151s each (noodling at start and end). Rendered to `song-splitting/trimmed/`. False positive region 62:10–69:34 dropped entirely (Song 6 confirmed as noodling/talking in Exp 04). |
