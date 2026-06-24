# Song Splitting: Lessons Learned

Key takeaways and insights from the song splitting project.

## Technical Lessons

### 1. Direct Measurements Beat Complex Analysis

**What we learned:**
- "Count the beats" (onset rate) works better than "estimate and analyze tempo stability"
- Shorter signal processing chains = less accumulated error
- Simple features with clear physical meaning are easier to debug and tune

**Example:**
```python
# Complex, failed
tempo → variance → instability → gap detection

# Simple, worked
onset_count / window_duration → gap detection
```

### 2. Know Your Algorithm's Assumptions

**What we learned:**
- `librosa.beat.tempo()` is designed for full songs, not 10-second windows
- Using algorithms outside their intended domain leads to garbage results
- Read the docs and understand what input the algorithm expects

**Red flags we should have noticed earlier:**
- Tempo estimates of 300 BPM (physically unrealistic for most music)
- Wild swings in consecutive windows (40→250→80 BPM in 6 seconds)
- These are signs the algorithm isn't working, not that the music is weird

### 3. Visualization is Essential

**What we learned:**
- Can't debug audio analysis without visualization
- Especially important for Claude Code since it can see images
- Multi-panel plots showing all stages of analysis are invaluable

**Our visualization workflow:**
1. Run analysis script with `-v output.png`
2. Look at visualization to understand what's happening
3. Identify which stage is broken
4. Adjust and repeat

**Example insight from visualization:**
- Onset rate panel showed **dramatic** drops to ~0 at boundaries
- Immediately clear this was the killer feature
- Much more obvious than looking at numbers in JSON

See [Visualization Best Practices](../../best-practices/visualization.md).

### 4. Feature Engineering > Algorithm Tuning

**What we learned:**
- Getting the right features matters more than tuning parameters
- Onset rate was the breakthrough, not better tempo estimation
- When an approach isn't working, try different features before trying different algorithms

**Failed:** 30+ parameter combinations with tempo-based approach
**Succeeded:** First try with onset rate approach

### 5. Preprocessing Can Transform Hard Problems

**What we learned:**
- Harmonic-percussive source separation (HPSS) was crucial
- Extracting percussive component removes noise from melodic/harmonic content
- Good preprocessing makes downstream tasks easier

**Impact of HPSS:**
- Cleaner onset detection (only percussive hits)
- Percussive energy clearly drops during crowd-only gaps
- Crowd noise has minimal percussive content

```python
# This one line changed everything
y_harmonic, y_percussive = librosa.effects.hpss(y)
```

### 6. Combine Complementary Features

**What we learned:**
- Percussive energy alone: pretty good
- Onset rate alone: very good
- Both together: excellent

**Why combination works:**
- Onset rate: detects absence of rhythmic hits
- Percussive energy: detects absence of drum sounds
- Together: more robust to edge cases

**Equal weighting worked fine:**
```python
gap_score = 0.5 * energy_gap_score + 0.5 * onset_gap_score
```

No need for complex weighting schemes.

## Process Lessons

### 7. Iterate Quickly with Visualizations

**Our workflow:**
1. Implement approach
2. Run on test audio
3. Generate visualization
4. Look at visualization (Claude can see it!)
5. Identify problems
6. Adjust approach
7. Repeat

**Speed matters:**
- Fast iteration = more experiments in less time
- Visualization allows quick assessment without listening to 90 minutes of audio
- Test on same audio file for consistency

### 8. Document Failures, Not Just Successes

**What we learned:**
- Understanding *why* tempo-based approaches failed informed the successful approach
- Future projects can avoid the same mistakes
- Negative results are valuable information

**This wiki includes:**
- Detailed analysis of what didn't work ([failed-approaches.md](failed-approaches.md))
- Root causes, not just symptoms
- Lessons extracted from each failure

### 9. Know When to Pivot

**Timeline:**
1. Windowed tempo: clearly not working (wild noise)
2. Added percussive separation: marginal improvement
3. Tried tempogram: complete failure
4. **Pivoted to onset rate approach** ← critical decision
5. Immediate success

**How we knew to pivot:**
- Multiple different tempo approaches all failed
- Fundamental issue (tempo estimation on short windows) not fixable with tweaks
- Visualization showed the problem was pervasive

**Lesson:** Don't keep debugging a fundamentally broken approach.

### 10. Start with a Good Design Document

**What we did right:**
- Created `tempo_segmentation_design.md` before coding
- Documented the problem, constraints, and approach
- Outlined multiple alternatives (Approach A vs Approach B)

**What we did wrong:**
- Didn't validate tempo estimation on sample data before building full pipeline
- Should have tested core assumptions (can we get stable tempo in 10s windows?)

**Better approach:**
1. Design document
2. **Validate core assumptions with quick experiments**
3. Build full pipeline
4. Iterate

## Domain Lessons

### 11. Concert Audio Has Specific Characteristics

**What makes concert segmentation unique:**
- High background noise (can't use volume thresholds)
- Gaps are crowd-only (no music, but not silent)
- Long recordings (1-2 hours) with many segments

**Key insight:**
- Crowd noise is spectrally different from music but not quieter
- Crowd noise lacks percussive/rhythmic structure
- This is why onset rate works so well

### 12. Prefer Over-Splitting to Under-Splitting

**Why:**
- Easy to manually merge segments (just don't use some clips)
- Hard to manually split segments (requires finding the boundary)

**How we achieved this:**
- Lower threshold = more boundaries detected
- Can tune threshold based on tolerance for false positives

**Actual results:**
- 12 segments detected
- Most were correct song boundaries
- A few were questionable but acceptable
- No major song boundaries were missed

## Generalizable Insights

### 13. Audio Analysis Principles

**What worked for us will work for similar problems:**

1. **Harmonic-percussive separation** - Useful whenever rhythm matters
2. **Onset detection** - Fundamental for finding "events" in audio
3. **Sliding window analysis** - Standard approach for time-varying features
4. **Peak detection** - Good for finding discrete events in continuous signals

**Applicable to:**
- Beat detection
- Event segmentation
- Rhythm analysis
- Music information retrieval

### 14. The Power of Rate Features

**Onset rate** (events per unit time) was the killer feature.

**Other rate features to try:**
- Beat rate (beats per minute, but measured locally)
- Chord change rate
- Spectral flux rate (frequency content changes)
- Zero-crossing rate

**Why rates work:**
- Dimensionless (not affected by absolute volume)
- Intuitive interpretation
- Robust to certain types of noise

### 15. Complementary Features > Single "Perfect" Feature

**Instead of searching for one perfect feature:**
- Combine multiple decent features
- Each captures different aspects
- Robust to edge cases that break individual features

**Our combination:**
```python
gap_score = 0.5 * (1 - normalized_energy) + 0.5 * (1 - normalized_onset_rate)
```

Simple average works well when features are complementary.

## For Future Projects

### Checklist for Audio Analysis Projects

- [ ] Create design document with problem definition and constraints
- [ ] **Validate core assumptions with quick experiments**
- [ ] Start with simplest/most direct approach
- [ ] Implement visualization from the beginning
- [ ] Test on same audio file for consistent comparison
- [ ] Document what doesn't work, not just what does
- [ ] Know when to pivot vs when to tune
- [ ] Combine complementary features
- [ ] Use domain knowledge (e.g., HPSS for rhythm tasks)
- [ ] Iterate quickly with visualizations

### Questions to Ask

**Before starting:**
1. What is the most direct way to measure what I care about?
2. What are the fundamental assumptions of the algorithms I'm using?
3. What preprocessing might help?

**While iterating:**
1. Does the visualization make sense?
2. Am I debugging a symptom or fixing a root cause?
3. Should I pivot or keep tuning?

**After finishing:**
1. What worked and why?
2. What failed and why?
3. What would I do differently next time?

## Summary

**The most important lesson:**

> Simple, direct measurements based on clear physical principles beat complex analysis of indirect signals.

**The most important tool:**

> Visualization that shows all stages of analysis and lets you see what's happening.

**The most important skill:**

> Knowing when to pivot from a broken approach rather than endlessly debugging it.
