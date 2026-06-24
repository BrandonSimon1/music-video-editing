# Visualization for Audio Analysis

Creating visualizations is essential for audio analysis, especially when working with Claude Code.

## Why Visualization Matters

### 1. Claude Code Can See Images

**Critical capability:**
- Claude Code can read image files via the Read tool
- You can ask "what do you think of visualization.png?"
- Claude can analyze plots, identify problems, suggest fixes

**Example interaction:**
```
User: Check out visualization-3.png, how did we do?
Claude: [reads image] Much better! The instability score now has clearer peaks...
```

This is **massively faster** than:
- Describing what you see in text
- Copy-pasting numbers
- Manually interpreting results

### 2. Can't Debug Audio Without Seeing It

**Problem:** Audio is temporal and complex
- Can't easily inspect hours of audio by listening
- Numbers in JSON are hard to interpret
- Need to see patterns over time

**Solution:** Multi-panel visualizations showing all stages of analysis

### 3. Iterate Quickly

**Workflow:**
1. Run analysis with `-v visualization.png`
2. Ask Claude to look at the visualization
3. Identify problems immediately
4. Adjust and re-run
5. Repeat

**Speed:** Minutes per iteration instead of hours

## Visualization Best Practices

### Multi-Panel Layout

**Show all stages of your pipeline:**

```python
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# Panel 1: Raw signal or fundamental features
ax1.plot(times, onset_strength)

# Panel 2: Intermediate feature 1
ax2.plot(times, percussive_energy)

# Panel 3: Intermediate feature 2
ax3.plot(times, onset_rate)

# Panel 4: Final detection score with threshold
ax4.plot(times, gap_score)
ax4.axhline(threshold, color='green', linestyle=':')
```

**Why multiple panels:**
- See which stage is broken
- Understand how features combine
- Validate intermediate steps

### Mark Detected Events

**Always show your detections on the visualization:**

```python
# Mark boundaries on all panels
for b in boundaries[1:-1]:  # Skip start/end
    ax.axvline(b, color='red', linestyle='--', alpha=0.6)
```

**Why:**
- Immediately see if detections make sense
- Correlate detections with signal features
- Identify false positives and false negatives

### Use Meaningful Axes

**Good:**
```python
ax.set_ylabel('Onset Rate (onsets/sec)')
ax.set_xlabel('Time (seconds)')
ax.set_title('Concert Audio Segmentation via Percussive Energy & Onset Rate Analysis')
```

**Bad:**
```python
ax.set_ylabel('Score')  # What score?
ax.set_xlabel('Index')   # Index into what?
```

**Why:**
- Clear labels help Claude understand the visualization
- Easy to refer to specific panels in discussion
- Makes debugging faster

### Choose Colors Wisely

**Convention:**
```python
# Raw signals: gray
ax1.plot(times, signal, color='gray', alpha=0.7)

# Features: distinctive colors
ax2.plot(times, feature1, color='purple')  # Energy
ax3.plot(times, feature2, color='blue')    # Rate

# Final score: warm color
ax4.plot(times, score, color='orange')

# Detections: red
ax.axvline(boundary, color='red', linestyle='--')

# Thresholds: green
ax.axhline(threshold, color='green', linestyle=':')
```

**Why:**
- Consistent colors across experiments
- Easy to distinguish elements
- Detections stand out in red

### Include Grid Lines

```python
ax.grid(True, alpha=0.3)
```

**Why:**
- Easier to read exact values
- Helps align features across panels
- Minimal visual clutter with low alpha

### Save at Good Resolution

```python
plt.savefig(output_file, dpi=150, bbox_inches='tight')
```

**Why:**
- `dpi=150`: readable without being huge
- `bbox_inches='tight'`: no wasted whitespace
- Claude can read the text and see details

## Song Splitting Example

Our final visualization has 4 panels:

```python
def visualize_analysis(
    onset_env,
    percussive_energy,
    onset_rate,
    time_points,
    gap_score,
    boundaries,
    sr,
    output_file=None
):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Panel 1: Onset Strength (fundamental feature)
    ax1.plot(onset_times, onset_env, color='gray', alpha=0.7)
    ax1.set_ylabel('Onset Strength')

    # Panel 2: Percussive Energy (intermediate feature)
    ax2.plot(time_points, percussive_energy, color='purple', linewidth=2)
    ax2.set_ylabel('Percussive Energy (RMS)')

    # Panel 3: Onset Rate (intermediate feature)
    ax3.plot(time_points, onset_rate, color='blue', linewidth=2)
    ax3.set_ylabel('Onset Rate (onsets/sec)')

    # Panel 4: Gap Score (final detection signal)
    ax4.plot(time_points, gap_score, color='orange', linewidth=2)
    ax4.axhline(0.65, color='green', linestyle=':', label='Threshold')
    ax4.set_ylabel('Gap Score')
    ax4.set_xlabel('Time (seconds)')

    # Mark boundaries on all panels
    for ax in axes:
        for b in boundaries[1:-1]:
            ax.axvline(b, color='red', linestyle='--', alpha=0.6)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
```

### What This Revealed

**Onset rate panel (panel 3) showed:**
- Dramatic drops to ~0 during gaps
- Clear 2-3 onsets/sec during songs
- This was THE killer feature

**Gap score panel (panel 4) showed:**
- Clear peaks above threshold at boundaries
- Smooth signal (thanks to median filtering)
- Why each boundary was detected

**We could see this immediately** without listening to 90 minutes of audio.

## Interactive Development with Claude

### Effective Questions to Ask

**Good:**
```
"Look at visualization-3.png - does the onset rate suggest anything we should change?"
```

**Better:**
```
"Check out the new visualization. The onset rate (panel 3) shows sharp drops at the
boundaries. Does this suggest we're on the right track?"
```

**Best:**
```
"I ran it with the new approach. See visualization-5.png. We detected 12 segments.
Looking at the onset rate (panel 3), do the boundaries make sense? Are there any
obvious peaks in the gap score that we're missing?"
```

### What Claude Can See

Claude Code can:
- ✅ Read axis labels and titles
- ✅ See overall patterns and trends
- ✅ Identify peaks, valleys, discontinuities
- ✅ Compare patterns across panels
- ✅ See marked boundaries and thresholds
- ✅ Assess whether detections align with features

Claude Code cannot:
- ❌ Read tiny text (use reasonable font sizes)
- ❌ See very subtle differences (make important features obvious)
- ❌ Analyze extremely complex plots (keep it simple)

## Visualization Types for Different Tasks

### Time Series Analysis
```python
plt.plot(times, values)
plt.xlabel('Time (seconds)')
plt.ylabel('Feature Value')
```
**Use for:** Any time-varying signal

### Spectrograms
```python
librosa.display.specshow(
    librosa.amplitude_to_db(S, ref=np.max),
    y_axis='hz',
    x_axis='time',
    sr=sr
)
```
**Use for:** Frequency content over time

### Multi-Feature Comparison
```python
fig, axes = plt.subplots(N, 1, figsize=(14, 3*N), sharex=True)
```
**Use for:** Comparing multiple signals aligned in time

### Distribution Analysis
```python
plt.hist(values, bins=50)
plt.xlabel('Feature Value')
plt.ylabel('Count')
```
**Use for:** Understanding feature statistics

## Common Pitfalls

### ❌ No Visualization
```python
# Just save JSON
with open('output.json', 'w') as f:
    json.dump(results, f)
```

**Problem:** Can't see what's happening

### ❌ Visualization Without Context
```python
plt.plot(values)
plt.savefig('output.png')
```

**Problem:** No labels, no title, unclear what it shows

### ❌ Too Many Subplots
```python
fig, axes = plt.subplots(10, 1, figsize=(14, 30))
```

**Problem:** Hard to see relationships, overwhelming

### ✅ Good Visualization
```python
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

for i, (feature, label, color) in enumerate(features):
    axes[i].plot(times, feature, color=color, linewidth=2, label=label)
    axes[i].set_ylabel(label)
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

    # Mark detections
    for b in boundaries:
        axes[i].axvline(b, color='red', linestyle='--', alpha=0.6)

axes[-1].set_xlabel('Time (seconds)')
plt.suptitle('Audio Segmentation Analysis')
plt.tight_layout()
plt.savefig('output.png', dpi=150, bbox_inches='tight')
```

## Template for New Projects

```python
def visualize_analysis(
    raw_signal,
    feature1,
    feature2,
    detection_score,
    time_points,
    detections,
    threshold,
    output_file
):
    """
    Visualize all stages of analysis.

    Args:
        raw_signal: Original signal or fundamental feature
        feature1, feature2: Intermediate features
        detection_score: Final detection metric
        time_points: Time axis for features
        detections: List of detected event times
        threshold: Detection threshold
        output_file: Path to save figure
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Panel 1: Raw signal
    ax1 = axes[0]
    ax1.plot(time_points, raw_signal, color='gray', alpha=0.7)
    ax1.set_ylabel('Raw Signal')
    ax1.set_title('Analysis Pipeline Visualization')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Feature 1
    ax2 = axes[1]
    ax2.plot(time_points, feature1, color='purple', linewidth=2, label='Feature 1')
    ax2.set_ylabel('Feature 1')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Feature 2
    ax3 = axes[2]
    ax3.plot(time_points, feature2, color='blue', linewidth=2, label='Feature 2')
    ax3.set_ylabel('Feature 2')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Detection score
    ax4 = axes[3]
    ax4.plot(time_points, detection_score, color='orange', linewidth=2)
    ax4.axhline(threshold, color='green', linestyle=':', label='Threshold')
    ax4.set_ylabel('Detection Score')
    ax4.set_xlabel('Time (seconds)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Mark detections on all panels
    for ax in axes:
        for d in detections:
            ax.axvline(d, color='red', linestyle='--', alpha=0.6,
                      label='Detection' if d == detections[0] and ax == ax4 else '')

    if len(detections) > 0:
        ax4.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {output_file}")
```

## Summary

**Key principles:**
1. **Always create visualizations** - Essential for debugging
2. **Multi-panel layouts** - Show all stages of analysis
3. **Mark detections** - Show results on top of signals
4. **Clear labels** - Help Claude understand what it's seeing
5. **Consistent style** - Makes comparisons easier
6. **Ask Claude to look** - Leverage its ability to see images

**Remember:**
> A good visualization lets you understand in 10 seconds what would take 10 minutes to figure out from numbers.

For Claude Code, visualization isn't optional—it's the primary way to collaborate on signal analysis.
