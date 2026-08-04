# Music Video Editing Wiki

Welcome to the music video editing project wiki. This wiki documents various music analysis and processing projects, including approaches tried, lessons learned, and best practices.

## Projects

### Splitting workstream

Turning long recordings into short clips.

- [Song Splitting](projects/song-splitting/index.md) - Automatic segmentation of concert recordings into individual songs
- [Clip Splitting](projects/clip-splitting/index.md) - Beat-aligned extraction of short clips from music videos

### Captioning workstream

A separate, independent workstream: taking already-cut clips and preparing them for posting.

- [Caption Overlay](projects/caption-overlay/index.md) - Burned-in text caption (song title + emoji) on video clips, in the style of social story captions

## Investigations

Active research threads exploring new approaches:

- [LLM-Augmented Audio/Video Analysis](investigations/llm-audio-video/index.md) — replacing allin1 with fast LLM-driven analysis; path to a service
- [Obsidian-Based Clip Approval](investigations/obsidian-clip-approval/index.md) — human review/approval of clips via the vault's ticket-style approval pattern, before captioning and upload

## Best Practices

- [Visualization for Analysis](best-practices/visualization.md) - Creating visualizations to inspect and validate analysis results
- [Audio Analysis Tips](best-practices/audio-analysis.md) - General tips and techniques for audio analysis

## Structure

Each project folder contains:
- **index.md** - Project overview and quick start
- **approach.md** - The working solution with technical details
- **failed-approaches.md** - What didn't work and why
- **lessons-learned.md** - Key takeaways and insights

## General Workflow

1. **Design** - Research and plan the approach
2. **Implement** - Write the analysis script
3. **Visualize** - Create visualizations to inspect results
4. **Iterate** - Refine based on visual feedback
5. **Validate** - Test on real data
6. **Document** - Record what worked and what didn't
