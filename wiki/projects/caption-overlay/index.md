# Caption Overlay Project

**Independent workstream** — not part of the song/clip splitting pipeline. This project takes already-cut video clips and adds a burned-in text caption (song title + optional emoji) in the style of Instagram/TikTok story captions, for social posting.

## Problem Statement

**Goal:** Given a short video clip and a caption string (e.g. a song title), overlay a rounded white "pill" caption near the top of the frame, matching the visual style used for casual social clips — bold black text, optional trailing emoji, soft drop shadow.

Reference style: white rounded rectangle, bold sans-serif text, single emoji suffix, positioned in the top ~10% of a vertical (9:16) frame.

## Solution

Two-stage render:

1. **Render the caption as a transparent PNG** with Pillow — rounded-rectangle pill (white fill, soft drop shadow), bold Arial text, and an emoji glyph pulled from Apple Color Emoji and rescaled to match the text size.
2. **Composite onto the video with ffmpeg** using the `overlay` filter, positioned centered horizontally with a top margin proportional to frame height. The caption can be shown for the whole clip or a `--start`/`--duration` window (e.g. only the first 3 seconds).

See [approach.md](approach.md) for implementation details, including the Apple Color Emoji fixed-pixel-size quirk.

## Files

- `caption-overlay/caption_overlay.py` — render + composite script

## Usage

```bash
# Add a caption to a clip
uv run python caption-overlay/caption_overlay.py add \
  --video clip.mp4 --text "Keep on Riding" --emoji "🐎" \
  --output clip_captioned.mp4

# Preview just the caption pill (no video needed)
uv run python caption-overlay/caption_overlay.py preview \
  --text "Keep on Riding" --emoji "🐎" --output preview.png
```

## Learn More

- [Technical Approach](approach.md)
- [Lessons Learned](lessons-learned.md)
