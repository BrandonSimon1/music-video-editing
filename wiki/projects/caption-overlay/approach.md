# Caption Overlay: Technical Approach

## Overview

`caption-overlay/caption_overlay.py` renders a caption as a standalone transparent PNG, then hands compositing off to ffmpeg. Splitting it this way keeps text layout (easy to iterate on with Pillow) separate from video encoding (best left to ffmpeg's `overlay` filter, which is fast and preserves audio via `-c:a copy`).

## Rendering the caption pill (Pillow)

`render_caption_image(text, emoji, font_size)`:

1. Measure the text with `ImageDraw.textbbox` using bold Arial (`/System/Library/Fonts/Supplemental/Arial Bold.ttf`).
2. Size a rounded rectangle ("pill") around the text with fixed padding (`PADDING_X=36`, `PADDING_Y=20` at the reference font size) and full corner radius (`CORNER_RADIUS_RATIO=0.5` of box height → pill ends are semicircles).
3. Draw a soft drop shadow first (a blurred, offset, semi-transparent rounded rectangle via `ImageFilter.GaussianBlur`), then the white pill on top, then the text.
4. If an emoji is given, append it after the text with a small gap.
5. Return an RGBA image sized to the pill plus shadow margin — small, and positioned later by ffmpeg, not baked into full-frame coordinates.

## Emoji rendering: the Apple Color Emoji quirk

Pillow can render color emoji from `/System/Library/Fonts/Apple Color Emoji.ttc` (`embedded_color=True`), **but the font only rasterizes at a fixed set of pixel sizes** — empirically `160, 96, 64, 48, 32, 20`. Requesting any other size raises `OSError: invalid pixel size`.

Workaround in `_load_emoji`: render at the smallest supported size ≥ the target size, crop to the glyph's actual bounding box (`Image.getbbox()`), then downscale with `Image.LANCZOS` to the exact target size. This keeps emoji visually matched to the text's cap-height regardless of the caption's font size.

## Compositing onto video (ffmpeg)

`add_caption_to_video`:

1. `ffprobe` the source clip for width/height (used to scale the caption proportionally — reference sizing is tuned for 1080px-wide frames, so `scale = video_w / 1080` adjusts font size for other resolutions).
2. Save the rendered caption PNG next to the output as a `.caption.png` sidecar (deleted after the run).
3. Run a single ffmpeg command:
   ```
   ffmpeg -i video.mp4 -i caption.png \
     -filter_complex "[1:v]format=rgba[ovl];[0:v][ovl]overlay=x=(W-w)/2:y=<top_margin>:enable='between(t,start,start+duration)'" \
     -c:a copy output.mp4
   ```
   - `x=(W-w)/2` centers the pill horizontally.
   - `y=<top_margin>` is `top_margin_ratio * frame_height` (default 6%), matching the reference image's placement below the phone's status-bar-safe area.
   - `enable='between(t,...)'` makes the caption appear only in the given time window when `--duration` is passed; otherwise it's shown for the whole clip.
   - `-c:a copy` avoids re-encoding audio.

## Why not moviepy?

moviepy is already a project dependency and can do text+image compositing, but its `TextClip` relies on ImageMagick for text rendering, which isn't part of this repo's existing toolchain (ffmpeg + Pillow already are, via [[clip-splitting]]'s rendering stage and PNG-based visualizations elsewhere in the repo). Rendering the caption as a Pillow image and compositing with ffmpeg's `overlay` filter reuses tools already proven in the codebase and keeps the caption's visual design fully controllable in Python rather than through ImageMagick's text layout.
