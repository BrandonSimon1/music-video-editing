"""Overlay a rounded-pill text caption (with optional emoji) onto a video clip.

Renders the caption as a transparent PNG with Pillow, then composites it onto
the video with ffmpeg. Mirrors the reference style: a white rounded pill near
the top of the frame, bold black text, optional trailing emoji.

Usage:
    uv run python caption-overlay/caption_overlay.py add \\
        --video clip.mp4 --text "Keep on Riding" --emoji "🐎" \\
        --output clip_captioned.mp4
"""

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
EMOJI_FONT = "/System/Library/Fonts/Apple Color Emoji.ttc"
# Apple Color Emoji only rasterizes at these fixed pixel sizes; render at the
# closest supported size and downscale to fit the caption's font size.
EMOJI_SUPPORTED_SIZES = [160, 96, 64, 48, 32, 20]

PADDING_X = 36
PADDING_Y = 20
CORNER_RADIUS_RATIO = 0.5  # of box height -> full pill
SHADOW_BLUR_MARGIN = 16
SHADOW_OFFSET = (0, 4)
SHADOW_OPACITY = 60  # 0-255


def _load_emoji(emoji: str, target_size: int) -> Image.Image:
    render_size = min((s for s in EMOJI_SUPPORTED_SIZES if s >= target_size),
                       default=EMOJI_SUPPORTED_SIZES[0])
    font = ImageFont.truetype(EMOJI_FONT, render_size)
    tmp = Image.new("RGBA", (render_size * 2, render_size * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tmp)
    draw.text((0, 0), emoji, font=font, embedded_color=True)
    bbox = tmp.getbbox()
    if bbox is None:
        return Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    glyph = tmp.crop(bbox)
    scale = target_size / max(glyph.size)
    new_size = (max(1, int(glyph.width * scale)), max(1, int(glyph.height * scale)))
    return glyph.resize(new_size, Image.LANCZOS)


def render_caption_image(text: str, emoji: str | None, font_size: int = 44) -> Image.Image:
    """Render a white rounded-pill caption with bold text and optional emoji.

    Returns an RGBA image sized to the pill plus shadow margin, ready to be
    composited onto a video frame.
    """
    font = ImageFont.truetype(FONT_BOLD, font_size)

    scratch = Image.new("RGBA", (10, 10))
    draw = ImageDraw.Draw(scratch)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    emoji_img = None
    emoji_gap = 0
    emoji_size = int(font_size * 1.0)
    if emoji:
        emoji_img = _load_emoji(emoji, emoji_size)
        emoji_gap = int(font_size * 0.35)

    content_w = text_w + (emoji_gap + emoji_img.width if emoji_img else 0)
    content_h = max(text_h, emoji_img.height if emoji_img else 0)

    box_w = content_w + 2 * PADDING_X
    box_h = content_h + 2 * PADDING_Y
    radius = int(box_h * CORNER_RADIUS_RATIO)

    canvas_w = box_w + 2 * SHADOW_BLUR_MARGIN
    canvas_h = box_h + 2 * SHADOW_BLUR_MARGIN + SHADOW_OFFSET[1]
    img = Image.new("RGBA", (int(canvas_w), int(canvas_h)), (0, 0, 0, 0))

    # Soft drop shadow.
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_box = (
        SHADOW_BLUR_MARGIN + SHADOW_OFFSET[0],
        SHADOW_BLUR_MARGIN + SHADOW_OFFSET[1],
        SHADOW_BLUR_MARGIN + SHADOW_OFFSET[0] + box_w,
        SHADOW_BLUR_MARGIN + SHADOW_OFFSET[1] + box_h,
    )
    shadow_draw.rounded_rectangle(shadow_box, radius=radius, fill=(0, 0, 0, SHADOW_OPACITY))
    from PIL import ImageFilter
    shadow = shadow.filter(ImageFilter.GaussianBlur(SHADOW_BLUR_MARGIN / 3))
    img = Image.alpha_composite(img, shadow)

    # White pill.
    draw = ImageDraw.Draw(img)
    pill_box = (SHADOW_BLUR_MARGIN, SHADOW_BLUR_MARGIN, SHADOW_BLUR_MARGIN + box_w, SHADOW_BLUR_MARGIN + box_h)
    draw.rounded_rectangle(pill_box, radius=radius, fill=(255, 255, 255, 255))

    # Text, vertically centered.
    text_x = SHADOW_BLUR_MARGIN + PADDING_X
    text_y = SHADOW_BLUR_MARGIN + (box_h - text_h) / 2 - text_bbox[1]
    draw.text((text_x, text_y), text, font=font, fill=(20, 20, 20, 255))

    if emoji_img:
        emoji_x = text_x + text_w + emoji_gap
        emoji_y = SHADOW_BLUR_MARGIN + (box_h - emoji_img.height) / 2
        img.alpha_composite(emoji_img, (int(emoji_x), int(emoji_y)))

    return img


def _probe_video(video_path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,duration",
         "-of", "json", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)["streams"][0]


def add_caption_to_video(
    video_path: Path,
    text: str,
    emoji: str | None,
    output_path: Path,
    start: float = 0.0,
    duration: float | None = None,
    top_margin_ratio: float = 0.06,
    font_size: int = 44,
) -> None:
    info = _probe_video(video_path)
    video_w = int(info["width"])

    # Scale font/pill size relative to video width so captions look consistent
    # across differently-sized source clips (reference target: 1080px wide).
    scale = video_w / 1080
    caption_img = render_caption_image(text, emoji, font_size=int(font_size * scale))

    overlay_path = output_path.with_suffix(".caption.png")
    caption_img.save(overlay_path)

    top_margin = int(int(info["height"]) * top_margin_ratio)
    enable_expr = f"between(t,{start},{start + duration})" if duration else "1"

    filter_complex = (
        f"[1:v]format=rgba[ovl];"
        f"[0:v][ovl]overlay=x=(W-w)/2:y={top_margin}:enable='{enable_expr}'"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(overlay_path),
        "-filter_complex", filter_complex,
        "-c:a", "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    overlay_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="Add a caption overlay to a video")
    add.add_argument("--video", required=True, type=Path)
    add.add_argument("--text", required=True)
    add.add_argument("--emoji", default=None)
    add.add_argument("--output", required=True, type=Path)
    add.add_argument("--start", type=float, default=0.0)
    add.add_argument("--duration", type=float, default=None,
                      help="Seconds the caption stays visible; default is the whole clip")
    add.add_argument("--font-size", type=int, default=44)

    preview = sub.add_parser("preview", help="Render just the caption PNG for inspection")
    preview.add_argument("--text", required=True)
    preview.add_argument("--emoji", default=None)
    preview.add_argument("--output", required=True, type=Path)
    preview.add_argument("--font-size", type=int, default=44)

    args = parser.parse_args()

    if args.command == "add":
        add_caption_to_video(
            args.video, args.text, args.emoji, args.output,
            start=args.start, duration=args.duration, font_size=args.font_size,
        )
        print(f"Wrote {args.output}")
    elif args.command == "preview":
        img = render_caption_image(args.text, args.emoji, font_size=args.font_size)
        img.save(args.output)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
