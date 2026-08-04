#!/usr/bin/env python3
"""
Exp 04: Video Frame Confirmation of Song Boundaries.

For each candidate song boundary from Exp 03 (MCP boundary clustering), extract
a frame strip covering the ±window_minutes around the boundary, build a labeled
grid montage, and ask Claude vision to confirm or deny the boundary.

Also handles arbitrary time windows (e.g., to resolve ambiguous quiet zones).

Algorithm:
  1. Load candidate boundaries from exp03_result.json (or accept --boundaries)
  2. For each boundary, run ffmpeg to extract 1 frame every frame_interval seconds
     over a window of ±window_minutes centered on the boundary
  3. Stamp each frame with its timestamp using ffmpeg drawtext
  4. Tile frames into a grid montage (PIL)
  5. Base64-encode montage, send to Claude via stream-json
  6. Parse Claude's response: confirmed / denied / uncertain + refined timestamp

Usage:
  uv run experiments/exp04_video_frames.py <video_file> [options]

  --boundaries-json PATH   exp03_result.json (default: experiments/exp03_result.json)
  --boundaries MM:SS,...   Override: comma-separated boundary timestamps to check
  --extra-windows MM:SS,...  Extra windows to check (e.g., the 60:44-65:43 quiet zone)
  --window-minutes FLOAT   Half-width of window to extract around each boundary (default: 3)
  --frame-interval INT     Seconds between extracted frames (default: 10)
  --model MODEL            Claude model (default: claude-sonnet-4-6)
  --dry-run                Extract frames and build montage but don't call Claude
  -o DIR                   Save montage PNGs to this directory
  --save-json PATH         Save results JSON
"""

import argparse
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def parse_time(s: str) -> float:
    """Parse MM:SS or SS into seconds."""
    s = s.strip()
    if ":" in s:
        parts = s.split(":")
        return int(parts[0]) * 60 + float(parts[1])
    return float(s)


def extract_frames(
    video_file: str,
    center_seconds: float,
    window_minutes: float,
    frame_interval: int,
    out_dir: Path,
) -> list[tuple[float, Path]]:
    """
    Extract frames at frame_interval spacing over [center - window, center + window].
    Returns list of (timestamp_seconds, frame_path).
    """
    start = max(0.0, center_seconds - window_minutes * 60)
    # Get video duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_file],
        capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip()) if probe.stdout.strip() else 1e9
    end = min(duration, center_seconds + window_minutes * 60)

    frames = []
    t = start
    while t <= end:
        frame_path = out_dir / f"frame_{int(t):05d}.jpg"
        result = subprocess.run(
            ["ffmpeg", "-ss", str(t), "-i", video_file,
             "-frames:v", "1", "-q:v", "3",
             str(frame_path), "-y"],
            capture_output=True,
        )
        if result.returncode == 0 and frame_path.exists():
            frames.append((t, frame_path))
        t += frame_interval

    return frames


def add_timestamp_label(img: "Image.Image", label: str, is_boundary: bool = False) -> "Image.Image":
    """Add a timestamp label at the bottom of a PIL image."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=max(12, h // 12))
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    padding = 4
    x = (w - text_w) // 2
    y = h - text_h - padding * 2

    bg_color = (200, 50, 50) if is_boundary else (0, 0, 0)
    draw.rectangle([x - padding, y - padding, x + text_w + padding, y + text_h + padding],
                   fill=bg_color)
    draw.text((x, y), label, fill=(255, 255, 255), font=font)
    return img


def build_montage(
    frames: list[tuple[float, Path]],
    center_seconds: float,
    boundary_label: str,
    cols: int = 6,
) -> "Image.Image":
    """Build a grid montage from extracted frame paths."""
    thumb_w, thumb_h = 320, 180
    rows = (len(frames) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb_w, rows * thumb_h), color=(20, 20, 20))

    for idx, (t, path) in enumerate(frames):
        try:
            img = Image.open(path).convert("RGB").resize((thumb_w, thumb_h))
        except Exception:
            img = Image.new("RGB", (thumb_w, thumb_h), color=(50, 50, 50))

        is_boundary = abs(t - center_seconds) < 30
        label = fmt_time(t)
        if is_boundary:
            label += " ◀"
        img = add_timestamp_label(img, label, is_boundary=is_boundary)

        row, col = divmod(idx, cols)
        canvas.paste(img, (col * thumb_w, row * thumb_h))

    return canvas


def call_claude_vision(montage_path: Path, prompt: str, model: str) -> str:
    """Send a montage image to Claude vision via stream-json."""
    with open(montage_path, "rb") as f:
        img_b64 = base64.standard_b64encode(f.read()).decode()

    msg_json = json.dumps({
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        },
    })

    cmd = [
        "claude", "--model", model, "-p",
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--verbose",
    ]
    result = subprocess.run(cmd, input=msg_json, capture_output=True, text=True, timeout=120)

    for line in result.stdout.splitlines():
        try:
            obj = json.loads(line)
            if obj.get("type") == "assistant":
                for block in obj.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        return block["text"].strip()
        except (json.JSONDecodeError, KeyError):
            continue

    if result.stderr:
        print(f"stderr: {result.stderr[:300]}", file=sys.stderr)
    return ""


def build_prompt(boundary_label: str, center_seconds: float, window_minutes: float) -> str:
    return f"""\
This is a grid of video frames from a music practice session recording.
Each frame is labeled with its timestamp (MM:SS). Frames marked with ◀ are near the candidate boundary at {boundary_label} (±30 seconds).

The frames cover approximately {fmt_time(max(0, center_seconds - window_minutes*60))} to {fmt_time(center_seconds + window_minutes*60)} (a {int(window_minutes*2)}-minute window).

I believe there may be a song boundary (one song ending, a gap or reset, a new song starting) somewhere near {boundary_label}.

Please analyze the frames and answer:

1. **Is there a song boundary here?** (Yes — confirmed / No — this is an intra-song pause or breakdown / Uncertain)
2. **If yes, at approximately what timestamp does the song end / new song begin?** Be as specific as the frames allow.
3. **What visual evidence supports your answer?** (e.g., musicians putting down instruments, talking, moving around, picking up instruments again, count-in gestures, new song posture)
4. **Anything else notable?** (e.g., applause, tuning, audio checks, someone leaving/entering frame)

Respond with ONLY this JSON (no markdown):
{{
  "verdict": "confirmed|denied|uncertain",
  "refined_timestamp": "MM:SS or null",
  "evidence": "what you saw",
  "notes": "anything else notable"
}}"""


def process_boundary(
    boundary_label: str,
    center_seconds: float,
    video_file: str,
    window_minutes: float,
    frame_interval: int,
    model: str,
    dry_run: bool,
    output_dir: Path | None,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Checking boundary: {boundary_label} ({center_seconds:.0f}s)")
    print(f"  Window: ±{window_minutes} min, frame every {frame_interval}s")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        frames = extract_frames(video_file, center_seconds, window_minutes, frame_interval, tmp_path)
        print(f"  Extracted {len(frames)} frames")

        if not frames:
            print("  No frames extracted — skipping")
            return {"boundary": boundary_label, "verdict": "error", "error": "no frames extracted"}

        if not PIL_AVAILABLE:
            print("  PIL not available — cannot build montage")
            return {"boundary": boundary_label, "verdict": "error", "error": "PIL not available"}

        montage = build_montage(frames, center_seconds, boundary_label)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_label = boundary_label.replace(":", "-")
            montage_path = output_dir / f"boundary_{safe_label}.jpg"
        else:
            montage_path = tmp_path / "montage.jpg"

        montage.save(str(montage_path), "JPEG", quality=85)
        print(f"  Montage: {montage.size[0]}x{montage.size[1]}px → {montage_path}")

        if dry_run:
            print("  [dry-run] Skipping Claude call")
            return {"boundary": boundary_label, "verdict": "dry-run", "montage": str(montage_path)}

        prompt = build_prompt(boundary_label, center_seconds, window_minutes)
        print(f"  Calling Claude ({model})...")
        raw = call_claude_vision(montage_path, prompt, model)
        print(f"  Response: {raw[:200]}")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            import re
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            try:
                parsed = json.loads(m.group(0)) if m else {}
            except (json.JSONDecodeError, AttributeError):
                parsed = {}

        result = {
            "boundary": boundary_label,
            "center_seconds": center_seconds,
            **parsed,
            "raw_response": raw,
        }
        if output_dir:
            result["montage_path"] = str(montage_path)
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Exp 04: Video frame confirmation of song boundaries"
    )
    parser.add_argument("video_file", help="Video file to analyze")
    parser.add_argument("--boundaries-json", default="experiments/exp03_result.json",
                        help="exp03_result.json with song_boundaries list")
    parser.add_argument("--boundaries",
                        help="Override: comma-separated MM:SS timestamps to check")
    parser.add_argument("--extra-windows",
                        help="Extra windows to check (e.g., ambiguous quiet zones), comma-separated MM:SS")
    parser.add_argument("--window-minutes", type=float, default=3.0,
                        help="Half-width of window around each boundary (default: 3)")
    parser.add_argument("--frame-interval", type=int, default=10,
                        help="Seconds between frames (default: 10)")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract frames and build montage but don't call Claude")
    parser.add_argument("-o", "--output-dir", help="Directory to save montage PNGs")
    parser.add_argument("--save-json", help="Save results as JSON")
    args = parser.parse_args()

    if not Path(args.video_file).exists():
        sys.exit(f"Error: video not found: {args.video_file}")

    if not PIL_AVAILABLE:
        sys.exit("Error: Pillow not installed. Run: uv add Pillow")

    # Build list of (label, center_seconds)
    boundaries: list[tuple[str, float]] = []

    if args.boundaries:
        for ts in args.boundaries.split(","):
            ts = ts.strip()
            s = parse_time(ts)
            boundaries.append((ts, s))
    else:
        json_path = Path(args.boundaries_json)
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            for b in data.get("llm_clustering_result", {}).get("song_boundaries", []):
                boundaries.append((b["time"], b["time_seconds"]))
        else:
            sys.exit(f"Error: boundaries JSON not found: {json_path}")

    if args.extra_windows:
        for ts in args.extra_windows.split(","):
            ts = ts.strip()
            s = parse_time(ts)
            boundaries.append((f"{ts} (extra)", s))

    output_dir = Path(args.output_dir) if args.output_dir else None

    print(f"Checking {len(boundaries)} boundary/window(s):")
    for label, t in boundaries:
        print(f"  {label} ({t:.0f}s)")

    results = []
    for label, center in boundaries:
        result = process_boundary(
            boundary_label=label,
            center_seconds=center,
            video_file=args.video_file,
            window_minutes=args.window_minutes,
            frame_interval=args.frame_interval,
            model=args.model,
            dry_run=args.dry_run,
            output_dir=output_dir,
        )
        results.append(result)
        print(f"  → {result.get('verdict', '?')}  refined={result.get('refined_timestamp', 'n/a')}")
        if result.get("evidence"):
            print(f"     {result['evidence'][:150]}")

    print(f"\n{'='*60}")
    print(f"Summary: {len(results)} boundaries checked")
    confirmed = [r for r in results if r.get("verdict") == "confirmed"]
    denied = [r for r in results if r.get("verdict") == "denied"]
    uncertain = [r for r in results if r.get("verdict") == "uncertain"]
    print(f"  Confirmed: {len(confirmed)}  Denied: {len(denied)}  Uncertain: {len(uncertain)}")

    if args.save_json:
        output = {
            "experiment": "exp04_video_frames",
            "video_file": args.video_file,
            "params": {
                "window_minutes": args.window_minutes,
                "frame_interval": args.frame_interval,
                "model": args.model,
            },
            "results": results,
        }
        with open(args.save_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved: {args.save_json}")


if __name__ == "__main__":
    main()
