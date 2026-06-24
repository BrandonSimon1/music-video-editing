#!/usr/bin/env python3
"""
Structure-Aware Clip Extraction using allin1

Uses the All-In-One Music Structure Analyzer to extract clips aligned with
actual downbeats and musical structure. Unlike the librosa-based approach,
this uses deep learning to jointly predict beats, downbeats, segment
boundaries, and segment labels.

Key improvements over beat_clip_extractor.py:
- Downbeat detection: measures start on actual downbeats (no phase ambiguity)
- Segment awareness: clips don't cut across structural transitions
- Per-beat tempo: no single global tempo assumption
"""

import argparse
import concurrent.futures
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Dict, Optional

import base64
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Stage 1: Analyze with allin1
# =============================================================================

def analyze_audio(audio_file: str, device: str = "cpu") -> Dict:
    """
    Run allin1 analysis on an audio/video file.

    Returns a plain dict with beats, downbeats, segments (as dicts), and BPM.
    This is the canonical "analysis" format — save it once, reuse forever.
    """
    import allin1  # noqa: PLC0415 — lazy import; allin1/madmom only needed for analyze command

    print(f"Analyzing: {audio_file}")
    print("This may take a while (source separation + inference)...")

    result = allin1.analyze(
        paths=audio_file,
        device=device,
        keep_byproducts=False,
    )

    beats = sorted(float(b) for b in result.beats)
    downbeats = sorted(float(d) for d in result.downbeats)
    total_duration = (beats[-1] + float(np.mean(np.diff(beats)))) if len(beats) > 1 else None

    print(f"BPM: {result.bpm}")
    print(f"Beats: {len(beats)}, Downbeats: {len(downbeats)}, Segments: {len(result.segments)}")
    for seg in result.segments:
        print(f"  {seg.start:7.2f}s - {seg.end:7.2f}s  {seg.label}")

    return {
        "audio_file": audio_file,
        "total_duration": float(total_duration) if total_duration else None,
        "bpm": result.bpm,
        "total_beats": len(beats),
        "total_downbeats": len(downbeats),
        "total_segments": len(result.segments),
        "beats": beats,
        "downbeats": downbeats,
        "segments": [
            {"start": float(s.start), "end": float(s.end), "label": s.label}
            for s in result.segments
        ],
    }


# =============================================================================
# Stage 2: Group downbeats into measures and create clips
# =============================================================================

def _find_clip_end_idx(
    start_idx: int,
    downbeats: List[float],
    min_duration: float = 30.0,
    max_duration: float = 60.0,
) -> tuple:
    """
    Find the best ending downbeat index for a clip starting at start_idx.

    Priority:
      1. Largest multiple-of-4 measure count whose duration is in [min, max].
      2. If none, largest any-measure count in [min, max].
      3. If none, measure count whose duration is closest to the window center.

    Returns (end_idx, num_measures).
    """
    start_time = downbeats[start_idx]
    center = (min_duration + max_duration) / 2

    candidates = []  # list of (n_measures, end_idx, duration)
    for n in range(1, len(downbeats) - start_idx):
        end_idx = start_idx + n
        dur = downbeats[end_idx] - start_time
        candidates.append((n, end_idx, dur))
        if dur > max_duration * 1.5:
            break

    if not candidates:
        return None, 0

    mod4_in_range = [(n, ei, d) for n, ei, d in candidates
                     if n % 4 == 0 and min_duration <= d <= max_duration]
    if mod4_in_range:
        n, ei, _ = mod4_in_range[0]  # shortest (first) multiple-of-4 in range
        return ei, n

    any_in_range = [(n, ei, d) for n, ei, d in candidates
                    if min_duration <= d <= max_duration]
    if any_in_range:
        n, ei, _ = any_in_range[0]  # shortest in range
        return ei, n

    # Nothing in range — pick the measure count whose duration is closest to center
    n, ei, _ = min(candidates, key=lambda x: abs(x[2] - center))
    return ei, n


def build_clips(
    analysis: Dict,
    min_duration: float = 30.0,
    max_duration: float = 60.0,
) -> List[Dict]:
    """
    Build clips from an analysis dict (output of analyze_audio or a rebuilt equivalent).

    Each clip starts on a downbeat and ends on the first multiple-of-4-measure
    boundary whose duration falls within [min_duration, max_duration] seconds.
    Falls back to any measure count in range, then to the measure count closest
    to the window center if the tempo is unusually fast or slow.
    """
    downbeats = sorted(analysis["downbeats"])
    beats = sorted(analysis.get("beats", []))
    segments = analysis.get("segments", [])  # list of dicts: {start, end, label}

    if len(downbeats) < 2:
        print("Error: fewer than 2 downbeats detected")
        return []

    print(f"\nBuilding clips from {len(downbeats)} downbeats "
          f"(target duration: {min_duration}–{max_duration}s, multiples of 4 measures)...")

    clips = []
    clip_id = 1
    i = 0

    while i < len(downbeats) - 1:
        clip_start = downbeats[i]

        end_idx, num_measures = _find_clip_end_idx(i, downbeats, min_duration, max_duration)
        if end_idx is None:
            break

        clip_end = downbeats[end_idx]
        clip_beats = [b for b in beats if clip_start <= b < clip_end]

        clip_labels = []
        for seg in segments:
            if seg["start"] < clip_end and seg["end"] > clip_start:
                clip_labels.append(seg["label"])

        duration = clip_end - clip_start
        beat_density = len(clip_beats) / duration if duration > 0 else 0.0

        beat_intervals = np.diff(clip_beats) if len(clip_beats) > 1 else []
        beat_cv = float(np.std(beat_intervals) / np.mean(beat_intervals)) if len(beat_intervals) > 1 else 1.0

        clips.append({
            "clip_id": clip_id,
            "start_time": float(clip_start),
            "end_time": float(clip_end),
            "duration": float(duration),
            "num_measures": num_measures,
            "num_beats": len(clip_beats),
            "beat_density": round(beat_density, 4),
            "beat_regularity_cv": round(beat_cv, 4),
            "segment_labels": clip_labels,
            "beats": [float(b) for b in clip_beats],
        })

        clip_id += 1
        i = end_idx

    print(f"Created {len(clips)} clips")
    return clips


# =============================================================================
# Visualization
# =============================================================================

def visualize_analysis(
    analysis: Dict,
    clips: List[Dict],
    output_file: Optional[str] = None,
):
    """
    Visualize beats, downbeats, segment boundaries, and clip boundaries.
    """
    print("Creating visualization...")

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    ax1 = axes[0]
    ax1.set_title(
        f"allin1 Analysis — {analysis['bpm']} BPM, "
        f"{analysis['total_beats']} beats, {analysis['total_downbeats']} downbeats, "
        f"{analysis['total_segments']} segments"
    )

    for beat in analysis["beats"]:
        ax1.axvline(beat, color="blue", alpha=0.2, linewidth=0.5)

    for db in analysis["downbeats"]:
        ax1.axvline(db, color="green", alpha=0.5, linewidth=1.0)

    for seg in analysis["segments"]:
        ax1.axvline(seg["start"], color="red", linestyle="--", alpha=0.8, linewidth=1.5)
        mid = (seg["start"] + seg["end"]) / 2
        ax1.text(mid, 0.8, seg["label"], ha="center", va="center",
                 fontsize=7, rotation=0,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.7),
                 transform=ax1.get_xaxis_transform())

    ax1.set_ylabel("Beats / Structure")
    ax1.set_yticks([])
    ax1.legend(
        handles=[
            plt.Line2D([0], [0], color="blue", alpha=0.4, label="Beat"),
            plt.Line2D([0], [0], color="green", alpha=0.6, label="Downbeat"),
            plt.Line2D([0], [0], color="red", linestyle="--", label="Segment boundary"),
        ],
        loc="upper right",
    )

    # Panel 2: Clip boundaries
    ax2 = axes[1]
    for i, clip in enumerate(clips):
        color = plt.cm.tab20(i % 20)
        ax2.axvspan(clip["start_time"], clip["end_time"],
                     alpha=0.3, color=color)
        mid = (clip["start_time"] + clip["end_time"]) / 2
        ax2.text(mid, 0.5, f"{clip['clip_id']}", ha="center", va="center",
                 fontsize=6, transform=ax2.get_xaxis_transform())

    ax2.set_ylabel("Clips")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_yticks([])

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved visualization: {output_file}")
    else:
        plt.show()

    plt.close(fig)


# =============================================================================
# Clip Rendering
# =============================================================================

def render_clips(
    clips_json: str,
    video_file: str,
    output_dir: str,
    indices: Optional[List[int]] = None,
):
    """
    Extract video clips using ffmpeg -c copy (fast, no re-encoding).

    Args:
        clips_json: Path to the clips JSON file
        video_file: Path to the source video
        output_dir: Directory to write clip files
        indices: Optional list of clip indices (0-based) to render.
                 If None, renders all clips.
    """
    with open(clips_json) as f:
        data = json.load(f)

    all_clips = data["clips"]
    if indices is not None:
        selected = [all_clips[i] for i in indices if i < len(all_clips)]
    else:
        selected = all_clips

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ext = Path(video_file).suffix
    for clip in selected:
        clip_id = clip["clip_id"]
        start = clip["start_time"]
        duration = clip["duration"]
        outfile = out / f"clip_{clip_id:04d}{ext}"

        print(f"  Clip {clip_id:4d}: {start:7.2f}s  ({duration:5.2f}s) -> {outfile.name}")

        cmd = [
            "ffmpeg",
            "-ss", str(start),
            "-t", str(duration),
            "-i", video_file,
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            "-y",
            str(outfile),
        ]
        subprocess.run(cmd, capture_output=True, check=True)

    print(f"Rendered {len(selected)} clips to {out}")


# =============================================================================
# Clip Filtering (music vs. talking)
# =============================================================================

def filter_clips(
    clips_json: str,
    output_json: Optional[str] = None,
    min_beat_density: float = 1.2,
    max_beat_cv: float = 0.25,
    dry_run: bool = False,
) -> Dict:
    """
    Filter clips to remove non-musical sections (talking, silence, tuning).

    Uses two metrics already computed per clip:
    - beat_density: beats per second. Music typically > 1.2 bps. Talking/silence
      produces sparse, hallucinated beats (< 1.0 bps).
    - beat_regularity_cv: coefficient of variation of beat intervals. Music has
      very regular beats (CV < 0.15). Talking has erratic spacing (CV > 0.3).

    A clip is kept if: beat_density >= min_beat_density AND beat_cv <= max_beat_cv.
    """
    with open(clips_json) as f:
        data = json.load(f)

    all_clips = data["clips"]

    # Check if metrics exist (older JSONs might not have them)
    if all_clips and "beat_density" not in all_clips[0]:
        print("Error: clips JSON missing beat_density/beat_regularity_cv fields.")
        print("Re-run 'analyze' to regenerate the JSON with these metrics.")
        sys.exit(1)

    kept = []
    removed = []
    for clip in all_clips:
        density = clip["beat_density"]
        cv = clip["beat_regularity_cv"]
        if density >= min_beat_density and cv <= max_beat_cv:
            kept.append(clip)
        else:
            removed.append(clip)

    print(f"Filter: beat_density >= {min_beat_density}, beat_cv <= {max_beat_cv}")
    print(f"  Total clips:    {len(all_clips)}")
    print(f"  Kept (music):   {len(kept)}")
    print(f"  Removed:        {len(removed)}")

    if dry_run:
        print(f"\n--- Removed clips (dry run) ---")
        for clip in removed:
            labels = ", ".join(clip["segment_labels"][:3])
            print(f"  Clip {clip['clip_id']:4d}: "
                  f"{clip['start_time']:7.1f}s  "
                  f"density={clip['beat_density']:.3f}  "
                  f"cv={clip['beat_regularity_cv']:.3f}  "
                  f"[{labels}]")
        print(f"\n--- Kept clips sample (first 10) ---")
        for clip in kept[:10]:
            labels = ", ".join(clip["segment_labels"][:3])
            print(f"  Clip {clip['clip_id']:4d}: "
                  f"{clip['start_time']:7.1f}s  "
                  f"density={clip['beat_density']:.3f}  "
                  f"cv={clip['beat_regularity_cv']:.3f}  "
                  f"[{labels}]")
        return data

    # Renumber clips sequentially
    for i, clip in enumerate(kept, 1):
        clip["clip_id"] = i

    data["clips"] = kept
    data["num_clips"] = len(kept)
    data["filter_parameters"] = {
        "min_beat_density": min_beat_density,
        "max_beat_cv": max_beat_cv,
        "original_clip_count": len(all_clips),
    }

    if output_json:
        with open(output_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved filtered clips: {output_json}")

    return data


# =============================================================================
# Clip Regrouping (merge short clips into longer target-duration clips)
# =============================================================================

def merge_clip_group(clips: List[Dict], group_id: int) -> Dict:
    """Merge a list of consecutive clips into a single clip."""
    start_time = clips[0]["start_time"]
    end_time = clips[-1]["end_time"]
    duration = end_time - start_time

    all_beats = []
    for c in clips:
        all_beats.extend(c["beats"])
    all_beats = sorted(set(all_beats))

    # Collect segment labels in order, deduplicating
    all_labels = []
    seen = set()
    for c in clips:
        for label in c["segment_labels"]:
            if label not in seen:
                all_labels.append(label)
                seen.add(label)

    num_measures = sum(c["num_measures"] for c in clips)

    beat_density = len(all_beats) / duration if duration > 0 else 0.0
    beat_intervals = np.diff(all_beats) if len(all_beats) > 1 else []
    if len(beat_intervals) > 1:
        beat_cv = float(np.std(beat_intervals) / np.mean(beat_intervals))
    else:
        beat_cv = 1.0

    return {
        "clip_id": group_id,
        "start_time": float(start_time),
        "end_time": float(end_time),
        "duration": float(duration),
        "num_measures": num_measures,
        "num_beats": len(all_beats),
        "beat_density": round(beat_density, 4),
        "beat_regularity_cv": round(beat_cv, 4),
        "segment_labels": all_labels,
        "beats": [float(b) for b in all_beats],
        "source_clip_ids": [c["clip_id"] for c in clips],
    }


def regroup_clips(
    clips_json: str,
    output_json: Optional[str] = None,
    min_duration: float = 30.0,
    max_duration: float = 60.0,
) -> Dict:
    """
    Merge consecutive short clips into longer clips targeting a duration range.

    Grouping rules:
    - Always keep adding clips until min_duration is reached.
    - Once min_duration is reached, stop at the next segment boundary or
      before exceeding max_duration (whichever comes first).
    - If no natural stopping point before max_duration, stop at max_duration.
    - Partial groups at the end of the recording are included as-is.
    """
    with open(clips_json) as f:
        data = json.load(f)

    all_clips = data["clips"]
    print(f"Regrouping {len(all_clips)} clips into {min_duration}-{max_duration}s clips...")

    groups = []
    group_id = 1
    i = 0

    while i < len(all_clips):
        group = [all_clips[i]]
        group_duration = all_clips[i]["duration"]
        j = i + 1

        while j < len(all_clips):
            next_clip = all_clips[j]

            # Once we've hit min_duration, check for natural stopping points
            if group_duration >= min_duration:
                # Stop if adding next clip would exceed max_duration
                if group_duration + next_clip["duration"] > max_duration:
                    break
                # Stop at segment boundaries
                cur_labels = set(group[-1]["segment_labels"])
                next_labels = set(next_clip["segment_labels"])
                if cur_labels != next_labels:
                    break

            group.append(next_clip)
            group_duration += next_clip["duration"]
            j += 1

        merged = merge_clip_group(group, group_id)
        groups.append(merged)
        group_id += 1
        i = j

    durations = [g["duration"] for g in groups]
    print(f"Created {len(groups)} clips")
    print(f"  Duration range: {min(durations):.1f}s - {max(durations):.1f}s")
    print(f"  Average duration: {sum(durations)/len(durations):.1f}s")

    result = dict(data)
    result["clips"] = groups
    result["num_clips"] = len(groups)
    result["regroup_parameters"] = {
        "min_duration": min_duration,
        "max_duration": max_duration,
        "original_clip_count": len(all_clips),
    }

    if output_json:
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved regrouped clips: {output_json}")

    return result


# =============================================================================
# Visual Filter (Claude vision — are people actively playing?)
# =============================================================================

def _extract_frame_bytes(video_file: str, timestamp: float) -> Optional[bytes]:
    """Return JPEG bytes for a single frame at the given timestamp via ffmpeg pipe."""
    cmd = [
        "ffmpeg",
        "-ss", str(timestamp),
        "-i", video_file,
        "-vframes", "1",
        "-vf", "scale=640:-1",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-q:v", "5",
        "-loglevel", "error",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        return None
    return result.stdout


def _classify_clip(
    clip: Dict,
    video_file: str,
    model: str,
    frames: int,
) -> Dict:
    """
    Ask Claude (via `claude -p --input-format stream-json`) whether people are
    actively playing instruments in the clip.

    Samples `frames` evenly-spaced timestamps and uses majority vote.
    Returns a result dict with keys: clip_id, is_playing, votes, responses, error.
    """
    duration = clip["end_time"] - clip["start_time"]
    if frames == 1:
        sample_times = [clip["start_time"] + duration / 2]
    else:
        sample_times = [
            clip["start_time"] + duration * (i + 1) / (frames + 1)
            for i in range(frames)
        ]

    prompt_text = (
        "Are people actively playing musical instruments "
        "in this image? Answer YES or NO, then one brief sentence."
    )

    votes = []
    responses = []
    for t in sample_times:
        frame_bytes = _extract_frame_bytes(video_file, t)
        if frame_bytes is None:
            votes.append(True)  # keep clip if frame extraction fails
            responses.append("frame extraction failed")
            continue

        image_b64 = base64.standard_b64encode(frame_bytes).decode("utf-8")
        msg_json = json.dumps({
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64,
                    }},
                    {"type": "text", "text": prompt_text},
                ],
            },
        })

        try:
            cmd = [
                "claude", "--model", model, "-p",
                "--input-format", "stream-json",
                "--output-format", "stream-json",
                "--verbose",
            ]
            result = subprocess.run(
                cmd, input=msg_json, capture_output=True, text=True, timeout=60,
            )
            # Extract text from the assistant message event
            text = ""
            for line in result.stdout.splitlines():
                try:
                    obj = json.loads(line)
                    if obj.get("type") == "assistant":
                        content = obj.get("message", {}).get("content", [])
                        for block in content:
                            if block.get("type") == "text":
                                text = block["text"].strip()
                                break
                except (json.JSONDecodeError, KeyError):
                    continue
            if not text:
                text = result.stderr.strip() or "empty response"
            votes.append(text.upper().startswith("YES"))
            responses.append(text[:200])
        except subprocess.TimeoutExpired:
            votes.append(True)  # keep clip on timeout
            responses.append("timeout")
        except Exception as e:
            votes.append(True)  # keep clip on unexpected error
            responses.append(f"error: {e}")

    is_playing = sum(votes) > len(votes) / 2  # majority vote
    return {
        "clip_id": clip["clip_id"],
        "is_playing": is_playing,
        "votes": votes,
        "responses": responses,
        "error": any(r in ("timeout", "frame extraction failed") or r.startswith("error:")
                     for r in responses),
    }


def visual_filter_clips(
    clips_json: str,
    video_file: str,
    output_json: Optional[str] = None,
    model: str = "claude-haiku-4-5",
    workers: int = 8,
    frames: int = 1,
    cache_file: Optional[str] = None,
    no_cache: bool = False,
    dry_run: bool = False,
) -> Dict:
    """
    Filter clips by asking Claude whether people are visibly playing instruments.

    Extracts a video frame per clip and uses Claude vision to classify it.
    Results are cached to a sidecar JSON so interrupted runs can resume.

    Args:
        clips_json:   Path to clips JSON (e.g. the beat-density-filtered output).
        video_file:   Path to the source video.
        output_json:  Where to write the filtered JSON. Defaults to
                      <clips_json stem>_visual.json.
        model:        Claude model for vision. Default: claude-haiku-4-5.
        workers:      Number of parallel API requests. Default: 8.
        frames:       Frames to sample per clip (majority vote). Default: 1.
        cache_file:   Path to sidecar cache JSON. Auto-derived if None.
        no_cache:     Ignore and overwrite the cache if True.
        dry_run:      Print stats but don't write output JSON.
    """
    with open(clips_json) as f:
        data = json.load(f)

    all_clips = data["clips"]
    print(f"Visual filter: {len(all_clips)} clips via `claude -p`, model={model}, "
          f"workers={workers}, frames_per_clip={frames}")

    # --- cache ---
    if cache_file is None:
        cache_file = str(Path(clips_json).with_suffix("")) + "_visual_cache.json"

    cache: Dict[str, Dict] = {}
    if not no_cache and Path(cache_file).exists():
        with open(cache_file) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached results from {cache_file}")

    cache_lock = threading.Lock()

    def save_cache():
        with cache_lock:
            with open(cache_file, "w") as f:
                json.dump(cache, f, indent=2)

    # --- classify ---
    to_classify = [c for c in all_clips if str(c["clip_id"]) not in cache]
    already_done = len(all_clips) - len(to_classify)
    if already_done:
        print(f"  Skipping {already_done} clips already cached, "
              f"classifying {len(to_classify)} new clips...")

    def classify_and_cache(clip: Dict) -> Dict:
        result = _classify_clip(clip, video_file, model, frames)
        with cache_lock:
            cache[str(clip["clip_id"])] = result
        return result

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(classify_and_cache, c): c for c in to_classify}
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            result = future.result()
            status = "PLAY" if result["is_playing"] else "SKIP"
            if completed % 20 == 0 or completed == len(to_classify):
                print(f"  [{completed}/{len(to_classify)}] clip {result['clip_id']}: "
                      f"{status}")
            # save cache every 25 classifications
            if completed % 25 == 0:
                save_cache()

    if to_classify:
        save_cache()

    # --- apply cache to full clip list ---
    kept = []
    removed = []
    errors = []
    for clip in all_clips:
        res = cache.get(str(clip["clip_id"]))
        if res is None:
            kept.append(clip)  # wasn't classified — keep
            continue
        if res.get("error"):
            errors.append(clip["clip_id"])
        if res["is_playing"]:
            kept.append(clip)
        else:
            removed.append(clip)

    print(f"\nVisual filter results:")
    print(f"  Total:   {len(all_clips)}")
    print(f"  Playing: {len(kept)}")
    print(f"  Not playing: {len(removed)}")
    if errors:
        print(f"  Errors (kept): {len(errors)}")

    if dry_run:
        print(f"\n--- Removed clips (dry run) ---")
        for clip in removed:
            cid = str(clip["clip_id"])
            resp = cache.get(cid, {}).get("responses", ["?"])
            print(f"  Clip {clip['clip_id']:4d}: {clip['start_time']:7.1f}s  "
                  f"| {resp[0][:80] if resp else '?'}")
        return data

    # renumber
    for i, clip in enumerate(kept, 1):
        clip["clip_id"] = i

    result_data = dict(data)
    result_data["clips"] = kept
    result_data["num_clips"] = len(kept)
    result_data["visual_filter_parameters"] = {
        "model": model,
        "frames_per_clip": frames,
        "original_clip_count": len(all_clips),
        "cache_file": cache_file,
    }

    if output_json:
        with open(output_json, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"\nSaved visually filtered clips: {output_json}")

    return result_data


# =============================================================================
# Pipeline entry points
# =============================================================================

def run_analysis(
    audio_file: str,
    output_json: Optional[str] = None,
    output_viz: Optional[str] = None,
    device: str = "cpu",
) -> Dict:
    """Run allin1, save the raw analysis JSON. No clip building."""
    analysis = analyze_audio(audio_file, device=device)

    if output_json:
        with open(output_json, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nSaved analysis JSON: {output_json}")

    if output_viz:
        visualize_analysis(analysis, [], output_viz)

    return analysis


def build_clips_from_file(
    analysis_json: str,
    output_json: Optional[str] = None,
    min_duration: float = 30.0,
    max_duration: float = 60.0,
    output_viz: Optional[str] = None,
) -> Dict:
    """Load an analysis JSON and build clips with the duration-based logic."""
    with open(analysis_json) as f:
        analysis = json.load(f)

    clips = build_clips(analysis, min_duration, max_duration)

    result = {
        **{k: v for k, v in analysis.items() if k not in ("beats", "downbeats")},
        "num_clips": len(clips),
        "clip_parameters": {
            "min_duration": min_duration,
            "max_duration": max_duration,
            "model": "allin1 (harmonix-all)",
        },
        "clips": clips,
    }

    if output_json:
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved clips JSON: {output_json}")

    if output_viz:
        visualize_analysis(analysis, clips, output_viz)

    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Structure-aware clip extraction using allin1"
    )
    sub = parser.add_subparsers(dest="command")

    # -- analyze command --
    p_analyze = sub.add_parser("analyze", help="Run allin1 and save raw analysis JSON (beats, downbeats, segments)")
    p_analyze.add_argument("audio_file", help="Path to audio or video file")
    p_analyze.add_argument("-o", "--output", help="Output JSON (default: <audio>_analysis.json)")
    p_analyze.add_argument("-v", "--visualize", help="Save visualization PNG")
    p_analyze.add_argument("--device", default="cpu", help="Torch device (cpu or cuda)")

    # -- build-clips command --
    p_build = sub.add_parser("build-clips", help="Build clips from an analysis JSON")
    p_build.add_argument("analysis_json", help="Path to analysis JSON (output of analyze)")
    p_build.add_argument("-o", "--output", help="Output clips JSON (default: <analysis>_clips.json)")
    p_build.add_argument("-v", "--visualize", help="Save visualization PNG")
    p_build.add_argument("--min-duration", type=float, default=30.0,
                         help="Minimum clip duration in seconds (default: 30)")
    p_build.add_argument("--max-duration", type=float, default=60.0,
                         help="Maximum clip duration in seconds (default: 60)")

    # -- filter command --
    p_filter = sub.add_parser("filter", help="Filter clips to remove non-musical sections")
    p_filter.add_argument("clips_json", help="Path to clips JSON")
    p_filter.add_argument("-o", "--output", help="Output filtered JSON (default: <input>_filtered.json)")
    p_filter.add_argument(
        "--min-beat-density", type=float, default=1.2,
        help="Minimum beats/sec to keep (default: 1.2)",
    )
    p_filter.add_argument(
        "--max-beat-cv", type=float, default=0.25,
        help="Maximum beat interval CV to keep (default: 0.25)",
    )
    p_filter.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be filtered without writing output",
    )

    # -- render command --
    p_render = sub.add_parser("render", help="Render video clips from clips JSON")
    p_render.add_argument("clips_json", help="Path to clips JSON")
    p_render.add_argument("video_file", help="Path to source video")
    p_render.add_argument("-o", "--output-dir", required=True, help="Output directory")
    p_render.add_argument(
        "--evenly-spaced", type=int, metavar="N",
        help="Render N evenly-spaced clips instead of all",
    )

    # -- visual-filter command --
    p_vis = sub.add_parser(
        "visual-filter",
        help="Filter clips using Claude vision (are people actively playing?)",
    )
    p_vis.add_argument("clips_json", help="Path to clips JSON")
    p_vis.add_argument("video_file", help="Path to source video")
    p_vis.add_argument(
        "-o", "--output",
        help="Output JSON (default: <clips_json stem>_visual.json)",
    )
    p_vis.add_argument(
        "--model", default="claude-haiku-4-5",
        help="Claude model for vision classification (default: claude-haiku-4-5)",
    )
    p_vis.add_argument(
        "--workers", type=int, default=8,
        help="Parallel API requests (default: 8)",
    )
    p_vis.add_argument(
        "--frames", type=int, default=1,
        help="Frames to sample per clip for majority vote (default: 1)",
    )
    p_vis.add_argument(
        "--cache-file",
        help="Sidecar cache JSON path (default: auto-derived from output path)",
    )
    p_vis.add_argument(
        "--no-cache", action="store_true",
        help="Ignore existing cache and re-classify all clips",
    )
    p_vis.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be removed without writing output",
    )

    args = parser.parse_args()

    if args.command == "analyze":
        audio_file = args.audio_file
        if not Path(audio_file).exists():
            print(f"Error: file not found: {audio_file}", file=sys.stderr)
            sys.exit(1)

        output_json = args.output
        if output_json is None:
            output_json = str(Path(audio_file).with_suffix("")) + "_analysis.json"

        run_analysis(
            audio_file=audio_file,
            output_json=output_json,
            output_viz=args.visualize,
            device=args.device,
        )

    elif args.command == "build-clips":
        if not Path(args.analysis_json).exists():
            print(f"Error: file not found: {args.analysis_json}", file=sys.stderr)
            sys.exit(1)

        output_json = args.output
        if output_json is None:
            base = str(Path(args.analysis_json).with_suffix(""))
            output_json = base.replace("_analysis", "") + "_clips.json"

        build_clips_from_file(
            analysis_json=args.analysis_json,
            output_json=output_json,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            output_viz=args.visualize,
        )

    elif args.command == "filter":
        if not Path(args.clips_json).exists():
            print(f"Error: file not found: {args.clips_json}", file=sys.stderr)
            sys.exit(1)

        output_json = args.output
        if output_json is None and not args.dry_run:
            base = str(Path(args.clips_json).with_suffix(""))
            if base.endswith("_clips"):
                output_json = base.replace("_clips", "_clips_filtered") + ".json"
            else:
                output_json = base + "_filtered.json"

        filter_clips(
            clips_json=args.clips_json,
            output_json=output_json,
            min_beat_density=args.min_beat_density,
            max_beat_cv=args.max_beat_cv,
            dry_run=args.dry_run,
        )

    elif args.command == "render":
        if not Path(args.clips_json).exists():
            print(f"Error: file not found: {args.clips_json}", file=sys.stderr)
            sys.exit(1)
        if not Path(args.video_file).exists():
            print(f"Error: file not found: {args.video_file}", file=sys.stderr)
            sys.exit(1)

        indices = None
        if args.evenly_spaced:
            with open(args.clips_json) as f:
                data = json.load(f)
            n_clips = data["num_clips"]
            n = args.evenly_spaced
            indices = [int(round(i * (n_clips - 1) / (n - 1))) for i in range(n)]

        render_clips(args.clips_json, args.video_file, args.output_dir, indices)

    elif args.command == "visual-filter":
        for path in (args.clips_json, args.video_file):
            if not Path(path).exists():
                print(f"Error: file not found: {path}", file=sys.stderr)
                sys.exit(1)

        output_json = args.output
        if output_json is None and not args.dry_run:
            base = str(Path(args.clips_json).with_suffix(""))
            output_json = base + "_visual.json"

        visual_filter_clips(
            clips_json=args.clips_json,
            video_file=args.video_file,
            output_json=output_json,
            model=args.model,
            workers=args.workers,
            frames=args.frames,
            cache_file=args.cache_file,
            no_cache=args.no_cache,
            dry_run=args.dry_run,
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
