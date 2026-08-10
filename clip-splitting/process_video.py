#!/usr/bin/env python3
"""
process_video.py — Run a clip extraction algorithm on a video folder.

Folder layout produced:
  <folder>/
    <video>.MOV
    analysis/
      YYYY-MM-DD-<algorithm>.json               ← algorithm cache (if the algorithm uses one)
      YYYY-MM-DD-<algorithm>_visual-cache.json   ← resume cache for visual filter (internal)
    clips/
      clip-<start-timecode>-<end-timecode>.mp4   ← flat, one per clip, self-naming
      ...

No clips.json manifest — every clip's metadata (timing, segment labels,
algorithm/params provenance, an analysis_path link) lives entirely in its
Obsidian #music-clip note. See wiki/investigations/obsidian-clip-approval/.

Reruns are idempotent: a clip whose timecode-derived filename already exists
is not re-rendered, and a note that already exists for that filename is not
overwritten (it may already be reviewed/approved by a human).

Adding a new algorithm:
  1. Create clip-splitting/algorithms/<name>.py
  2. Implement ALGORITHM_NAME, ALGORITHM_VERSION, add_args(parser), run(video, cache_dir, args)
  3. Pass --algorithm <name> at runtime

Usage:
  uv run process_video.py <folder> [--algorithm allin1] [options]
"""

import argparse
import importlib
import json
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

ALGORITHMS_DIR = Path(__file__).parent / "algorithms"
sys.path.insert(0, str(Path(__file__).parent.parent / "obsidian-clip-approval"))
import vault_notes

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".mts", ".m4v"}


def find_video(folder: Path, hint: str | None) -> Path:
    if hint:
        p = folder / hint
        if not p.exists():
            raise FileNotFoundError(f"Video not found: {p}")
        return p
    videos = [f for f in folder.iterdir()
              if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS]
    if not videos:
        raise FileNotFoundError(f"No video file found in {folder}")
    if len(videos) > 1:
        movs = [v for v in videos if v.suffix.upper() == ".MOV"]
        if len(movs) == 1:
            return movs[0]
        raise ValueError(
            f"Multiple video files in {folder}: {[v.name for v in videos]}\n"
            "Use --video <filename> to pick one."
        )
    return videos[0]


def load_algorithm(name: str):
    """Import algorithms/<name>.py and validate its interface."""
    sys.path.insert(0, str(ALGORITHMS_DIR.parent))
    try:
        mod = importlib.import_module(f"algorithms.{name}")
    except ModuleNotFoundError:
        available = [p.stem for p in ALGORITHMS_DIR.glob("*.py") if p.stem != "__init__"]
        raise SystemExit(
            f"Unknown algorithm '{name}'. Available: {', '.join(available)}\n"
            f"Add a new one in {ALGORITHMS_DIR}/"
        )
    for attr in ("ALGORITHM_NAME", "ALGORITHM_VERSION", "add_args", "run"):
        if not hasattr(mod, attr):
            raise SystemExit(f"Algorithm module '{name}' is missing '{attr}'")
    return mod


def render_clips_mp4(clips: list, video_file: Path, output_dir: Path) -> dict[int, Path]:
    """Render each clip to <output_dir>/clip-<start-timecode>-<end-timecode>.mp4.

    Skips clips whose file already exists (idempotent reruns). Returns a map
    of the clip's index in `clips` to its rendered path, for every clip
    (freshly rendered or already present) — callers need the path either way.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    n_rendered = 0
    for i, clip in enumerate(clips):
        filename = vault_notes.clip_filename(clip["start_time"], clip["end_time"])
        outfile = output_dir / filename
        paths[i] = outfile
        if outfile.exists():
            continue
        cmd = [
            "ffmpeg",
            "-ss", str(clip["start_time"]),
            "-t", str(clip["duration"]),
            "-i", str(video_file),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            "-y", str(outfile),
        ]
        print(f"  {clip['start_time']:7.2f}s  ({clip['duration']:5.2f}s) → {outfile.name}")
        subprocess.run(cmd, capture_output=True, check=True)
        n_rendered += 1
    print(f"Rendered {n_rendered} new clips ({len(clips) - n_rendered} already present) in {output_dir}")
    return paths


def main():
    # Two-pass arg parsing: first identify --algorithm so we can load its add_args,
    # then do a full parse with all flags registered.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--algorithm", default="allin1")
    pre_args, _ = pre.parse_known_args()

    algo = load_algorithm(pre_args.algorithm)

    parser = argparse.ArgumentParser(
        description="Run a clip extraction algorithm on a video folder"
    )
    parser.add_argument("folder", help="Folder containing the video file")
    parser.add_argument("--algorithm", default="allin1",
                        help="Algorithm to use (default: allin1)")
    parser.add_argument("--video", help="Video filename (if folder has multiple videos)")

    # Beat filter (skipped automatically if clips lack beat_density)
    g_beat = parser.add_argument_group("beat filter")
    g_beat.add_argument("--no-beat-filter", action="store_true",
                        help="Skip beat density / regularity filter")
    g_beat.add_argument("--min-beat-density", type=float, default=1.2,
                        help="Minimum beats/sec to keep (default: 1.2)")
    g_beat.add_argument("--max-beat-cv", type=float, default=0.25,
                        help="Maximum beat interval CV to keep (default: 0.25)")

    # Visual filter
    g_vis = parser.add_argument_group("visual filter")
    g_vis.add_argument("--no-visual-filter", action="store_true",
                       help="Skip the Claude vision step")
    g_vis.add_argument("--visual-model", default="claude-haiku-4-5",
                       help="Claude model for visual filter (default: claude-haiku-4-5)")
    g_vis.add_argument("--visual-workers", type=int, default=8,
                       help="Parallel workers for visual filter (default: 8)")
    g_vis.add_argument("--visual-frames", type=int, default=1,
                       help="Frames per clip for majority vote (default: 1)")

    # Obsidian clip-approval notes
    g_vault = parser.add_argument_group("obsidian notes")
    g_vault.add_argument("--no-vault-notes", action="store_true",
                         help="Skip creating Obsidian #music-clip notes for review")

    # Algorithm-specific flags
    algo.add_args(parser)

    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        sys.exit(f"Error: not a directory: {folder}")

    try:
        video_file = find_video(folder, args.video)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")
    print(f"Video:     {video_file.name}")
    print(f"Algorithm: {algo.ALGORITHM_NAME} ({algo.ALGORITHM_VERSION})")

    # Analysis cache dir (algorithms may use or ignore it)
    cache_dir = folder / "analysis"
    cache_dir.mkdir(exist_ok=True)

    # Run the algorithm → raw clips
    print(f"\n=== {algo.ALGORITHM_NAME} ===")
    clips = algo.run(video_file, cache_dir, args)
    n_built = len(clips)
    print(f"Got {n_built} clips from algorithm")

    # Find the analysis file the algorithm used/created, the same way it does
    # internally (most recent *-<algorithm>.json in cache_dir) — algorithms
    # don't return this directly, so we rediscover it for the note's
    # analysis_path link and to name the visual-filter cache after it.
    analysis_candidates = sorted(cache_dir.glob(f"*-{algo.ALGORITHM_NAME}.json"), reverse=True)
    analysis_path = analysis_candidates[0] if analysis_candidates else None
    cache_stem = analysis_path.stem if analysis_path else f"{date.today().isoformat()}-{algo.ALGORITHM_NAME}"

    clips_dir = folder / "clips"
    clips_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tmp_clips = tmp / "_clips.json"
        tmp_beat_filtered = tmp / "_beat_filtered.json"
        tmp_visual_filtered = tmp / "_visual_filtered.json"
        visual_cache = cache_dir / f"{cache_stem}_visual-cache.json"

        # Write raw clips to temp file (filter functions expect JSON on disk)
        with open(tmp_clips, "w") as f:
            json.dump({"clips": clips, "num_clips": len(clips)}, f)

        # Beat filter (skip if clips lack the metrics or --no-beat-filter)
        has_beat_metrics = clips and "beat_density" in clips[0]
        n_beat = n_built

        if not args.no_beat_filter and has_beat_metrics:
            print("\n=== Beat filter ===")
            sys.path.insert(0, str(Path(__file__).parent))
            from allin1_clip_extractor import filter_clips
            beat_data = filter_clips(
                str(tmp_clips),
                output_json=str(tmp_beat_filtered),
                min_beat_density=args.min_beat_density,
                max_beat_cv=args.max_beat_cv,
            )
            n_beat = len(beat_data["clips"])
            current_json = tmp_beat_filtered
        else:
            if not has_beat_metrics and not args.no_beat_filter:
                print("\n(Beat filter skipped — algorithm did not provide beat_density metrics)")
            current_json = tmp_clips

        # Visual filter
        n_visual = n_beat
        if not args.no_visual_filter:
            print("\n=== Visual filter ===")
            from allin1_clip_extractor import visual_filter_clips
            visual_data = visual_filter_clips(
                str(current_json),
                str(video_file),
                output_json=str(tmp_visual_filtered),
                model=args.visual_model,
                workers=args.visual_workers,
                frames=args.visual_frames,
                cache_file=str(visual_cache),
            )
            n_visual = len(visual_data["clips"])
            final_clips = visual_data["clips"]
        else:
            with open(current_json) as f:
                final_clips = json.load(f)["clips"]

    # Render (flat clips/, timecode-named, idempotent)
    print("\n=== Render ===")
    clip_paths = render_clips_mp4(final_clips, video_file, clips_dir)

    # Params, for provenance on each note (no clips.json anymore)
    params = {
        "beat_filter": not args.no_beat_filter and has_beat_metrics,
        "visual_filter": not args.no_visual_filter,
    }
    if not args.no_beat_filter and has_beat_metrics:
        params["beat_density_min"] = args.min_beat_density
        params["beat_cv_max"] = args.max_beat_cv
    if not args.no_visual_filter:
        params["visual_filter_model"] = args.visual_model
        params["visual_filter_frames"] = args.visual_frames
        params["visual_filter_workers"] = args.visual_workers
    algo_params = algo.get_params(args) if hasattr(algo, "get_params") else {}
    params.update(algo_params)

    # Obsidian #music-clip notes, one per rendered clip, for human review/approval.
    # See wiki/investigations/obsidian-clip-approval/index.md for the design.
    n_notes = 0
    if not args.no_vault_notes:
        print("\n=== Obsidian notes ===")
        vault = vault_notes.resolve_vault_path()
        for i, c in enumerate(final_clips):
            note_path = vault_notes.create_clip_note(
                vault=vault,
                video_folder_name=folder.name,
                clip=c,
                clip_path=clip_paths[i],
                source_video=video_file.name,
                analysis_path=analysis_path,
                algorithm=algo.ALGORITHM_NAME,
                algorithm_version=algo.ALGORITHM_VERSION,
                params=params,
            )
            if note_path is not None:
                n_notes += 1
        print(f"Created {n_notes} new notes ({len(final_clips) - n_notes} already existed) in "
              f"{vault / vault_notes.NOTES_SUBDIR / folder.name}")

    print(f"\nDone! {n_visual} clips in {clips_dir.relative_to(folder)}")


if __name__ == "__main__":
    main()
