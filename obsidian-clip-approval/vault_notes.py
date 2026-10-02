"""Shared helpers for the Obsidian clip-approval workflow.

Used by both the clip-splitting pipeline (which creates one note per
rendered clip) and the caption-overlay pickup script (which scans for
approved notes — those without the `needs-approval` tag — and updates
them). Kept as a small shared module rather than duplicated in each
workstream so the two stay in sync on note shape.

See wiki/investigations/obsidian-clip-approval/index.md for the full design.
"""

import json
import os
import re
from pathlib import Path

import yaml

NOTES_SUBDIR = "Music Clips"
TAG = "music-clip"
NEEDS_APPROVAL_TAG = "needs-approval"

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)


def _represent_none(dumper, _):
    return dumper.represent_scalar("tag:yaml.org,2002:null", "")


yaml.add_representer(type(None), _represent_none)


def resolve_vault_path() -> Path:
    """Find the currently-open Obsidian vault (or the only configured one)."""
    cfg = Path.home() / "Library/Application Support/obsidian/obsidian.json"
    data = json.loads(cfg.read_text())
    vaults = data["vaults"]
    open_vaults = [v["path"] for v in vaults.values() if v.get("open")]
    path = open_vaults[0] if open_vaults else next(iter(vaults.values()))["path"]
    return Path(path)


def read_note(path: Path) -> tuple[dict, str]:
    """Return (frontmatter dict, body) for a note."""
    match = FRONTMATTER_RE.match(path.read_text())
    if not match:
        raise ValueError(f"No frontmatter found in {path}")
    frontmatter = yaml.safe_load(match.group(1)) or {}
    return frontmatter, match.group(2)


def write_note(path: Path, frontmatter: dict, body: str) -> None:
    fm_text = yaml.dump(frontmatter, sort_keys=False, default_flow_style=False, allow_unicode=True)
    path.write_text(f"---\n{fm_text}---\n{body}")


def update_frontmatter(path: Path, updates: dict) -> None:
    frontmatter, body = read_note(path)
    frontmatter.update(updates)
    write_note(path, frontmatter, body)


def to_timecode(seconds: float) -> str:
    """53.81 -> '00_00_53_810' (HH_MM_SS_mmm), for self-describing clip filenames."""
    total_ms = round(seconds * 1000)
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1_000)
    return f"{hours:02d}_{minutes:02d}_{secs:02d}_{ms:03d}"


def clip_filename(start_time: float, end_time: float, ext: str = "mp4") -> str:
    """Self-describing, collision-free (per video) clip filename from its boundaries."""
    return f"clip-{to_timecode(start_time)}-{to_timecode(end_time)}.{ext}"


def create_clip_note(
    vault: Path,
    video_folder_name: str,
    clip: dict,
    clip_path: Path,
    source_video: str,
    analysis_path: Path | None,
    algorithm: str,
    algorithm_version: str,
    params: dict,
) -> Path | None:
    """Create a #music-clip note for one rendered clip.

    Returns the note path, or None if a note for this exact clip (same
    video folder + filename) already exists — reruns that regenerate an
    identical clip must not clobber a note a human may have already
    reviewed/approved.
    """
    notes_dir = vault / NOTES_SUBDIR / video_folder_name
    notes_dir.mkdir(parents=True, exist_ok=True)

    note_name = clip_path.stem
    note_path = notes_dir / f"{note_name}.md"
    if note_path.exists():
        return None

    frontmatter = {
        "tags": [TAG, NEEDS_APPROVAL_TAG],
        "status": "clip-created",
        "clip_path": str(clip_path.resolve()),
        "source_video": source_video,
        "analysis_path": str(analysis_path.resolve()) if analysis_path else None,
        "algorithm": algorithm,
        "algorithm_version": algorithm_version,
        "params": params,
        "start_time": clip["start_time"],
        "end_time": clip["end_time"],
        "duration": clip["duration"],
        "segment_labels": clip.get("segment_labels", []),
        "num_measures": clip.get("num_measures"),
        "beat_density": clip.get("beat_density"),
        "caption_text": None,
        "caption_emoji": None,
        "captioned_path": None,
    }

    body = (
        f"# {note_name}\n\n"
        f"**Source:** {source_video}\n"
        f"**Time:** {clip['start_time']:.2f}s – {clip['end_time']:.2f}s "
        f"({clip['duration']:.2f}s)\n"
        f"**Segment:** {', '.join(clip.get('segment_labels', [])) or 'n/a'}\n\n"
        f'<video src="file://{clip_path.resolve()}" controls></video>\n'
    )

    write_note(note_path, frontmatter, body)
    return note_path


def append_video_embed(path: Path, heading: str, video_path: Path) -> None:
    """Append a labeled HTML5 video embed to a note's body.

    Used to add the captioned-output preview below the original clip embed
    already in the note (see create_clip_note), so both the source clip and
    the captioned result are reviewable from the same note — same `file://`,
    no-copy-into-vault convention as the original embed.
    """
    frontmatter, body = read_note(path)
    body = (
        body.rstrip("\n")
        + f"\n\n## {heading}\n\n"
        + f'<video src="file://{video_path.resolve()}" controls></video>\n'
    )
    write_note(path, frontmatter, body)


def find_notes(vault: Path, tag: str = TAG) -> list[Path]:
    """All notes under Music Clips/ carrying the given tag."""
    notes_dir = vault / NOTES_SUBDIR
    if not notes_dir.exists():
        return []
    found = []
    for path in notes_dir.rglob("*.md"):
        frontmatter, _ = read_note(path)
        if tag in (frontmatter.get("tags") or []):
            found.append(path)
    return found


def find_pending_caption_notes(vault: Path) -> list[tuple[Path, dict]]:
    """(path, frontmatter) for notes approved (no `needs-approval` tag) and waiting to be captioned."""
    results = []
    for path in find_notes(vault):
        frontmatter, _ = read_note(path)
        tags = frontmatter.get("tags") or []
        if NEEDS_APPROVAL_TAG not in tags and frontmatter.get("status") == "pending-caption":
            results.append((path, frontmatter))
    return results
