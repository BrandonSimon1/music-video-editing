"""Shared helpers for the Obsidian clip-approval workflow.

Used by both the clip-splitting pipeline (which creates one note per
rendered clip) and the caption-overlay pickup script (which scans for
approved notes and updates them). Kept as a small shared module rather
than duplicated in each workstream so the two stay in sync on note shape.

See wiki/investigations/obsidian-clip-approval/index.md for the full design.
"""

import json
import os
import re
from pathlib import Path

import yaml

NOTES_SUBDIR = "Music Clips"
TAG = "music-clip"

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


def create_clip_note(
    vault: Path,
    run_name: str,
    clip: dict,
    clip_path: Path,
    source_video: str,
) -> Path:
    """Create a #music-clip note for one rendered clip. Returns the note path."""
    notes_dir = vault / NOTES_SUBDIR / run_name
    notes_dir.mkdir(parents=True, exist_ok=True)

    note_name = Path(clip["filename"]).stem  # e.g. "clip-001"
    note_path = notes_dir / f"{note_name}.md"

    frontmatter = {
        "tags": [TAG],
        "approved": False,
        "status": "clip-created",
        "clip_path": str(clip_path.resolve()),
        "source_video": source_video,
        "start_time": clip["start_time"],
        "duration": clip["duration"],
        "segment_labels": clip.get("segment_labels", []),
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
    """(path, frontmatter) for notes approved and waiting to be captioned."""
    results = []
    for path in find_notes(vault):
        frontmatter, _ = read_note(path)
        if frontmatter.get("approved") is True and frontmatter.get("status") == "pending-caption":
            results.append((path, frontmatter))
    return results
