"""Add Google Drive links for the original and captioned files to clip notes.

For every #music-clip note under the vault's `Music Clips/` folder, looks up
the Drive file behind its `clip_path` and `captioned_path` (see
drive_links.py: read-only, from Drive for Desktop's local database) and:

- sets `clip_drive_url` / `captioned_drive_url` in the frontmatter, editing
  just that line so nothing else in the note is reformatted, and
- puts an "Open ... in Google Drive" link line right above the matching
  `file://` video embed in the body. The embeds themselves stay, for inline
  review on this Mac. A note captioned before the captioned-preview embed
  existed gets the same `## Captioned Preview` section (link plus embed)
  that the captioning job writes now.

Idempotent: a note whose links are already right is left untouched, so it's
safe to re-run (the captioning job runs the same pass after each batch).
Files that don't resolve yet (not synced, not in My Drive, missing) are
listed with the reason and left blank; a later run fills them in.

Usage:
    uv run python obsidian-clip-approval/backfill_drive_links.py --dry-run
    uv run python obsidian-clip-approval/backfill_drive_links.py
    uv run python obsidian-clip-approval/backfill_drive_links.py --vault ~/vault-sandbox

Exit status: 0 when every file resolved, 1 if any were left unresolved
(the notes that did resolve are still written), 2 on a usage error.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import vault_notes
from drive_links import DriveResolver, try_view_url

CAPTIONED_HEADING = "Captioned Preview"  # as caption_pending_clips.py writes it


@dataclass
class NoteResult:
    path: Path
    changed: bool = False
    links: dict = field(default_factory=dict)       # Drive field -> url
    unresolved: dict = field(default_factory=dict)  # path field -> (local path, reason)


def fill_note(path: Path, resolver: DriveResolver, write: bool = True) -> NoteResult:
    """Fill in one note's Drive links. Writes only if something changed and `write`."""
    result = NoteResult(path)
    original = path.read_text()
    frontmatter, _ = vault_notes.read_note(path)
    text = original
    for path_field, (drive_field, link_text) in vault_notes.DRIVE_LINK_FIELDS.items():
        local = frontmatter.get(path_field)
        if not local:
            continue  # e.g. not captioned yet
        url, reason = try_view_url(local, resolver)
        if url is None:
            result.unresolved[path_field] = (local, reason)
            continue
        result.links[drive_field] = url
        if frontmatter.get(drive_field) != url:
            text = vault_notes.set_frontmatter_text_field(text, drive_field, url, after_key=path_field)
        match = vault_notes.FRONTMATTER_RE.match(text)
        new_body = vault_notes.set_body_drive_link(match.group(2), str(local), url, link_text)
        embed = f'<video src="file://{local}" controls></video>'
        if path_field == "captioned_path" and embed not in new_body:
            # Captioned before the preview embed existed (TKT-12): add the
            # section new captioning runs write, so the link has a home.
            new_body = (new_body.rstrip("\n") + f"\n\n## {CAPTIONED_HEADING}\n\n"
                        + f"[{link_text}]({url})\n\n{embed}\n")
        text = f"---\n{match.group(1)}\n---\n{new_body}"
    if text != original:
        result.changed = True
        if write:
            path.write_text(text)
    return result


def fill_notes(paths: list[Path], resolver: DriveResolver, write: bool = True) -> list[NoteResult]:
    return [fill_note(p, resolver, write) for p in paths]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without writing any note")
    parser.add_argument("--vault", type=Path,
                        help="vault root (default: the open Obsidian vault)")
    parser.add_argument("--my-drive-root", type=Path, default=Path.home() / "Google Drive",
                        help="where Drive for Desktop mirrors My Drive (default: ~/Google Drive)")
    parser.add_argument("--drivefs-dir", type=Path,
                        default=Path.home() / "Library" / "Application Support" / "Google" / "DriveFS",
                        help="Drive for Desktop's data folder (default: the standard location)")
    parser.add_argument("--quiet", action="store_true", help="only print the summary and problems")
    args = parser.parse_args(argv)

    vault = (args.vault.expanduser() if args.vault else vault_notes.resolve_vault_path())
    if not (vault / vault_notes.NOTES_SUBDIR).is_dir():
        print(f"error: no {vault_notes.NOTES_SUBDIR}/ folder in {vault}", file=sys.stderr)
        return 2

    resolver = DriveResolver(my_drive_root=args.my_drive_root, drivefs_dir=args.drivefs_dir)
    notes = sorted(vault_notes.find_notes(vault))
    results = fill_notes(notes, resolver, write=not args.dry_run)

    verb = "would update" if args.dry_run else "updated"
    changed = [r for r in results if r.changed]
    unresolved = [(r, f, v) for r in results for f, v in r.unresolved.items()]
    if not args.quiet:
        for r in changed:
            print(f"{verb}: {r.path.relative_to(vault)} ({', '.join(sorted(r.links))})")
    for r, path_field, (local, reason) in unresolved:
        print(f"unresolved: {r.path.relative_to(vault)} {path_field}: {reason}")

    print(f"\n{len(notes)} clip note(s): {len(changed)} {verb}, "
          f"{len(notes) - len(changed)} already up to date or nothing to add, "
          f"{len(unresolved)} file(s) unresolved"
          + (" (dry run, nothing written)" if args.dry_run else ""))
    return 1 if unresolved else 0


if __name__ == "__main__":
    sys.exit(main())
