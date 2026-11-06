"""Caption every approved, pending clip found in the Obsidian vault.

Scans for #music-clip notes without the `needs-approval` tag and with
`status: pending-caption`
(set by the human during review — see
wiki/investigations/obsidian-clip-approval/index.md), runs caption_overlay.py
on each, and writes the result into a shared `captioned-clips/` folder that
sits alongside all the per-video session folders (not nested under any one
video), so uploaders only need to watch one place. Updates the note with the
captioned file's path and `status: captioned`, and appends a second video
embed to the note body ("Captioned Preview") so the captioned output can be
checked for caption placement/sizing/timing right from the vault, alongside
the original clip embed, before it gets uploaded anywhere. Upload
(status: uploaded) is a separate, out-of-scope process that picks up from
there.

It also records Google Drive links (TKT-139): `captioned_drive_url` for each
new captioned file, with a link line above its preview embed. A file that
was just written hasn't synced to Drive yet, so after the batch the script
waits up to --drive-wait seconds for the new files to sync, then runs
backfill_drive_links' pass over every clip note, filling in any link
(original or captioned) that has become resolvable since. Links that still
don't resolve are listed and left blank for the next run.

Usage:
    uv run python caption-overlay/caption_pending_clips.py [--drive-wait SECONDS]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "obsidian-clip-approval"))

import vault_notes
from backfill_drive_links import fill_notes
from caption_overlay import add_caption_to_video
from drive_links import DriveResolver, try_view_url


def caption_pending_clips(drive_wait: float = 120) -> int:
    vault = vault_notes.resolve_vault_path()
    pending = vault_notes.find_pending_caption_notes(vault)
    resolver = DriveResolver()
    captioned_files: list[Path] = []

    if not pending:
        print("No clips pending captioning.")
        fill_drive_links(vault, resolver)
        return 0

    for note_path, frontmatter in pending:
        clip_path = Path(frontmatter["clip_path"])
        text = frontmatter.get("caption_text")
        if not text:
            print(f"Skipping {note_path.name}: approved (needs-approval tag removed) but caption_text is blank")
            continue

        # clip_path is <mcs-root>/<video-folder>/clips/clip-....mp4. The
        # video-folder name prefixes the output filename so clips from
        # different videos can't collide in the shared captioned-clips/ dir.
        video_folder = clip_path.parent.parent
        mcs_root = video_folder.parent
        captioned_dir = mcs_root / "captioned-clips"
        captioned_dir.mkdir(exist_ok=True)
        captioned_path = captioned_dir / f"{video_folder.name}_{clip_path.name}"

        print(f"Captioning {clip_path.name}: \"{text}\" {frontmatter.get('caption_emoji') or ''}")
        add_caption_to_video(
            video_path=clip_path,
            text=text,
            emoji=frontmatter.get("caption_emoji"),
            output_path=captioned_path,
        )

        captioned_drive_url, _ = try_view_url(captioned_path, resolver)
        vault_notes.update_frontmatter(note_path, {
            "captioned_path": str(captioned_path.resolve()),
            vault_notes.CAPTIONED_DRIVE_FIELD: captioned_drive_url,
            "status": "captioned",
        })
        vault_notes.append_video_embed(
            note_path, "Captioned Preview", captioned_path,
            drive_url=captioned_drive_url,
            link_text=vault_notes.DRIVE_LINK_FIELDS["captioned_path"][1],
        )
        captioned_files.append(captioned_path)
        print(f"  -> {captioned_path} (note updated: status=captioned, captioned preview embedded)")

    wait_for_drive_sync(captioned_files, resolver, drive_wait)
    fill_drive_links(vault, resolver)
    return len(pending)


def wait_for_drive_sync(paths: list[Path], resolver: DriveResolver, timeout: float,
                        interval: float = 10) -> None:
    """Give Drive for Desktop up to `timeout` seconds to upload the new files,
    so their links can go into the notes in this same run."""
    waiting = [p for p in paths if try_view_url(p, resolver)[0] is None]
    if not waiting or timeout <= 0:
        return
    print(f"\nWaiting up to {timeout:.0f}s for {len(waiting)} captioned file(s) to sync to Google Drive...")
    deadline = time.monotonic() + timeout
    while waiting and time.monotonic() < deadline:
        time.sleep(interval)
        waiting = [p for p in waiting if try_view_url(p, resolver)[0] is None]
    if waiting:
        print(f"  {len(waiting)} still not synced; their Drive links will be filled in on a later run")


def fill_drive_links(vault: Path, resolver: DriveResolver) -> None:
    """Fill in any missing or stale Drive links across all clip notes."""
    results = fill_notes(sorted(vault_notes.find_notes(vault)), resolver)
    changed = [r for r in results if r.changed]
    unresolved = [(r, f, v) for r in results for f, v in r.unresolved.items()]
    print(f"\nGoogle Drive links: {len(changed)} note(s) updated, {len(unresolved)} file(s) unresolved")
    for r in changed:
        print(f"  updated: {r.path.relative_to(vault)}")
    for r, path_field, (_, reason) in unresolved:
        print(f"  unresolved: {r.path.relative_to(vault)} {path_field}: {reason}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Caption approved, pending clips from the vault.")
    parser.add_argument("--drive-wait", type=float, default=120,
                        help="seconds to wait for new captioned files to sync to Google Drive "
                             "before recording their links (default 120; 0 to skip waiting)")
    args = parser.parse_args()
    n = caption_pending_clips(drive_wait=args.drive_wait)
    print(f"\nDone! Captioned {n} clip(s).")
