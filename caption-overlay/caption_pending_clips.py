"""Caption every approved, pending clip found in the Obsidian vault.

Scans for #music-clip notes with `approved: true` and `status: pending-caption`
(set by the human during review — see
wiki/investigations/obsidian-clip-approval/index.md), runs caption_overlay.py
on each, and writes the result into a shared `captioned-clips/` folder that
sits alongside all the per-video session folders (not nested under any one
video), so uploaders only need to watch one place. Updates the note with the
captioned file's path and `status: captioned`. Upload (status: uploaded) is a
separate, out-of-scope process that picks up from there.

Usage:
    uv run python caption-overlay/caption_pending_clips.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "obsidian-clip-approval"))

import vault_notes
from caption_overlay import add_caption_to_video


def caption_pending_clips() -> int:
    vault = vault_notes.resolve_vault_path()
    pending = vault_notes.find_pending_caption_notes(vault)

    if not pending:
        print("No clips pending captioning.")
        return 0

    for note_path, frontmatter in pending:
        clip_path = Path(frontmatter["clip_path"])
        text = frontmatter.get("caption_text")
        if not text:
            print(f"Skipping {note_path.name}: approved but caption_text is blank")
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

        vault_notes.update_frontmatter(note_path, {
            "captioned_path": str(captioned_path.resolve()),
            "status": "captioned",
        })
        print(f"  -> {captioned_path} (note updated: status=captioned)")

    return len(pending)


if __name__ == "__main__":
    n = caption_pending_clips()
    print(f"\nDone! Captioned {n} clip(s).")
