"""Tests for the clip notes' Google Drive links (TKT-139).

Run: uv run python -m unittest discover -s tests
Uses a fake Drive for Desktop database in a temp dir; never reads the real one.
"""

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "obsidian-clip-approval"))

import vault_notes  # noqa: E402
from backfill_drive_links import fill_note, main as backfill_main  # noqa: E402
from drive_links import DriveLinkError, DriveResolver, view_url  # noqa: E402


class FakeDrive:
    """A My Drive folder plus a minimal DriveFS account database describing it."""

    def __init__(self, base: Path):
        self.root = base / "My Drive"
        self.root.mkdir()
        self.drivefs = base / "DriveFS"
        account = self.drivefs / "123"
        account.mkdir(parents=True)
        self.mirror = sqlite3.connect(account / "mirror_sqlite.db")
        self.mirror.executescript(
            "CREATE TABLE root_config (local_stable_id INTEGER, is_my_drive INTEGER);"
            "CREATE TABLE mirror_item (local_stable_id INTEGER, stable_id INTEGER,"
            " parent_local_stable_id INTEGER, local_filename TEXT, inode INTEGER,"
            " local_size INTEGER, local_md5_checksum TEXT, cloud_md5_checksum TEXT);")
        self.meta = sqlite3.connect(account / "mirror_metadata_sqlite.db")
        self.meta.execute("CREATE TABLE items (stable_id INTEGER, id TEXT, trashed INTEGER)")
        self.next_id = 1
        self.ids = {self.root: self._add(self.root, None, synced=True)}
        self.mirror.execute("INSERT INTO root_config VALUES (?, 1)", (self.ids[self.root],))
        self.commit()

    def _add(self, path: Path, parent_id, synced: bool) -> int:
        local_id = self.next_id
        self.next_id += 1
        st = path.stat()
        md5 = f"md5-{local_id}"
        self.mirror.execute("INSERT INTO mirror_item VALUES (?,?,?,?,?,?,?,?)", (
            local_id, local_id + 1000, parent_id, path.name, st.st_ino,
            st.st_size if path.is_file() else 0, md5, md5 if synced else None))
        self.meta.execute("INSERT INTO items VALUES (?,?,0)", (local_id + 1000, f"drive-id-{local_id}"))
        return local_id

    def add_file(self, rel: str, synced: bool = True) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        for parent in reversed(path.relative_to(self.root).parents[:-1]):
            folder = self.root / parent
            if folder not in self.ids:
                self.ids[folder] = self._add(folder, self.ids[folder.parent], synced=True)
        path.write_bytes(b"video")
        self.ids[path] = self._add(path, self.ids[path.parent], synced)
        self.commit()
        return path

    def commit(self):
        self.mirror.commit()
        self.meta.commit()

    def resolver(self) -> DriveResolver:
        return DriveResolver(my_drive_root=self.root, drivefs_dir=self.drivefs)

    def url(self, path: Path) -> str:
        return view_url(f"drive-id-{self.ids[path]}")


NOTE = """---
tags:
- music-clip
status: uploaded
clip_path: {clip}
source_video: v.MOV
caption_emoji:  🌸
captioned_path: {captioned}
uploaded_url: https://youtube.com/shorts/x
buffer_posts:
  youtube: {{post_id: abc, status: sent}}
---
# clip

<video src="file://{clip}" controls></video>

## Captioned Preview

<video src="file://{captioned}" controls></video>
"""


class DriveLinkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name).resolve()
        self.drive = FakeDrive(base)
        self.clip = self.drive.add_file("Content/mcs/v/clips/clip-a.mp4")
        self.captioned = self.drive.add_file("Content/mcs/captioned-clips/v_clip-a.mp4")
        self.vault = base / "vault"
        self.note = self.vault / "Music Clips" / "v" / "clip-a.md"
        self.note.parent.mkdir(parents=True)
        self.note.write_text(NOTE.format(clip=self.clip, captioned=self.captioned))

    def tearDown(self):
        self.drive.mirror.close()
        self.drive.meta.close()
        self.tmp.cleanup()

    def test_resolves_file_id(self):
        self.assertEqual(self.drive.resolver().view_url_for(self.clip), self.drive.url(self.clip))

    def test_unsynced_file_is_an_error(self):
        path = self.drive.add_file("Content/mcs/v/clips/clip-b.mp4", synced=False)
        with self.assertRaisesRegex(DriveLinkError, "finished syncing"):
            self.drive.resolver().file_id(path)

    def test_unknown_and_outside_files_are_errors(self):
        stray = self.drive.root / "Content" / "untracked.mp4"
        stray.write_bytes(b"x")
        with self.assertRaisesRegex(DriveLinkError, "no record"):
            self.drive.resolver().file_id(stray)
        with self.assertRaisesRegex(DriveLinkError, "not inside My Drive"):
            self.drive.resolver().file_id(self.note)

    def test_backfill_adds_links_and_keeps_everything_else(self):
        before = self.note.read_text()
        result = fill_note(self.note, self.drive.resolver())
        self.assertTrue(result.changed)
        self.assertEqual(result.unresolved, {})
        text = self.note.read_text()
        fm, body = vault_notes.read_note(self.note)
        self.assertEqual(fm["clip_drive_url"], self.drive.url(self.clip))
        self.assertEqual(fm["captioned_drive_url"], self.drive.url(self.captioned))
        # Every original line is kept byte for byte; only the 2 frontmatter
        # fields and the 2 body link lines (each with a blank line) are new.
        removed = [l for l in before.split("\n") if l not in text.split("\n")]
        self.assertEqual(removed, [])
        self.assertEqual(len(text.split("\n")) - len(before.split("\n")), 6)
        self.assertIn("buffer_posts:\n  youtube: {post_id: abc, status: sent}\n", text)
        self.assertIn("caption_emoji:  🌸\n", text)
        self.assertIn(f"clip_path: {self.clip}\nclip_drive_url: ", text)
        self.assertIn(f"captioned_path: {self.captioned}\ncaptioned_drive_url: ", text)
        self.assertIn(f"[Open original in Google Drive]({self.drive.url(self.clip)})\n\n"
                      f'<video src="file://{self.clip}" controls></video>', body)
        self.assertIn(f"[Open captioned clip in Google Drive]({self.drive.url(self.captioned)})\n\n"
                      f'<video src="file://{self.captioned}" controls></video>', body)

    def test_backfill_adds_missing_captioned_section(self):
        text = self.note.read_text()
        self.note.write_text(text.split("\n## Captioned Preview")[0] + "\n")
        fill_note(self.note, self.drive.resolver())
        _, body = vault_notes.read_note(self.note)
        self.assertTrue(body.endswith(
            f"\n\n## Captioned Preview\n\n[Open captioned clip in Google Drive]"
            f"({self.drive.url(self.captioned)})\n\n"
            f'<video src="file://{self.captioned}" controls></video>\n'))
        once = self.note.read_text()
        self.assertFalse(fill_note(self.note, self.drive.resolver()).changed)
        self.assertEqual(self.note.read_text(), once)

    def test_backfill_is_idempotent(self):
        fill_note(self.note, self.drive.resolver())
        once = self.note.read_text()
        self.assertFalse(fill_note(self.note, self.drive.resolver()).changed)
        self.assertEqual(self.note.read_text(), once)

    def test_backfill_replaces_a_stale_link(self):
        fill_note(self.note, self.drive.resolver())
        text = self.note.read_text().replace(self.drive.url(self.clip), view_url("old-id"))
        self.note.write_text(text)
        self.assertTrue(fill_note(self.note, self.drive.resolver()).changed)
        text = self.note.read_text()
        self.assertNotIn("old-id", text)
        self.assertEqual(text.count("Open original in Google Drive"), 1)

    def test_dry_run_writes_nothing_and_reports_unresolved(self):
        unsynced = self.drive.add_file("Content/mcs/v/clips/clip-c.mp4", synced=False)
        other = self.note.parent / "clip-c.md"
        other.write_text(f"---\ntags:\n- music-clip\nclip_path: {unsynced}\ncaptioned_path:\n---\n"
                         f'<video src="file://{unsynced}" controls></video>\n')
        before = {p: p.read_text() for p in (self.note, other)}
        code = backfill_main(["--dry-run", "--quiet", "--vault", str(self.vault),
                              "--my-drive-root", str(self.drive.root),
                              "--drivefs-dir", str(self.drive.drivefs)])
        self.assertEqual(code, 1)  # clip-c is unresolved
        self.assertEqual({p: p.read_text() for p in before}, before)
        result = fill_note(other, self.drive.resolver(), write=False)
        self.assertIn("clip_path", result.unresolved)
        self.assertIn("finished syncing", result.unresolved["clip_path"][1])

    def test_new_note_records_link(self):
        clip = {"start_time": 1.0, "end_time": 2.0, "duration": 1.0}
        path = vault_notes.create_clip_note(
            self.vault, "v", clip, self.drive.add_file("Content/mcs/v/clips/clip-d.mp4"),
            "v.MOV", None, "allin1", "x", {}, clip_drive_url=view_url("abc"))
        fm, body = vault_notes.read_note(path)
        self.assertEqual(fm["clip_drive_url"], view_url("abc"))
        self.assertIsNone(fm["captioned_drive_url"])
        self.assertIn(f"[Open original in Google Drive]({view_url('abc')})\n\n<video", body)
        # Unresolved at creation: blank field, plain embed, filled in by the backfill.
        path2 = vault_notes.create_clip_note(
            self.vault, "v", clip, self.drive.add_file("Content/mcs/v/clips/clip-e.mp4"),
            "v.MOV", None, "allin1", "x", {})
        fm2, body2 = vault_notes.read_note(path2)
        self.assertIsNone(fm2["clip_drive_url"])
        self.assertNotIn("Google Drive", body2)
        self.assertTrue(fill_note(path2, self.drive.resolver()).changed)
        self.assertIn("Open original in Google Drive", path2.read_text())


    def test_captioning_job_records_captioned_link(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "caption-overlay"))
        import caption_pending_clips as job

        clip = self.drive.add_file("Content/mcs/v2/clips/clip-f.mp4")
        note = self.vault / "Music Clips" / "v2" / "clip-f.md"
        note.parent.mkdir(parents=True)
        note.write_text(f"---\ntags:\n- music-clip\nstatus: pending-caption\nclip_path: {clip}\n"
                        f"clip_drive_url:\ncaption_text: Hi\ncaption_emoji:\ncaptioned_path:\n"
                        f"captioned_drive_url:\n---\n# clip-f\n\n"
                        f'<video src="file://{clip}" controls></video>\n')

        def fake_caption(video_path, text, emoji, output_path):
            # Stands in for ffmpeg, and for Drive for Desktop syncing the result.
            self.drive.add_file(str(Path(output_path).relative_to(self.drive.root)))

        originals = (job.vault_notes.resolve_vault_path, job.add_caption_to_video, job.DriveResolver)
        job.vault_notes.resolve_vault_path = lambda: self.vault
        job.add_caption_to_video = fake_caption
        job.DriveResolver = self.drive.resolver
        try:
            job.caption_pending_clips(drive_wait=0)
        finally:
            job.vault_notes.resolve_vault_path, job.add_caption_to_video, job.DriveResolver = originals

        captioned = self.drive.root / "Content/mcs/captioned-clips/v2_clip-f.mp4"
        fm, body = vault_notes.read_note(note)
        self.assertEqual(fm["status"], "captioned")
        self.assertEqual(fm["captioned_path"], str(captioned))
        self.assertEqual(fm["captioned_drive_url"], self.drive.url(captioned))
        self.assertEqual(fm["clip_drive_url"], self.drive.url(clip))  # filled by the sweep
        self.assertIn(f"[Open captioned clip in Google Drive]({self.drive.url(captioned)})\n\n"
                      f'<video src="file://{captioned}" controls></video>', body)
        self.assertIn(f"[Open original in Google Drive]({self.drive.url(clip)})", body)


if __name__ == "__main__":
    unittest.main()
