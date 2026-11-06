"""Google Drive links for clip files, from Drive for Desktop's local database.

The clips (and their captioned versions) live in My Drive, which Google
Drive for Desktop mirrors to ~/Google Drive. A clip note's `clip_path` and
`captioned_path` are local paths into that mirror, so they only resolve on
this Mac. This module turns such a path into the file's Drive link, so the
note also works from anywhere Brandon is signed in to Drive.

No Drive API, OAuth or network: the file ID comes from Drive for Desktop's
own SQLite databases under
~/Library/Application Support/Google/DriveFS/<account id>/, opened
read-only. mirror_sqlite.db maps each local file to a `stable_id` (table
mirror_item, keyed by parent and file name), and mirror_metadata_sqlite.db
maps `stable_id` to the Drive file ID (table items).

The lookup is a port of `GoogleDriveMediaHost.file_id` in
~/repos/magic-carpet-sessions/uploads/media_host.py, which the Buffer upload
job uses to host captioned clips. The two repos don't depend on each other,
so it's copied here rather than imported; keep the two in step if Drive for
Desktop's schema changes. These databases are internal to Drive for Desktop,
not a public API, so every lookup is cross-checked (the My Drive folder's
inode, the file's inode and size, and that the local and cloud checksums
match, i.e. the file has finished syncing) and raises DriveLinkError rather
than guessing.
"""

from __future__ import annotations

import sqlite3
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

# The normal "open in Drive" link. For a private file it needs a signed-in
# Google account with access, which is the point: it works for Brandon from
# any device. (It's an HTML viewer page, not the file's bytes, so it can't be
# a <video src>.)
DRIVE_VIEW_URL = "https://drive.google.com/file/d/{id}/view"


class DriveLinkError(RuntimeError):
    """The path couldn't be resolved to a Drive file (yet)."""


def view_url(file_id: str) -> str:
    return DRIVE_VIEW_URL.format(id=file_id)


@dataclass
class DriveResolver:
    my_drive_root: Path = field(default_factory=lambda: Path.home() / "Google Drive")
    drivefs_dir: Path = field(default_factory=lambda: (
        Path.home() / "Library" / "Application Support" / "Google" / "DriveFS"))

    def view_url_for(self, path: Path) -> str:
        return view_url(self.file_id(path))

    def file_id(self, path: Path) -> str:
        root = self.my_drive_root.expanduser().resolve()
        target = Path(path).expanduser().resolve()
        try:
            parts = target.relative_to(root).parts
        except ValueError as exc:
            raise DriveLinkError(f"{target} is not inside My Drive ({root})") from exc
        if not parts:
            raise DriveLinkError(f"{target} is the My Drive folder itself, not a file")
        try:
            root_ino = root.stat().st_ino
            file_stat = target.stat()
        except OSError as exc:
            raise DriveLinkError(f"can't stat {exc.filename}: {exc.strerror}") from exc

        account = self._account_for_root(root_ino)
        if account is None:
            raise DriveLinkError(
                f"no Drive for Desktop account under {self.drivefs_dir} mirrors {root}. "
                "Is Google Drive for Desktop installed, signed in, and set to mirror My Drive "
                "to that folder?")
        account_dir, root_local_id = account

        with _open_ro(account_dir / "mirror_sqlite.db") as db:
            local_id = root_local_id
            row = None
            for name in parts:
                row = db.execute(
                    "SELECT local_stable_id, stable_id, inode, local_size, local_md5_checksum, "
                    "cloud_md5_checksum FROM mirror_item "
                    "WHERE parent_local_stable_id = ? AND local_filename = ?",
                    (local_id, name)).fetchone()
                if row is None:
                    raise DriveLinkError(f"Drive for Desktop has no record of {target} "
                                         "(not synced to Drive yet?)")
                local_id = row[0]
        _, stable_id, inode, size, local_md5, cloud_md5 = row
        if inode != file_stat.st_ino or size != file_stat.st_size:
            raise DriveLinkError(f"Drive for Desktop's record of {target} doesn't match the file "
                                 "on disk (changed since it last synced?); try again later")
        if not local_md5 or local_md5 != cloud_md5:
            raise DriveLinkError(f"{target} hasn't finished syncing to Google Drive yet")

        with _open_ro(account_dir / "mirror_metadata_sqlite.db") as db:
            item = db.execute("SELECT id, trashed FROM items WHERE stable_id = ?",
                              (stable_id,)).fetchone()
        if item is None or not item[0]:
            raise DriveLinkError(f"no Drive file ID recorded for {target}")
        if item[1]:
            raise DriveLinkError(f"{target} is in the Drive trash")
        return item[0]

    def _account_for_root(self, root_ino: int) -> tuple[Path, int] | None:
        """Find the account whose mirrored My Drive root folder is this inode."""
        if not self.drivefs_dir.is_dir():
            return None
        for db_path in sorted(self.drivefs_dir.glob("*/mirror_sqlite.db")):
            try:
                with _open_ro(db_path) as db:
                    row = db.execute(
                        "SELECT r.local_stable_id FROM root_config r "
                        "JOIN mirror_item m ON m.local_stable_id = r.local_stable_id "
                        "WHERE r.is_my_drive = 1 AND m.inode = ?", (root_ino,)).fetchone()
            except (sqlite3.Error, DriveLinkError):
                continue
            if row:
                return db_path.parent, row[0]
        return None


class _open_ro:
    """Read-only SQLite connection that is closed on exit (sqlite3's own context
    manager only commits)."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self) -> sqlite3.Connection:
        if not self.path.is_file():
            raise DriveLinkError(f"Drive for Desktop database not found: {self.path}")
        uri = "file:" + urllib.parse.quote(str(self.path)) + "?mode=ro"
        try:
            self.conn = sqlite3.connect(uri, uri=True, timeout=10)
        except sqlite3.Error as exc:
            raise DriveLinkError(f"can't open {self.path}: {exc}") from exc
        return self.conn

    def __exit__(self, *exc) -> None:
        self.conn.close()


def try_view_url(path: Path | str | None, resolver: DriveResolver | None = None) -> tuple[str | None, str | None]:
    """(url, None) if the path resolves, else (None, reason). Never raises for a
    path that just isn't in Drive yet, so callers can leave the link blank and
    let a later backfill run fill it in."""
    if not path:
        return None, "no path"
    resolver = resolver or DriveResolver()
    try:
        return resolver.view_url_for(Path(path)), None
    except DriveLinkError as exc:
        return None, str(exc)
    except sqlite3.Error as exc:
        return None, f"Drive for Desktop database error: {exc}"
