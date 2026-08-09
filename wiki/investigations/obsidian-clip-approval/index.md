# Investigation: Obsidian-Based Clip Approval

**Status:** implemented.

## Goal

Give the [[clip-splitting]] pipeline a human review/approval step before clips get captioned and posted, using the existing Obsidian vault ticket-approval pattern (`approved` boolean frontmatter + a filtered Base) rather than inventing a new mechanism. Downstream, hand approved clips to [[caption-overlay]] to produce a captioned file, and leave uploading (already implemented elsewhere, out of scope) to pick up captioned output on its own.

## Why this shape

The vault already has a working approval pattern for tickets: `Tickets/*.md` notes with `tags: [ticket]` and `approved: false` frontmatter, filtered by `Bases/Needs Approval.base` (`approved == false`) for the human to review, and consumed by the `work-ticket` skill once flipped to `approved: true`. Reusing that pattern for clips means:

- No new review UI — Obsidian Bases already do table/card views with sort/filter.
- No new mental model for the user — same tag+frontmatter+Base shape as tickets.
- Clip metadata and human decisions (caption text, approval) live in a plain-text, git-independent store (the vault), separate from the repo's clip files and JSON manifests.

## Folder layout

Clips are **not** produced inside this git repo. `process_video.py`'s `folder` argument is meant to point at the video's real session folder — one per video, living in Google Drive under `Content/mcs/<video-folder>/` (confirmed by finding an existing session folder there whose video was byte-identical, by md5, to a copy that had been sitting in this repo for local dev). Layout:

```
Content/mcs/<video-folder>/
  <video>.MOV
  analysis/
    YYYY-MM-DD-<algorithm>.json               ← raw algorithm cache (beats, downbeats, segments, BPM)
    YYYY-MM-DD-<algorithm>_visual-cache.json   ← visual-filter resume cache (internal), named after its analysis file
  clips/
    clip-<start-timecode>-<end-timecode>.mp4   ← flat, one per clip, e.g. clip-00_46_56_600-00_47_27_000.mp4
    ...

Content/mcs/captioned-clips/                   ← shared, sibling to every video folder, not nested under one
  <video-folder>_clip-<start-timecode>-<end-timecode>.mp4
```

No `clips.json` manifest — every fact about a clip (timing, segment labels, algorithm/params provenance, a link to the analysis file it came from) lives entirely in that clip's Obsidian note; the note *is* the manifest entry. This was a deliberate simplification once the note-per-clip system existed: keeping both a JSON manifest and a note per clip was duplicate bookkeeping that could drift.

Clip filenames are derived from their start/end timecodes (`HH_MM_SS_mmm-HH_MM_SS_mmm`) rather than a sequential `clip-NNN` index — self-describing, and collision-free within a video without needing a run-name subfolder. This also makes reruns idempotent: regenerating the same boundaries just resolves to the same filename, so `process_video.py` skips re-rendering a clip whose file already exists, and skips creating a note if one already exists for that filename (a human may have already reviewed/approved it — a rerun must never clobber that).

Captioned output goes in a *shared* `captioned-clips/` folder outside every video folder (not a subfolder of any one video's `clips/`), since uploaders only need to watch one place across all videos. Its filenames are prefixed with the source video-folder's name to stay collision-free across videos sharing this one folder.

## Design

### 1. Note creation — hook in `process_video.py`

After clips are rendered in `process_video.py`, create one Obsidian note per clip in the vault at:

```
Music Clips/<video-folder-name>/clip-<start-timecode>-<end-timecode>.md
```

(Notes are grouped by video, matching the flat `clips/` folder — there's no run-name grouping anymore.)

Frontmatter:

```yaml
---
tags:
  - music-clip
approved: false
status: clip-created
clip_path: /absolute/path/to/Content/mcs/2025-10-30-mcs-practice/clips/clip-00_00_34_090-00_01_08_920.mp4
source_video: 2025-10-30-mcs-practice.MOV
analysis_path: /absolute/path/to/Content/mcs/2025-10-30-mcs-practice/analysis/2026-06-23-allin1.json
algorithm: allin1
algorithm_version: harmonix-all
params:
  beat_density_min: 1.2
  beat_cv_max: 0.25
  visual_filter_model: claude-haiku-4-5
start_time: 34.09
end_time: 68.92
duration: 34.83
segment_labels: [intro]
num_measures: 12
beat_density: 1.378
caption_text:
caption_emoji:
captioned_path:
---
```

- `clip_path` and `analysis_path` are absolute filesystem paths (not copied/symlinked into the vault — see "Video preview" below).
- `caption_text` / `caption_emoji` / `captioned_path` start blank and are filled in later (by the human during review, and by the captioning script respectively).
- `status` starts at `clip-created` and is the single source of truth for pipeline stage — kept independent of `approved` exactly like tickets separate `status` from `approved`.
- `algorithm` / `algorithm_version` / `params` replace what used to be top-level fields in `clips.json` — now per-note since there's no manifest to hold them once.

Body: a heading with the clip name, key facts (source video, timing, segment label) as plain text, and a video embed (see below) for review.

### 2. Video preview — plain file link, no copy/symlink into the vault

The vault lives under Google Drive (`~/Google Drive/obsidian/vault-1/`), same as the clips themselves (`Content/mcs/`) — but they're different Drive folders, and copying or symlinking every clip into the vault folder specifically would still push every clip's bytes through an extra round of Drive sync — rejected. Instead, embed the clip directly from its real path with a raw HTML5 video tag in the note body:

```html
<video src="file:///Users/.../Content/mcs/2025-10-30-mcs-practice/clips/clip-00_00_34_090-00_01_08_920.mp4" controls></video>
```

Obsidian renders raw HTML in notes and Electron's `file://` protocol can load local media directly, so this plays inline without any extra copying.

### 3. Review — human step, no new tooling

A new Base, `Bases/Music Clips - Needs Approval.base`, filtered like the ticket one but on the `music-clip` tag:

```yaml
filters:
  and:
    - file.hasTag("music-clip")
    - approved == false
```

The human opens each note from the Base, watches the embedded clip, fills in `caption_text` (and `caption_emoji` if wanted), then sets:

```yaml
approved: true
status: pending-caption
```

### 4. Captioning script — `caption-overlay/caption_pending_clips.py`

Run on demand (`uv run python caption-overlay/caption_pending_clips.py`) — not scheduled, since captioning is cheap/local and there's no queue to rate-limit like ticket work. Each run:

1. Scans the vault for notes tagged `music-clip` with `approved == true` and `status == pending-caption`.
2. For each, calls `add_caption_to_video()` (imported directly from `caption_overlay.py`, not shelled out) with `clip_path`/`caption_text`/`caption_emoji`.
3. Writes the result into the shared `captioned-clips/` folder (a sibling of every video folder, derived from `clip_path` as `clip_path.parent.parent.parent / "captioned-clips"`), named `<video-folder>_<clip-filename>` for cross-video uniqueness.
4. Sets the note's `captioned_path` to the new file's absolute path and `status: captioned`.

Processes the whole eligible batch per invocation rather than one at a time (unlike `work-ticket`, which picks a single oldest ticket) since there's no reason to serialize local ffmpeg runs.

### 5. Upload — explicitly out of scope

A separate process (already implemented elsewhere, not part of this repo) is expected to watch for `status: captioned` notes, upload `captioned_path`, and set `status: uploaded` on success. Nothing here should assume how that happens.

## Status lifecycle

| Status | Set by | Meaning |
|---|---|---|
| `clip-created` | `process_video.py` hook | Clip rendered, note created, awaiting review |
| `pending-caption` | Human, during review (alongside `approved: true`) | Approved with caption text filled in, awaiting captioning |
| `captioned` | Captioning script | Captioned file written, awaiting upload |
| `uploaded` | External upload process (out of scope) | Posted |

## Implementation

- `obsidian-clip-approval/vault_notes.py` — shared helper module: vault-path resolution (same dynamic `obsidian.json` lookup `create-ticket`/`work-ticket` use), frontmatter read/write, timecode filename generation (`to_timecode` / `clip_filename`), note creation, and querying for pending-caption notes. Shared rather than duplicated because both the note-creation side and the captioning-pickup side must agree on exact note shape and filename convention.
- `clip-splitting/process_video.py` — renders clips flat into `<folder>/clips/`, then creates one note per clip via `vault_notes.create_clip_note()`. Skippable with `--no-vault-notes`. No `clips.json` is written.
- `caption-overlay/caption_pending_clips.py` — the captioning script described in step 4 above; writes into the shared `captioned-clips/` folder.
- `Bases/Music Clips.base` (in the vault, not this repo) — two views: "Needs approval" (`hasTag("music-clip") && approved == false`, mirrors `Needs Approval.base`) and "By status" (grouped by `status`, mirrors `Tickets - Available.base`).

**Revision note:** the first implementation nested clips under `clips/<run-name>/` with a `clips.json` manifest, mirroring the run-oriented layout `process_video.py` already had. After building the note system, that turned out to be redundant — the note already carries everything the manifest did — so it was flattened to `clips/` with timecode-named files and no manifest, as described above. This also surfaced that `process_video.py` was never actually pointed at the real per-video folders in Google Drive; a one-off script migrated an existing legacy render (pre-dating `process_video.py` entirely) into this layout to backfill notes for it.

## Related

- [[caption-overlay]] — provides `caption_overlay.py`, the tool `caption_pending_clips.py` wraps.
- [[clip-splitting]] — the pipeline this hooks into (`process_video.py`).
