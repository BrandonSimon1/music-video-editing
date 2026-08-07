# Investigation: Obsidian-Based Clip Approval

**Status:** implemented.

## Goal

Give the [[clip-splitting]] pipeline a human review/approval step before clips get captioned and posted, using the existing Obsidian vault ticket-approval pattern (`approved` boolean frontmatter + a filtered Base) rather than inventing a new mechanism. Downstream, hand approved clips to [[caption-overlay]] to produce a captioned file, and leave uploading (already implemented elsewhere, out of scope) to pick up captioned output on its own.

## Why this shape

The vault already has a working approval pattern for tickets: `Tickets/*.md` notes with `tags: [ticket]` and `approved: false` frontmatter, filtered by `Bases/Needs Approval.base` (`approved == false`) for the human to review, and consumed by the `work-ticket` skill once flipped to `approved: true`. Reusing that pattern for clips means:

- No new review UI — Obsidian Bases already do table/card views with sort/filter.
- No new mental model for the user — same tag+frontmatter+Base shape as tickets.
- Clip metadata and human decisions (caption text, approval) live in a plain-text, git-independent store (the vault), separate from the repo's clip files and JSON manifests.

## Design

### 1. Note creation — hook in `process_video.py`

After `clips.json` is written in `process_video.py` (right after the final `clips` list and `run_dir` are known — see `render_clips_mp4` / the `output` dict construction), create one Obsidian note per rendered clip in the vault at:

```
Music Clips/<run-name>/clip-NNN.md
```

(`<run-name>` matches the repo's own `clips/<run-name>/` folder name, e.g. `2026-07-14-allin1`, so a note and its clip file are trivially correlated by path.)

Frontmatter:

```yaml
---
tags:
  - music-clip
approved: false
status: clip-created
clip_path: /absolute/path/to/session-folder/clips/2026-07-14-allin1/clip-001.mp4
source_video: 2025-10-30-mcs-practice.MOV
start_time: 53.81
duration: 33.42
segment_labels: [verse]
caption_text:
caption_emoji:
captioned_path:
---
```

- `clip_path` is an absolute filesystem path (not copied/symlinked into the vault — see "Video preview" below).
- `caption_text` / `caption_emoji` / `captioned_path` start blank and are filled in later (by the human during review, and by the captioning script respectively).
- `status` starts at `clip-created` and is the single source of truth for pipeline stage — kept independent of `approved` exactly like tickets separate `status` from `approved`.

Body: a heading with the clip name, key facts (source video, timing, segment label) as plain text, and a video embed (see below) for review.

### 2. Video preview — plain file link, no copy/symlink into the vault

The vault lives under Google Drive (`~/Google Drive/obsidian/vault-1/`), so copying or symlinking every clip into it would push every clip's bytes through Drive sync — rejected. Instead, embed the clip directly from its repo-local path with a raw HTML5 video tag in the note body:

```html
<video src="file:///Users/.../clips/2026-07-14-allin1/clip-001.mp4" controls></video>
```

Obsidian renders raw HTML in notes and Electron's `file://` protocol can load local media directly, so this plays inline without touching the vault's synced storage.

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
3. Writes the result into a new `captioned/` subfolder inside that clip's session folder (i.e. sibling to `clips/<run-name>/`, so `session-folder/clips/<run-name>/captioned/clip-001.mp4`).
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

- `obsidian-clip-approval/vault_notes.py` — shared helper module (vault-path resolution via the same dynamic `obsidian.json` lookup `create-ticket`/`work-ticket` use, frontmatter read/write, note creation, and querying for pending-caption notes). Shared rather than duplicated because both the note-creation side and the captioning-pickup side must agree on exact note shape.
- `clip-splitting/process_video.py` — hook added after `clips.json` is written; creates one note per rendered clip via `vault_notes.create_clip_note()`. Skippable with `--no-vault-notes`.
- `caption-overlay/caption_pending_clips.py` — the captioning script described in step 4 above.
- `Bases/Music Clips.base` (in the vault, not this repo) — two views: "Needs approval" (`hasTag("music-clip") && approved == false`, mirrors `Needs Approval.base`) and "By status" (grouped by `status`, mirrors `Tickets - Available.base`).

## Related

- [[caption-overlay]] — provides `caption_overlay.py`, the tool `caption_pending_clips.py` wraps.
- [[clip-splitting]] — the pipeline this hooks into (`process_video.py`).
