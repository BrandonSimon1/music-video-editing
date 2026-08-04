# Investigation: Obsidian-Based Clip Approval

**Status:** design approved-pending — not yet implemented. This doc is the spec to build against once the user signs off.

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

### 4. Captioning script — new, in this repo

A script near `caption-overlay/` (not yet named/written) that:

1. Scans the vault for notes tagged `music-clip` with `approved == true` and `status == pending-caption`.
2. For each, calls `caption_overlay.py add` with `--video <clip_path> --text <caption_text> --emoji <caption_emoji>`.
3. Writes the result into a new `captioned/` subfolder inside that clip's session folder (i.e. sibling to `clips/<run-name>/`, so `session-folder/clips/<run-name>/captioned/clip-001.mp4`).
4. Sets the note's `captioned_path` to the new file's absolute path and `status: captioned`.

This mirrors how `work-ticket` scans for `approved == true` + a specific `status` and picks the oldest eligible one — except this script processes the whole eligible batch per run rather than one at a time, since captioning is cheap/local (no reason to rate-limit it like ticket work).

### 5. Upload — explicitly out of scope

A separate process (already implemented elsewhere, not part of this repo) is expected to watch for `status: captioned` notes, upload `captioned_path`, and set `status: uploaded` on success. Nothing here should assume how that happens.

## Status lifecycle

| Status | Set by | Meaning |
|---|---|---|
| `clip-created` | `process_video.py` hook | Clip rendered, note created, awaiting review |
| `pending-caption` | Human, during review (alongside `approved: true`) | Approved with caption text filled in, awaiting captioning |
| `captioned` | Captioning script | Captioned file written, awaiting upload |
| `uploaded` | External upload process (out of scope) | Posted |

## Open items for implementation (not decided here)

- Exact vault-path resolution mechanism for the `process_video.py` hook and the captioning script — should reuse the same dynamic `obsidian.json` lookup the `create-ticket`/`work-ticket` skills use, rather than hardcoding the vault path.
- Note template / exact body layout beyond the video embed and frontmatter.
- Whether the captioning script runs on demand (manually invoked) or on a schedule.

## Related

- [[caption-overlay]] — provides `caption_overlay.py`, the tool the captioning script wraps.
- [[clip-splitting]] — the pipeline this hooks into (`process_video.py`).
