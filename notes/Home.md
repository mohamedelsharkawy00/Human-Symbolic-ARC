# ARC Notes

> ## ⟢ NOW — read this the moment you start drifting
>
> - **The order, and the one puzzle in front of you** — walk the training set in
>   order, finish each puzzle before starting the next, and take the steps inside
>   one in order: [Method Order Reminder](<4 - To Do/Method Order Reminder.md>).
> - **Why the ceiling sits where it does, and what to measure against** — where
>   everyone else landed is in [Human Work Landscape](<5 - Competition/Human Work Landscape.md>),
>   and the reasoning is in [Research Synthesis](<5 - Competition/Research Synthesis.md>).
> - **The one lever that fits the deadline** — §5 of
>   [Backward Deduction Design](<1 - Overall Design/Backward Deduction Design.md>).
> - **A resubmit is not work** — §11.3 of
>   [Research Synthesis](<5 - Competition/Research Synthesis.md>).

## Marked for removal

- **The greedy fallback in matching** — `fall_back_to_greedy` in
  `include/core/5_matching.hpp`, run whenever the fitting ends without an exact
  cover. Marked 2026-09-24, after 09915ca let the fitting finish instead of
  spending its budget on the first look.
  - **It still decides a lot**: 54% of elected pairs across the training set, 73%
    of every reading's pairs. Nearly all are *no fit* (no exact cover exists),
    not *gave up* — so deciding what a no-fit pair reports comes before removing it.
  - **It credits what an exact cover never would**: it keeps any match that adds
    one new cell, so overlapping and half-hidden placements count. 10 of the 60
    elections that moved in 09915ca moved because a winner lost exactly that credit.
  - **It is where 09915ca's slowdown lives**: its sort recomputes every match's
    footprint in every comparison, and that time doubled.

These notes are browsed as a small MkDocs (Material) site.
**Quick rule: double-click `build_notes.bat` — it opens a live preview in the browser.**

- **To read or write** — double-click `scripts\build_notes.bat`. It starts a local preview server and opens the browser to `http://127.0.0.1:8000/Home.html`. Edit any `.md` file in a numbered folder and save; the browser auto-refreshes — no rebuild, no manual reload. Search works in this mode (it needs the server). Keep that terminal window open while browsing; close it (or Ctrl+C) to stop the server.
- **How it works** — MkDocs reads this folder directly (no copy step) and renders into `..\out\notes_site\`, a build folder regenerated each run (safe to delete). Your notes are never modified by the build.
- **Start here** — the *Research Synthesis* note (the canonical strategy doc).
- **Titles** — sidebar labels are the real file and folder names (e.g. `Backward Deduction Design`, `1 - Overall Design`), set by `.mkdocs/title_hook.py` (`on_page_markdown` for files, `on_nav` for folders), so the outline mirrors the folders regardless of what heading a file starts with. Rename a file or folder and its label updates on the next build.

## What's where

- `1 - Overall Design/` … `6 - Handoffs/` — the notes, one level deep, nothing nests (most recent date in `6 - Handoffs/` = current state).
- **Prior art lives OUTSIDE the repo** — `concept_sources/` and `nvarc_reference/` moved to `Desktop\ARC Extras\` on 2026-09-05. Third-party, reference only, never paste into the submission. Other notes name them; this is the only line that says where they are. Nothing third-party nests in the numbered folders any more, so there is nothing here to step over when reading or searching.
- `.mkdocs/` — toolchain: `mkdocs.yml` (config), `title_hook.py` (label naming). No venv: mkdocs is installed on the machine Python.
- `scripts\build_notes.bat` — live preview server (auto-refresh on save + search); renders into `out/notes_site/`.
- `stylesheets/extra.css` — layout tweaks (full-width grid, wide nav, no-clip labels).
- `javascripts/autosize_nav.js` — auto-sizes the nav pane to the longest label, once per load.

