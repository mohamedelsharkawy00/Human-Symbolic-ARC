# ARC Notes

> ## ⟢ NOW — read this the moment you start drifting
>
> **The direction is RIGHT. Stop relitigating it.** The order is fixed and correct:
>
> 1. Hand-write the DSL solution for **ONE** eval task.
> 2. Mark it `expressible` / `inexpressible`.
> 3. Repeat for a small batch.
> 4. **ONLY THEN** touch solver optimizations.
>
> Hand-writing solutions first **IS** the verification — it is the load-bearing
> step, not procrastination. Eval is capability-limited (confirmed: not a
> pipeline-zero), so *expressibility* is the exact unknown to resolve, and the
> only way to resolve it is by hand. You are not going in the wrong direction.
>
> - **ONE TASK AT A TIME.** The whole job is the task in front of you — not the
>   plan for all of them. Write each verdict to disk immediately. Drift happens
>   because nothing is anchored and anxiety fills the gap; the anchor is the file.
> - **NO RESUBMITTING.** Same solver → same score. A resubmit touches nothing —
>   it's a control ritual / anxiety discharge, not work (the same loop as
>   restarting the terminal over a cosmetic warning). Off the table until the DSL
>   audit actually changes the solver.
> - **THE CEILING IS REAL AND NOT YOUR FAULT.** Forward DSL beam search is the
>   Icecuber family: ~20% on v1 eval, ~0 on v2 eval **by design**. ARC-AGI-2 was
>   built to defeat shallow-composition search. No amount of more-beam / more-GPU
>   moves a v1-shaped search to v2-competitive — true for *everyone* with this
>   approach. **20% on v2 *training* with a hand-built CUDA solver is respectable.**
> - **MEASURE AGAINST OFF-ZERO, NOT WINNING.** With this engine + this deadline,
>   "win v2" was never on the table for anyone. Real targets: 0 → a few eval
>   solves; ship a clean working 20%-train solver; bank the design thinking.
> - **THE ONE DEADLINE-FEASIBLE LEVER:** output-first pruning
>   (`Backward Deduction Design`, §5 option 1). *Narrower* search, not smarter —
>   mask the op-set + seed the beam per puzzle from the output type-signature so
>   search reaches deeper on the **expressible-but-unfound** eval tasks. It only
>   helps tasks the DSL can already express — which is exactly what the hand-write
>   audit identifies. **The audit feeds the lever.** That's why step 1 comes first.

These notes are browsed as a small MkDocs (Material) site.
**Quick rule: double-click `build_notes.bat` — it opens a live preview in the browser.**

- **To read or write** — double-click `scripts\build_notes.bat`. It starts a local preview server and opens the browser to `http://127.0.0.1:8000/Home.html`. Edit any `.md` file in a numbered folder and save; the browser auto-refreshes — no rebuild, no manual reload. Search works in this mode (it needs the server). Keep that terminal window open while browsing; close it (or Ctrl+C) to stop the server.
- **How it works** — MkDocs reads this folder directly (no copy step) and renders into `..\out\notes_site\`, a build folder regenerated each run (safe to delete). Your notes are never modified by the build.
- **Start here** — the *Research Synthesis* note (the canonical strategy doc).
- **Titles** — sidebar labels are the real file and folder names (e.g. `Backward Deduction Design`, `1 - Overall Design`), set by `.mkdocs/title_hook.py` (`on_page_markdown` for files, `on_nav` for folders), so the outline mirrors the tree regardless of what heading a file starts with. Rename a file or folder and its label updates on the next build.

## What's where

- `1 - Overall Design/` … `6 - Handoffs/` — the notes (most recent date in `6 - Handoffs/` = current state).
- `5 - Competition/concept_sources/` and `nvarc_reference/` — external prior art, **not** rendered in the site, browse directly. Reference only, never paste into the submission. See the TODO below.
- `.mkdocs/` — toolchain: `mkdocs.yml` (config), `title_hook.py` (label naming), `.venv/`.
- `scripts\build_notes.bat` — live preview server (auto-refresh on save + search); renders into `out/notes_site/`.
- `stylesheets/extra.css` — layout tweaks (full-width grid, wide nav, no-clip labels).
- `javascripts/autosize_nav.js` — auto-sizes the nav pane to the longest label, once per load.

## TODO — vendored prior art to remove later

The only nested folders are the external reference dumps inside `5 - Competition/`:

- `5 - Competition/concept_sources/` (and its sub-folders: `agemo_arcsolver`, `arga_graph_dsl`, `barc_seeds`, `bidir_synth_alford`, `hodel_arc_dsl`, `icecuber_arc_solution`, `madil_arc_mdl`, `pearl_dreamcoder_arc`)
- `5 - Competition/nvarc_reference/`

    These are NOT my notes — they're downloaded third-party prior art (reference only, licensed MIT/Apache/research; never paste into the CC0/MIT-0 submission).

    When removing them:

    1. Delete the two folders above.
    2. Remove their two lines from `exclude_docs` in `.mkdocs/mkdocs.yml` (they're only excluded because they exist; once gone, the excludes are dead config).
    3. Rebuild. Everything else in the notes tree is flat (one level deep) and genuinely mine.
