# Contest Rules — what the rules actually constrain, and how to re-read them

> Read and summarised 2026-08-15 from the live pages. Not legal advice; re-read before submitting.
> Section numbers below are Kaggle's own numbering.

## Where the rules live (two sources, and they disagree)

1. Kaggle, the binding one: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules
2. ARC Prize, the plainer one: https://arcprize.org/competitions/2026

They give DIFFERENT license answers. See "The license question" below.

## How to read the Kaggle page (for the next session)

Plain fetching FAILS. Kaggle ships a ~5.5 KB shell and builds the text in the browser, so an
HTTP GET returns no rules at all. Render it with headless Chrome instead:

```
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --headless=new --disable-gpu `
  --virtual-time-budget=20000 --dump-dom `
  "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules"
```

That returns ~75,000 characters of rendered DOM. Strip tags, keep lines, read. No login needed —
the rules are public even though the page looks gated. The same trick works for any
JavaScript-only page.

## What actually constrains this project

1. **Never publish the competition data.** (§2.4.b) The agreement is not to transmit, duplicate,
   publish, or redistribute Competition Data to anyone not in the competition. `kaggle/input/`
   holds exactly that. It is excluded in `.gitignore` — keep it excluded, permanently.

2. **No private code sharing during the competition.** (§3.6.a) Sharing solver code privately
   with anyone outside an official Kaggle team is a breach and grounds for disqualification.
   A solo private repo is fine because nobody else can see it. Do NOT add collaborators to the
   GitHub repo.

3. **Public sharing has a required venue.** (§3.6.b) Code shared publicly must be shared on
   Kaggle — the competition's discussion forum or notebooks. A GitHub repo alone is not what the
   rule describes. When going public, post it on Kaggle too.

4. **The submission must be our own original work.** (§3.14.a) We warrant sole ownership and the
   right to grant the licenses. This is the rule behind the standing instruction never to paste
   concept_sources or nvarc_reference into the submission.

5. **Weights count as source.** (§2.5.a) Open source system, model, AND weights/parameters are
   required, per the OSI Open Source AI definition. If the learned policy head or a TRM lands
   (see `Research Synthesis.md` §6), its weights ship with the code.

6. **One Kaggle account.** Submitting from a second account is instant disqualification.

7. **Dependencies must be OSI-approved and must not limit commercial use.** (§3.6.c)
   `include/third_party/fmt` is MIT — clears it. Copyleft (GPLv3) technically clears this clause
   too, but would fight our own MIT-0 license, so avoid it in practice.

8. **External data and tools** must be publicly available to all participants at no cost, or meet
   a "reasonableness" standard on cost. (§2.6)

## The license question

- Kaggle (§2.5): winner license type is **CC BY 4.0**, which requires attribution.
- ARC Prize: code must be under a **permissive public domain license (eg. CC0 or MIT-0)**.
- The strings CC0 and MIT-0 appear ZERO times on the Kaggle page.

We chose **MIT-0** (`LICENSE` at repo root). It is more permissive than both, so it satisfies
either reading, and the Kaggle clause is a license we GRANT the sponsor — not a cap on what our
own repo may use. Nothing to change unless the wording changes.

## Deadlines and limits

- One submission per day. Two final submissions selected for judging.
- Maximum team size five. We are solo.
- Open sourcing happens BEFORE private scores are released — the repo must be publishable at
  submission time, not afterwards.
- Innovation Prize artifacts: attached to a Solution Writeup within seven days of the deadline.
- Winner obligations: deliver training code, inference code, a description of the compute
  environment, an interview with the sponsor, and work with their technical writer.
- Governing law is California. Sanctioned-country residents are ineligible.

## The data we develop against is NOT the data Kaggle ships (read before uploading)

The repo stores no puzzles — the rules forbid redistributing them (§2.4.b) — so
`scripts/fetch_data.py` rebuilds the whole tree from the public ARC-AGI-2
repository into the layout Kaggle mounts. Checked against the files Kaggle
actually shipped: training, test and sample_submission come out byte-identical.

SIX EVALUATION TASKS DIFFER: 4a21e3da, abc82100, b6f77b65, d8e07eb2, f560132c,
faa9f03d. Same ids either way, but Kaggle's copy carries one extra test pair on
five of them, plus a train-pair difference on d8e07eb2. The public repo's
changelog (2025-04-14, 2025-04-17) records those edits as fixes for
off-by-one-pixel errors and ambiguous test pairs, so the public copy is the
CORRECTED one.

This will not resolve itself. The public file was last edited 2025-05-15, long
before the 2026 competition was packaged, and Kaggle still ships the other
version. Treat them as two durable variants, not as a lag.

What it means when uploading:
1. Nothing to do at submission time. Kaggle mounts its OWN data at
   /kaggle/input, so the submission is always scored against Kaggle's copy
   whatever sits on the dev machine.
2. Our eval numbers come from the corrected set, which is the better proxy for
   the hidden 240 — ambiguous pairs are exactly the ones not worth scoring.
3. One old artifact predates the switch: `benchmarks/benchmark_2026-07-16_10-35-21_AM.txt`
   was scored against Kaggle's uncorrected copy and touches all six tasks. It is
   not comparable to later runs on those tasks.
4. Kaggle's copy is re-downloadable any time with
   `kaggle competitions download -c arc-prize-2026-arc-agi-2`, so nothing is lost
   by not keeping it on disk.

How the data arrives and when it changes:
1. `scripts\build.bat` fetches the puzzles the first time they are missing, and
   never updates them on its own.
2. Updating is a deliberate act: `python scripts\fetch_data.py` overwrites.
3. After an update, benchmark numbers from before and after are NOT comparable
   on any task that changed. Refresh AFTER a benchmark run, not before one, and
   write down the date.
4. Every build asks GitHub whether the upstream puzzles moved and prints a notice
   if they did, so the ground is never changing without us choosing it. Silent
   when offline, and silent when current. As of 2026-08-16 the upstream data has
   not changed since 2025-05-15, so this should stay quiet for long stretches.

## The part worth acting on

The **Innovation Prize is $275,000** — the same pot as all eight Progress prizes combined — and it
is scored on six equally weighted criteria: Accuracy, Universality, Progress, **Theory**,
Completeness, **Novelty**. Theory is explicitly "why it works, as opposed to merely how".

Our `notes/` tree is already that artifact, and `Human Work Landscape.md` shows the human-aligned
niche is empty, which is the Novelty argument. So publishing the notes is not only a risk to weigh
against competitors reading them — it is a scored deliverable. Decide it on those terms.
