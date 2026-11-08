# Reusable Chapter-Authoring Prompt

> **How to use this file.** Fill in the `VARIABLES` block below, then paste the whole document to the
> authoring agent as its task prompt. It is a self-contained recipe for writing any *new* chapter of the
> DSA Quarto book to the same bar as Chapters 23–26, and it encodes only facts verified against this repo
> (exact scripts, interpreters, gate behaviours, file conventions). Every command is PowerShell-safe.

---

## 0. VARIABLES (fill these in first)

```text
NUM       = <chapter number, e.g. 26>              # must exist in outline.yml
TITLE     = <title from outline.yml>               # new_chapter.py pulls this automatically
VOLUME    = <folder, e.g. arrays | dp | trees | strings | graphs>
COVERS    = <the outline.yml "covers" string — your lesson/pattern scope>
PREFIX    = <short lowercase label prefix, letters only, e.g. hsh>   # -> anchors become sec-<PREFIX>-*
SLUG      = <file slug, e.g. hash_multisets>       # default derived from TITLE if omitted

PATTERNS          = <number of Part B patterns, default 4>
SOLVED_PER_PATTERN= <default 2  ->  total solved slots = PATTERNS * SOLVED_PER_PATTERN>
EXERCISES         = <>= 16>
INTERVIEWS        = <>= 13>
```

The chapter file will be `books/<VOLUME>/<NUM:03d>_<SLUG>.qmd`.

---

## 1. Ground rules (read before writing anything)

1. **Authoritative inputs.** `outline.yml` defines `title`, `covers`, `folder`, and per-chapter `status`.
   The four gates under `scripts/` are the definition of "done". Provenance/structure must mirror the
   nearest already-written sibling chapter in the same volume.
2. **Never fabricate or mislabel a judge link.** Before a Codeforces/CSES/SPOJ problem is used as a
   solved `.solved` slot, live-verify its **intended** solution is genuinely this chapter's technique.
   A slot that fails verification becomes a fully-stressed **self-authored** task and the shortfall is
   recorded in the provenance — exactly the Ch23/24/25/26 precedent.
3. **Do not modify previously written chapters** (or their registry `problem.qmd`, `traces/` modules,
   cheatsheets, `outline.yml` `status`, or shared assets) without first showing the user the exact file
   + line diff and receiving approval. Only the *new* chapter's own files and its `outline.yml` `status`
   entry are exempt from this rule.
4. **Comprehensiveness over compactness.** Match (or exceed) Ch25's depth: 6 lessons, 4 patterns × 2
   solved each, pattern map, a single pitfalls box, >=16 exercises, >=13 interview questions, summary,
   practice ladder, further reading.

---

## 2. Toolchain (use these exact interpreters — this is the #1 source of failed runs)

- Default `python` = **3.14** (has NO matplotlib, NO PyYAML). Use it for `check_style.py`, `check_cpp.py`,
  `check_links.py`.
- `py -3.11` = **has matplotlib + PyYAML**. Use it for `check_chapter.py`, `new_chapter.py`, `gen_book.py`,
  `new_problem.py` (anything that renders figures or parses `outline.yml`).
- Windows PowerShell: chain commands with `;`, **never** `&&`. `head`/`grep` are unavailable — use the
  `Grep`/`Select-String` tools instead.

Gate invocations (run from the repo root, `dsa_book/`):

```powershell
python   scripts/check_style.py  books/<VOLUME>/<NUM:03d>_<SLUG>.qmd
py -3.11 scripts/check_chapter.py books/<VOLUME>/<NUM:03d>_<SLUG>.qmd
python   scripts/check_cpp.py    books/<VOLUME>/<NUM:03d>_<SLUG>.qmd
python   scripts/check_links.py  books/<VOLUME>/<NUM:03d>_<SLUG>.qmd        # add --online for live names
```

---

## 3. Step 1 — Prefix collision check + anchor plan

Before scaffolding, confirm `PREFIX` is globally free (a duplicate `sec-<prefix>` silently breaks crossrefs):

```powershell
Select-String -Path books/**/*.qmd -Pattern '\{#sec-' | ForEach-Object { $_.Matches.Value } | Sort-Object -Unique
```

Pick a short, letters-only prefix not in that list (note: in `arrays`, `hash` is already taken by Ch07, and
`pref/tp/sw/srt/bst/salgo/lts/int/part/idx/mat/ds/cnt/gcd/mex/coord/sqrt/mo/nd` are used — choose a new one).

Anchor map to lay down (all `<p>` = `sec-<PREFIX>`):

- Top: `{#<p>}` · Why: `{#<p>-why}` · idea figure `{#fig-<p>-idea}`
- Lessons: `{#<p>-a1}` … `{#<p>-aN}` (each with a matching `{#fig-<p>-aK}` where a figure exists)
- Patterns: `{#<p>-p1}` … `{#<p>-pM}`
- Solved anchors: `sec-<p>-p<slot>` for self-authored; judge registry files carry their own anchor
  `sec-<p>-p<slot>-<key>` (e.g. `sec-hsh-p1-subperm`)
- Refined Python cell labels: `sec-<p>-p<slot>-sol`
- Pitfalls: `{#<p>-pit}` — **defined exactly once**, referenced from lessons/problems/exercises.

## 4. Step 2 — Scaffold

```powershell
py -3.11 scripts/new_chapter.py <NUM> --prefix <PREFIX> --slug <SLUG>
```

This writes `books/<VOLUME>/<NUM:03d>_<SLUG>.qmd` from the template, fills the title from `outline.yml`,
substitutes `{#sec-<PREFIX>}`, sets `status: drafting` (optional `--owner <you>`), and errors if a file for
`<NUM>` already exists. Confirm the `status: drafting` line was inserted in `outline.yml`.

---

## 5. Required chapter structure (build top-to-bottom)

1. **`<!-- author-facing provenance -->`** comment (not rendered): what's covered, the anchor/lesson order,
   the roster honesty note (which slots are real judge problems vs self-authored and why), any paraphrased
   statement constraints, and which visual devices are used.
2. **`::: {.callout-note title="Sources"}`** — the references this chapter draws on + sibling chapters via
   `@sec-...`.
3. **`::: {.callout-tip title="What you will be able to do"}`** — bulleted, each item pointing at its anchor.
4. **`#| include:false` preamble cell** — standard imports, then:
   ```python
   sys.path[:0] = ["../shared", "../traces"]
   import dsa_viz as V
   random.seed(0)
   ```
5. **`## Why this matters {#<p>-why}`** — a plain-words callout, a triage table ("statement hint → what it
   really is → which section"), the idea figure, and a roadmap of the lessons/patterns.
6. **`## Part A — Lessons`** A1..AN — each lesson: an *In plain words* callout → the engine cell → a stress
   cell comparing the fast engine to a brute oracle over hundreds of random cases → a figure → a takeaway
   paragraph.
7. **`## Part B — Patterns`** P1..PM — each pattern: a **"Where we are"** intro paragraph that explicitly
   states what the previous pattern assumed → **"Recognise it"** → a `callout-important` **Template** box →
   `SOLVED_PER_PATTERN` solved problems separated by `{.flow-bridge}` lines → closing boxes.
8. **Closing**: Pattern map, Pitfalls, Exercises, Interview questions, Summary, Practice ladder, Further
   reading (see §10).

---

## 6. Style-B solved-problem template (the fence rules are strict — get them verbatim)

Order inside each solved problem:

````
### Problem: <Name> {#<anchor>}

::: {.p-statement}        # Source line: judge link {.solved rating="…"} OR self-authored note; example
:::
::: {.p-intuition}
:::
::: {.p-approach}         # numbered solution ladder + a bold **Invariant.**
:::

The buttons are in order. Each one keeps the previous solution and changes only what that form changes.

::: {.panel-tabset .solution-forms}         # OUTER tabset

#### <brute title>
::: {.form-next} … :::
::: {.panel-tabset group="lang"}            # INNER lang tabset
##### Python
```{python} … ```
##### C++
```cpp … ```
:::                                        # close INNER lang

#### <refined title>
::: {.form-next} … :::
::: {.panel-tabset group="lang"}
##### Python
```{python}
#| label: <anchor>-sol
…                                          # asserts vs brute oracle + random stress
```
##### C++
```cpp … ```
:::                                        # close INNER lang
:::                                        # close the OUTER .solution-forms with ONE lone :::

::: {.p-complexity} … :::                   # SEPARATE block, AFTER the outer close
::: {.p-edge} … :::
::: {.practice} … :::
````

Hard rules the gates enforce:
- **Blank line above every `::: {` opener** (`check_style` ERROR otherwise).
- The refined form (and any form with both languages) **must** wrap Python+C++ in an inner
  `::: {.panel-tabset group="lang"}` with `##### Python` / `##### C++` headers, closed by a lone `:::`.
- The **outer `.solution-forms` tabset gets exactly one lone `:::`** and it closes **before** `.p-complexity`
  (a very common Ch25 break when the inner lang tabsets were omitted).
- Registry-included judge problems already carry their own `### Problem:` heading → **do not** add a
  chapter-side heading above the `{{< include … >}}`.

## 7. Registry judge files

- Put each judge problem in `books/problems/cf-<id>-<kebab-slug>/problem.qmd` (create with
  `py -3.11 scripts/new_problem.py`, or lift a finished inline block with `scripts/to_registry.py`).
- In the chapter, include it: `{{< include ../problems/<key>/problem.qmd >}}`.
- Run the four gates on each registry file **standalone** (includes are not expanded there).

---

## 8. Roster & live-verification protocol (do this before writing Part B)

- **Metadata source:** `scripts/.cache/cf_problemset.json` → `result.problems[]`, each with `contestId`,
  `index`, `name`, `rating`, `tags`. Problems are keyed by `(contestId, index)`; `index` is the
  contest-relative letter (A/B/C…), so filter by the tuple, not the letter alone. Quote names/ratings
  **verbatim** from this cache.
- **Verify the intended solution** matches the chapter technique: fetch the CF statement + editorial
  (`WebSearch`/`WebFetch` of `https://codeforces.com/problemset/problem/<contestId>/<index>`). Be skeptical
  of tag-only "looks like" fits (a `matrices` tag does not mean prefixing; a `hashing` tag does not mean
  the editorial uses hashing). If the honest framing cannot be kept, **demote to cite-only** and solve a
  self-authored task instead.
- **Duplicate-`.solved` scan** before locking a slot:
  ```powershell
  Select-String -Path books/**/*.qmd -Pattern 'problemset/problem/<id>/<idx>'
  ```
  A problem already `.solved` elsewhere, or already a prominent practice link elsewhere, is **cite-only**
  (`{rating=…}`, never `.solved`, never self-referenced).
- **Link text formats** (`check_links` rejects anything else):
  - CF solved source: `[CF <id> · <Name>](https://codeforces.com/problemset/problem/<id>/<idx>){.solved rating="<r>"}` — `.solved` **only** on the problem's own source line.
  - Practice links: `…{rating="<r>"}` (no `.solved`).
  - CSES: link text must be `<JUDGE> <id> · <name>`; a bare `cses.fi/problemset` is treated as a malformed
    judge link; `spoj.com` and `cses.fi/book` are ignored by the checker.
  - LeetCode is unreliable for randomized/batch techniques — prefer CF/CSES/SPOJ; any LC link must be
    format-checked and collision-free.
- **Recorded shortfall**: if fewer than the baseline judge problems survive verification, state how many and
  why in the provenance and each pattern intro; fill the rest with stressed self-authored tasks.

---

## 9. Visuals & animations (from `books/shared/dsa_viz.py`)

Two figure families — use whichever teaches the point:

**Static** (renders light+dark):
```python
#| echo: false
#| renderings: [light, dark]
def draw(ax, p):
    V.grid(ax, p, cells, x=0.0, y=0.0, colors={(3,3): "green"})
V.show(draw)                                # MUST be at column 0
```
Wrap it in `::: {#fig-<p>-aK}` … `:::` with a blank line before the closing `:::` and a prose caption after.

**Animation** (self-themed, no `renderings` tag) — the exact gate-passing idiom:
```python
#| echo: false
<n>_frames, <n>_caps = [], []
for i, x in enumerate(seq):
    <n>_frames.append((i, state_copy))
    <n>_caps.append(f"i={i}: {what changed}")
assert <invariant on final state>

def <n>_frame(ax, p, k):
    i, st = <n>_frames[k]
    V.array(ax, p, seq, colors=..., pointers=[(i, "x", "indigo", "top")])
    V.box(ax, p, 8.0, 0.05, 1.8, 0.9, f"hash {st}", color="green")

V.animate(<n>_frame, len(<n>_frames), captions=<n>_caps, interval=1100)
del i, x, st                                # keep the shared kernel namespace clean
```

Helpers available: `V.array` (`regions`, `pointers`, `colors`), `V.grid` (`cells`, `colors`, labels),
`V.bars`, `V.flow` (nodes+edges → pipelines/decision trees), `V.box`/`V.arrow`/`V.text`, `V.tree`,
`V.stack`, `V.intervals`, `V.iso_grid`, `V.story` (step-card board; needs `trace_log` dicts — prefer
`V.animate`), `V.show`, `V.animate`. Aim for Ch25-level figure density (~7–10 static figures + 2–3
animations across the chapter). **Any random number a figure displays must come from the seeded
`random.seed(0)` map** so `check_chapter` reproduces it exactly.

---

## 10. Closing sections (checklist)

- **Pattern map** — a `Signal in the statement | Pattern | Section` table with `@sec-<p>-*` refs, one row per
  technique + a "this looks related but is NOT (and is which other structure)" row.
- **Pitfalls `{#<p>-pit}`** — one `callout-warning` box, defined once, referenced throughout.
- **Exercises** — `>=16` items `**N. [E/M/H] …**`, each with `::: {.callout-note collapse="true" title="Solution"}`;
  mix derivations, conversions, and spot-the-look-alike.
- **Interview questions** — `>=13` items with collapsed `Answer` callouts.
- **Summary** — `::: {.callout-important title="Templates to remember"}` + takeaway bullets.
- **Practice ladder** — the judge `.solved` entries (with `@sec-<p>-*` refs), the self-authored solved
  problems (named with anchors + brute/stress note), and extras as `{rating}`/`{difficulty}` cite links.
- **Further reading** — `>=3` bullets, each containing a real hyperlink (`check_links`/`check_style` reject
  a further-reading bullet with no hyperlink), then a non-bullet prose paragraph deferring out-of-scope
  structures.

---

## 11. The four validation gates (run until ALL green — target 0 errors)

1. **`check_style.py`** (default `python`) — fenced-div balance including the single outer `.solution-forms`
   close; blank line above every `::: {`; `V.show(draw)`/`V.animate(...)` at column 0; **no unescaped `%`
   inside `$math$`** (write `2^{-64}` or backtick it); no `D:\` Windows paths; further-reading bullets carry
   hyperlinks. Target **0 errors**.
2. **`check_chapter.py`** (`py -3.11`, needs matplotlib) — runs every `{python}` cell in **one shared kernel
   in document order** and expands registry includes; all asserts must pass. **Validate every new engine in a
   throwaway `_tmp_*.py` against a brute oracle BEFORE** pasting the cell into the chapter, and make asserts
   deterministic under the seed (never let a fresh random hash decide an assert's truth).
3. **`check_cpp.py`** (default `python`, GCC-6) — 0 compile/link failures. Portability: `vector<vector<…> >`
   with a **space**, no `std::gcd`, no structured bindings, explicit `long long`/`unsigned long long`. The
   harness prepends `<bits/stdc++.h>` + `using namespace std;`. `// fragment` header ONLY on oracle fragments
   (they get `-fsyntax-only`); real solutions must start with code/`int main` so they are linked and run.
4. **`check_links.py`** (default `python`; `--online` to hit the API) — names/ratings verbatim vs cache,
   judge-link text format, duplicate `.solved` scan. Run on the chapter **and each registry file standalone**.

**Crossref integrity:** scan for 0 dangling `@sec-…` (use `scripts/qmdlib.py` `.expand`); verify every
referenced sibling anchor actually exists; **do NOT reference `@sec-seg`** (no segment-tree chapter exists, so
it dangles) — describe segment trees / 2D BIT in prose instead.

---

## 12. Finalize (only after every gate above is green)

```powershell
# 1. set outline.yml: this chapter's status: drafting -> status: written   (edit only the new chapter's entry)
py -3.11 scripts/gen_book.py                 # 2. regenerate the book (needs PyYAML -> 3.11)
python scripts/check_links.py                # 3. whole-book link + duplicate-.solved scan (0 problems)
```

4. Final crossref scan of `sec-<PREFIX>` (0 dangling).
5. Delete every `_tmp_*.py` scratch file you created for this chapter. Leave unrelated pre-existing scratch
   files untouched.

---

## 13. Milestone cadence (advance in turns; "continue" resumes from the last stop)

1. Scaffold + confirm `sec-<PREFIX>` free and `status: drafting`.
2. Live-verify & lock the roster (demote as needed; record look-alikes/cite-only).
3. Front matter (provenance, Sources, WYBATD, preamble, Why-this-matters).
4. Part A lessons + visuals → `check_chapter` green.
5. Part B, pattern by pattern: build/validate each engine in `_tmp_*.py` → write registry/inline solved block
   → run all four gates on it → delete temp → append the pattern section.
6. Closing sections (pattern map, pitfalls, exercises, interviews, summary, ladder, further reading).
7. Full-chapter gates all green + crossref scan 0 dangling.
8. Finalize (§12).

## Definition of done
All four gates report 0 errors / 0 failures / 0 problems on the chapter and each registry file; whole-book
`check_links` is clean; `outline.yml` shows `status: written`; `gen_book.py` succeeded; 0 dangling
`@sec-<PREFIX>` refs; scratch files removed.
