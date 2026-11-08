# Chapter style guide (DSA book, v2)

Every chapter is a hand-written Quarto file, `books/<topic-folder>/NNN_<slug>.qmd` (folders are listed in PLAN.md §2). Chapters are organised **by pattern**: a short theory block per pattern, then fully solved problems, then practice.

## Depth standard (comprehensive guide, not a short textbook)

| Requirement | Minimum |
|---|---|
| Coverage | Everything CLRS / Erickson / CP-Algorithms / CPH cover for the topic, plus the interview patterns from LeetCode-style sources. List the coverage in the Sources box |
| Length | About 25–60 printed pages. If longer, **split the chapter**. Never compress |
| Patterns | As many as the topic needs. Each pattern: when to recognise it, the template, its invariant, and its complexity |
| Solved problems | **≥ 2 per pattern**, written in the full problem format below. Mix classics, concept builders, recent contest/interview problems and tricky edge-case problems, but **never print a kind tag** |
| Practice | ≥ 1 practice problem per solved problem, ordered **by concept, not rating**. Prefer Codeforces, then LeetCode, CSES, AtCoder |
| Exercises | ≥ 15 per chapter ([E]/[M]/[H]), each with a full collapsed solution |
| Interview questions | ≥ 12 per chapter (conceptual, complexity, "what breaks", follow-ups), each with a collapsed answer |

## Problem format

Start from `templates/chapter.qmd`. The rendered result is shown in `books/appendix/anatomy.qmd`. Every solved problem uses these blocks, in this order. Each block is a styled div with its own colour and label:

````markdown
### Problem: Pair with a target sum in a sorted array {#sec-tp-two-sum-sorted}

::: {.p-statement}
Restatement with constraints (n ≤ 2·10^5, values ≤ 10^9).

Source: [LC 167 · Two Sum II - Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/){.solved difficulty="Medium"}
:::

::: {.p-intuition}
The one observation that makes the problem easy.
:::

::: {.p-approach}
Brute force first (with its complexity), then the refined algorithm step by step, with the invariant. Number the steps `1.`, `2.`, … — each numbered step becomes one **form** in the tabset below.
:::

The buttons are in order. Each one keeps the previous solution and changes only what that form changes.

::: {.panel-tabset .solution-forms}

#### Sort, then two pointers

::: {.form-next}
One sentence on what this form does and what it costs. This is the brute force / first idea: faithful to the wording, provably right, and more work than the problem needs.
:::

::: {.panel-tabset group="lang"}
##### Python
```{python}
...this form's function, named *_brute for the first form...
```
##### C++
```cpp
...same algorithm...
```
:::

#### Count once, answer in O(1)

::: {.form-next}
What changes from the previous form and why it is better. This is the refined pattern.
:::

```{python}
#| echo: false
#| renderings: [light, dark]
def draw(ax, p):
    title, inp, steps, final = Traces.get("<key>", "<form>")
    assert str(final) == "<literal>"
    V.story(ax, p, title, inp, steps, final)
V.show(draw)
```

::: {.panel-tabset group="lang"}
##### Python
```{python}
#| label: <slug>
...the optimised solution, stress-tested against the brute form above...
```
##### C++
```cpp
...same algorithm...
```
:::
:::

::: {.p-complexity}
Price **each form by name** — the same names as the `####` tab headings — with its time and space. Do not give a single unlabelled bound.
:::

::: {.p-edge}
- Empty input, duplicates, negatives, overflow, n = 1, all equal, …
:::

::: {.practice}
- [CF 279B · Books](https://codeforces.com/problemset/problem/279/B){rating="1400"}
- [LC 15 · 3Sum](https://leetcode.com/problems/3sum/){difficulty="Medium"}
:::
````

### Solution forms: the `.solution-forms` tabset

The code is presented as **progressive solution forms**, the way the DP book does it, so the reader watches one solution turn into the next instead of jumping from a paragraph to a wall of code.

- **Lead-in line.** Immediately after the `.p-approach` div, one fixed sentence: *"The buttons are in order. Each one keeps the previous solution and changes only what that form changes."*
- **Outer tabset.** `::: {.panel-tabset .solution-forms}` holds one `#### Form name` tab per numbered step in `.p-approach` — usually **2–3 forms**. The first form is the brute force (its Python helper is named `..._brute`), the last is the pattern you want the reader to remember. Each form **keeps the previous solution and changes only what that form changes**.
- **`.form-next` prose.** Every form tab opens with a `::: {.form-next}` block: one to three sentences naming what this form does, what it costs, and what it improves over the previous form.
- **Language tabs move down a level.** Because the form is now the `####` heading, the inner language tabset uses `::: {.panel-tabset group="lang"}` with `##### Python` / `##### C++` — **five** hashes, one below the form heading. The `group="lang"` still carries the reader's language choice across the whole page.
- **Close two fences.** At the very end of the last form, close the inner language tabset (`:::`) *and* the outer `.solution-forms` tabset (`:::`) — two bare `:::` lines. `check_style.py` fails the build if the count is off.
- **Complexity prices the forms by name.** The `.p-complexity` div references each `#### Form name` in bold and gives its time/space, so the reader sees what each step bought.
- **No math in tab headings.** Do not put `$…$` in a `#### Form name` or `##### Python` label — Quarto renders tab labels into the navigation bar, where inline math does not lay out. Spell it out (`O(1)` as plain text).

### Trace boards (only where state mutates)

When a form's **state actually changes** — a running array being compacted, a window sliding, a carry propagating — show it with a `V.story` step board between the `.form-next` prose and the language tabset. Skip it entirely for forms that are set/dict based or where the prose is already clear; a board for a form that does not mutate state is noise.

- **Board data lives in a shared module**, `books/shared/arr005_traces.py`, `arr006_traces.py`, … (mirroring the DP trace modules). Import it in the setup cell as `Traces`: `import arr005_traces as Traces`.
- **Contract.** `Traces.get(problem_key, form_key)` returns `(title, input_line, steps, final)`. The module re-runs the real algorithm to build `steps`, so the picture is computed, not hand-typed.
- **Three-way honesty.** The final value must agree in three places: the module computes it, the drawing cell asserts it (`assert str(final) == "<literal>"`), and the last form's Python cell asserts the *real function* returns the same thing on the same sample input — that line is tagged `# the board sample`.
- The drawing cell is the standard figure cell (`#| echo: false`, `#| renderings: [light, dark]`) calling `V.story(ax, p, title, inp, steps, final)` via `V.show(draw)`.

- **Code tabs.** In a solved problem the code lives inside the `.solution-forms` ladder, so the language tabset is nested under a form and uses `##### Python` / `##### C++`. Only where a code tab is **not** part of the forms ladder (a standalone snippet in a template box, say) does it use a bare `{.panel-tabset group="lang"}` with `####` headings one level under the enclosing heading. In both cases the `group` keeps the reader's language choice across the page.
- **Judge links** render as badges. Link text is exactly `<JUDGE> <id> · <official name>`, where JUDGE is `CF`, `LC`, `CSES` or `AC`. Codeforces links carry `rating="NNNN"` (shown in the Codeforces colour). LeetCode links carry `difficulty="Easy|Medium|Hard"`. Mark the problem a section solves with `.solved`; it is listed as *Solved* in the problem index.

## Skeleton

```markdown
# Two Pointers {#sec-two-pointers}

::: {.callout-note title="Sources"}
CLRS § … · CPH ch. … · CP-Algorithms … · Erickson ch. …
:::

::: {.callout-tip title="What you will be able to do"}
- 3–6 concrete, checkable outcomes
:::

## Why this matters
## Pattern 1: <name> {#sec-tp-p1}
**Where we are.** …               <- the ladder of patterns, in one paragraph (first pattern only)
### Recognise it / Template / Invariant / Complexity
{{< include ../problems/<key>/problem.qmd >}}   <- ≥ 2 per pattern, from the registry
[**From X to Y.** …]{.flow-bridge}             <- one-line connector after every problem but the last
{{< include ../problems/<key2>/problem.qmd >}}
## Pattern 2: …
**From <what Pattern 1 could not do>.** …     <- bridge paragraph under every pattern heading
…
**Looking back.** …                <- before the toolkit / pattern map
## Pattern map                     <- table: signal in the statement → pattern → section
## Pitfalls
## Exercises                       <- ≥ 15, [E]/[M]/[H], collapsed solutions
## Interview questions             <- ≥ 12, collapsed answers
## Summary                         <- templates callout + 6–10 takeaways
## Practice ladder                 <- by concept; ≥ 1 per solved problem
## Further reading                 <- 4–8 clickable links; no local paths (see Rules)
```

## Rules

- **No YAML `title:`.** The chapter title is the single H1 with its `{#sec-…}` label.
- **No bare `---` rules.** A `---` line at column 0, with no front matter above it, makes Quarto read the text that follows as YAML (`YAMLException: unidentified alias …`) and the chapter does not render. Separate sections with a heading or a callout — never a thematic break.
- **A fenced div opens after a blank line.** `::: {.callout-note …}` glued to the line above it is parsed as that paragraph's continuation, and Pandoc emits the `:::` as literal text (the render warns: "The following string was found in the document: :::"). Leave an empty line above every opening fence; `check_style.py` makes this an ERROR.
- **Code.** Python cells are executable (` ```{python} `). Each chapter starts with a hidden setup cell (`#| include: false`) that imports packages and seeds `random`. Every optimised solution is **stress-tested against a brute force** on random inputs with `assert`, plus the edge cases from the problem. Cells must run in under ~60 s per chapter.
- **Diagrams and film strips.** The LaTeX notes fit a printed page, so they often stop after one picture. This book is the web edition: add a diagram or a film strip wherever a picture makes the idea easier to see. There is no quota and no requirement that every section have one. Use one when the structure, the sample, or the changing state is hard to hold from the prose alone, and skip one when the text is already clear.

  A film strip is the animation: several states of one walk, top to bottom, usually via `V.trace`, with rows labelled `step 1`, `step 2`, …. Record the rows by running the algorithm inside the drawing cell when that function is already defined. A tree or a graph that changes can be several `V.tree` or `V.flow` drawings in one figure. There is no video and no GIF. A second picture that repeats the first one, with no new state, is wasted.

  Draw them with the shared helper `books/shared/dsa_viz.py` (imported as `V` in the setup cell: `sys.path.insert(0, "../shared")`), in this exact form (the label and caption go on a **figure div**, not in the cell: with `#| label`/`#| fig-cap` inside the cell, Quarto shows both theme versions as subfigures (a) and (b)):

  ````markdown
  ::: {#fig-<prefix>-name}

  ```{python}
  #| echo: false
  #| renderings: [light, dark]
  def draw(ax, p):
      ...
  V.show(draw)
  ```

  Caption: what the picture shows and what to notice.

  :::
  ````

  `V.show(draw)` draws the figure once per theme. The drawing code is hidden (`echo: false`) but still runs and is checked. Available: `V.array` (cells, regions, pointers), `V.trace` (stacked steps), `V.stack`, `V.tree` with `V.Node` (recursion/decision trees, level notes), `V.intervals` (timelines), `V.bars`, `V.grid` (DP tables, game tables), `V.flow` (boxes and arrows), `V.box`, `V.arrow`, `V.text`, and `V.plot` for ordinary data plots. Colours are named (`indigo`, `cyan`, `green`, `amber`, `pink`, `violet`, `red`, `muted`). **Compute the diagram from the algorithm** whenever the function is already defined above the figure, so the picture is checked like the code. Do not use Mermaid, Graphviz, or an embedded video: they do not follow the site's dark mode. Refer to a figure the prose depends on with `@fig-…`.
- **C++** is shown as a listing (` ```cpp `), not executed. It must compile with `g++ -std=c++17 -O2` and match the Python logic.
- **Math.** `$…$` / `$$…$$`; label referenced equations `{#eq-name}`. Use $n$ for input size, $\log$ is base 2 unless stated.
- **Cross-references.** `@sec-…`, `@eq-…`, `@fig-…`. Never write "Chapter 7" by hand.
- **Collapsed solutions.**
  ```markdown
  ::: {.callout-note collapse="true" title="Solution"}
  …
  :::
  ```
- **Callouts.** `callout-important` for templates and key facts, `callout-warning` for pitfalls, `callout-tip` for interview tips.
- **Links.** Practice links go to the problem page (Codeforces `problemset/problem/<contest>/<index>`, LeetCode `problems/<slug>/`). Do not invent problem IDs; check each link.
- **Language: beginner-friendly first.** Write for a motivated reader who knows arrays and loops but not this topic. Give the plain-words idea *before* the formal definition, then the precise version. Prefer short sentences, concrete numbers, and "in other words" glosses; a reader should never need to look up a term to follow the next paragraph. Add a `{.callout-note}` titled "In plain words" (or a warm opener under *Why this matters*) wherever a section is unusually dense or formal.
- **No local file paths in rendered content.** Never print `D:\…`, a bare filename, or an editor path in the Sources callout, Further reading, or anywhere on the page — those links are dead for a web reader. Keep personal note paths only in the author-facing HTML comment at the top of the template.
- **Further reading.** End every chapter with `## Further reading`: 4–8 references, **each a working hyperlink** to the book, paper, or blog, with one clause on what it covers. Verify the URL resolves; do not invent DOIs or blog ids.
- **What not to include:** kind tags (Classic/Concept/Recent/Tricky), TODOs, unchecked claims, study-ritual advice.

## Project layout

```
dsa_book/
  outline.yml          volumes → parts → chapters (numbers, titles, patterns). The ONLY place the structure lives.
  PLAN.md  STYLE.md  README.md
  templates/chapter.qmd
  scripts/  gen_book.py  build.ps1  check_chapter.py  check_style.py  check_links.py
            new_chapter.py  new_problem.py  to_registry.py  qmdlib.py
  books/
    <folder>/NNN_<slug>.qmd          chapters (folder and NNN from outline.yml)
    problems/<key>/problem.qmd       a solved problem, written once (the problem registry)
    problems/<key>/solution.qmd      optional: the tested solution as its own cell, to reuse the code
    problems/<key>/card.qmd          generated: a short pointer to the problem's home write-up
    traces/<chapter>_traces.py       step-board data for V.story (one module per chapter)
    shared/                          dsa_viz.py, style.css, themes, site.js (player, tilt, tracker)
    appendix/                        anatomy.qmd, problem_index.qmd (generated)
```

- **Start a chapter:** `python scripts/new_chapter.py 230 --prefix dp1d` (title, folder and number come from `outline.yml`; the outline's pattern list is copied into a comment at the top).
- **Start a problem:** `python scripts/new_problem.py CF 1155D "Beautiful Array" --rating 1900` creates `books/problems/cf-1155d-beautiful-array/`.
- **Move an existing chapter's problems into the registry:** `python scripts/to_registry.py books/<folder>/NNN_<slug>.qmd` (labels are kept; the chapter is identical after include expansion).
- **Change the structure** (rename, add, reorder chapters): edit `outline.yml`, then build. Never hand-edit the sidebar in `_quarto.yml` or the outline on the home page.
- Chapter numbers in the sidebar, titles, section numbers and figure captions are the book's numbers from `outline.yml` (`site.js` maps Quarto's running numbers to them).

## Problem registry: write once, reuse anywhere

Every solved problem lives in **one** folder, `books/problems/<key>/`, keyed `<judge>-<id>-<name>` (`lc-0070-climbing-stairs`, `cf-1155d-beautiful-array`, `ac-dp_a-frog-1`).

| You want to… | Write in the chapter |
|---|---|
| solve the problem here (its **home**) | `{{< include ../problems/<key>/problem.qmd >}}` |
| mention it again later and point to the full solution | `{{< include ../problems/<key>/card.qmd >}}` |
| run or show the same tested code again (a variant, a comparison) | `{{< include ../problems/<key>/solution.qmd >}}` |
| link to it in a sentence | `@sec-…` with the label from its `### Problem:` heading |

Rules:

- A `problem.qmd` is included in full by **one** chapter only (its home). `check_links.py` fails the build if two chapters include it, or if the same judge link opens a full write-up (a `Source:` line) twice.
- **Per-problem traces.** Helper code that draws one problem's figures lives next to it, in `books/problems/<key>/traces.py`, so the folder moves as one piece. Load it in the problem file with `import registry; T = registry.traces("<key>")`. Trace code shared by several problems of a chapter stays in `books/traces/<chapter>_traces.py`.
- Include paths are always `../problems/<key>/…`, also inside a problem file: Quarto resolves includes relative to the chapter, and every chapter is one folder below `books/`.
- The card is generated by `gen_book.py` from the problem's heading, its `Source:` line and its `.p-intuition` paragraph — so write the intuition as a self-contained sentence or two.
- Editing a problem (or a trace module a chapter imports) re-executes every chapter that uses it: `gen_book.py` hashes each chapter with its includes and imports and clears that chapter's frozen output when the hash changes.

## Chapter flow: one climb, not a list of boxes

A chapter reads like one argument that gets stronger, not a set of independent sections.

1. **The patterns form a ladder.** Order them so each one is the previous one plus **one idea** (a new decision, a new index, a new fact in the state, a new phase). Write that ladder down in the opening of the first pattern ("**Where we are.** …").
2. **Every pattern opens with a bridge**, a bold-led paragraph right under its heading: what the previous pattern could not handle, and the one idea this pattern adds (`**From one index to two.** …`).
3. **Every solved problem ends with a connector** to the next problem, as a styled one-liner:
   `[**From counting to costs.** Same graph as before, but each arrow has a price, so + becomes min.]{.flow-bridge}`
   It names the single twist the next problem adds. If you cannot name one, the order is wrong.
4. **Problems inside a pattern are ordered by the idea they add**, not by rating (the same rule as the practice ladders).
5. **Close the loop.** Before the toolkit/pattern map, a short **Looking back** paragraph names the thread that ran through every pattern.
6. **Reuse instead of repeating.** When a later chapter needs a problem already solved, include its card and say what is new; never solve it twice.

## Animations (2–15 frames)

Use an animation when the **order of events** is the point: a table filling, a pointer moving, a state machine stepping, a window sliding. A static figure (or a `V.story` board) is better when the reader must compare several states side by side.

````markdown
::: {#fig-<prefix>-anim-name}

```{python}
#| echo: false
def frame(ax, p, k):              # draw frame k (0-based) with the same V.* primitives as figures
    ...
V.animate(frame, n_frames, captions=[...], interval=900)   # 2 ≤ n_frames ≤ 15
```

Caption: what moves, and what to watch.

:::
````

- Compute the frames by running the algorithm in the cell, and `assert` the final state — the animation is checked like the code.
- One caption per frame says what just happened (`dp[5] = dp[4] + dp[3] = 5 + 3 = 8`).
- No `renderings` option: `V.animate` already renders both themes, and `site.js` shows the matching one, with play/pause/step controls; it autoplays when scrolled into view (never for readers who ask for reduced motion).
- Keep it to one idea and at most 15 frames; split a longer process into two animations.

## Page features (site.js)

- **3D cards and depth.** Volume and problem cards tilt slightly under the pointer; problem blocks lift on hover. Nothing to write — the classes do it.
- **Practice tracker.** Every link in a `.practice` list and in the problem index gets a checkbox, saved in the reader's browser, with a solved / total count per list. The problem index can export and import that progress as a JSON file (merged, never overwritten).
- **Chapter badge.** Real chapter number, volume, part, reading time, and `status` from `outline.yml`. Browser tab titles and the search index (`fix_search.py`, run by `build.ps1`) use the same numbers.
- **Read first / Next chapter.** From `needs:` in `outline.yml` (default: the previous chapter of the part) and the next written chapter. Part opener pages (`books/parts/`, generated) list each part's chapters and what to read before it.
- **Revision pages**, generated from the chapters by `gen_book.py`: *Cheat Sheets* (each chapter's Summary template callout + Pattern map), *Question Bank* (each chapter's Interview questions, with a quiz mode), *Glossary* (from `glossary.yml`: add a term there with the section label that introduces it). So keep those three sections in the standard shape: a `callout-important` in the Summary, a table in the Pattern map, and `**N. Question?**` + a collapsed `Answer` callout per interview question.
- **Problem browser.** The problem index has a filter bar: text, role (solved/practice), your progress, and a rating range.
- **Run.** Plain Python listings (no drawing or trace imports) get a *Run* button that executes them in the browser with Pyodide, loaded on first click. So keep a listing self-contained when you can.
- **Accessibility.** Animation frames carry their caption as alt text; a focused animation steps with ← → and plays with space; keyboard focus is outlined. Write captions that describe the state, not "see above".
- **3D tables.** `V.iso_grid(ax, p, (ni, nj, nk), values=…, colors=…)` draws `dp[i][j][k]` as stacked isometric plates. Use it for three-index DP; see the Visual Gallery appendix for every primitive.

## Coordination between sessions

Several sessions (and people) work on this book. To avoid silent overwrites:

- **A chapter has one owner at a time**: `owner:` on the chapter in `outline.yml`. Set it (and `status: drafting`) before editing a chapter; clear it when done. Never edit a chapter another owner holds.
- **Registry problems belong to their home chapter's owner.** To reuse a problem from another chapter, include its card or solution — never edit the other chapter's `problem.qmd`.
- **Shared files have one owner** — `scripts/`, `books/shared/`, `templates/`, `STYLE.md`, `outline.yml` structure: the structure session. Other sessions propose changes instead of editing them in place.
- `status` moves `planned → drafting → written → audited`. A chapter is **audited** only after the quality checklist below passes, including `check_cpp.py`.

## C++ listings compile

`scripts/check_cpp.py` compiles every ```` ```cpp ```` listing (with `int main` as a program, otherwise as a translation unit after a prelude with `<bits/stdc++.h>`, `using namespace std;`, `ll`, `ListNode`, `TreeNode`). A listing must be self-contained: define the constants (`MOD`) and helpers it uses. A deliberately partial listing starts with `// fragment` and is skipped. On a compiler older than GCC 7, listings with structured bindings are reported as *unverified*, not failed.

## Claims in prose

State only what the code in the chapter checks, what a cited source says, or what is common knowledge in the field. A judge-specific number (stack size, time limit, a library's internals) is either cited or softened ("on most judges", "typically"). Never invent a statistic, a rating, or a problem ID.

## Files and numbering

A chapter lives at `books/<folder>/NNN_<slug>.qmd`, where `NNN` is its number and `<folder>` its part's folder in `outline.yml`. Do not add chapters to `_quarto.yml` by hand: `scripts/gen_book.py` (run by `build.ps1`) adds them to the sidebar under their part, marks them as written on the home page, generates registry cards, and, from Part II on, indexes their judge links in the problem index.

## Verifying a chapter

```bash
.venv/Scripts/python scripts/check_chapter.py books/foundations/001_complexity.qmd   # runs every Python cell
.venv/Scripts/python scripts/check_style.py books/foundations/001_complexity.qmd     # structural lint (see below)
.venv/Scripts/python scripts/check_links.py --online                                  # checks every judge link
powershell scripts/build.ps1                                                          # regenerates and renders
```

`check_links.py` checks Codeforces id, name and rating against the cached problemset (`--refresh` downloads a new copy). With `--online` it also checks the LeetCode number, title and difficulty, and the CSES and AtCoder task names (cached in `scripts/.cache/judge_names.json`). A chapter is not done until all three commands pass.

Faster loops while writing:

```bash
.venv/Scripts/python scripts/check_changed.py                    # check_chapter on chapters whose content or includes changed
.venv/Scripts/python scripts/check_cpp.py books/<folder>/NNN_x.qmd  # compile the C++ tabs
.venv/Scripts/python scripts/review_figs.py books/<folder>/NNN_x.qmd  # light + dark contact sheets in scripts/.cache/figs/
```

`check_style.py` is the structural lint that guards the mistakes we keep repeating. Run with no arguments it checks every chapter; `build.ps1` runs it before render and **stops the build on any ERROR**. It reports:

- **ERROR** a local path (`D:\…`) in rendered text — dead link for a web reader.
- **ERROR** an unbalanced fenced div (`:::` opened but never closed, or a stray `:::`).
- **ERROR** a `::: {…}` opening fence with no blank line above it — it renders as literal text.
- **ERROR** `V.show(draw)` indented inside `def draw` — it never runs and the figure silently disappears.
- **WARN** no `## Further reading`, or a Further-reading bullet that is not a clickable `[…](https://…)` link.
- **WARN** a chapter longer than about 120 pages (400 prose words or 50 code lines a page, includes expanded): split it into two chapters in `outline.yml`. Never compress.

Errors block; warnings do not (yet) — but a chapter is not "audited" until its warnings are cleared too.

## Chapter quality checklist

Work top to bottom when auditing a chapter. Each line maps to something that has actually gone wrong.

1. **Runs clean.** `check_chapter.py` passes (every `{python}` cell executes).
2. **No dead links.** `check_style.py` shows 0 ERRORs for this file — no `D:\…` anywhere in Sources or on the page.
3. **Fenced divs balanced.** No Quarto "Failed To Evaluate"/fenced-div warning for this chapter.
4. **Further reading present and linked.** A `## Further reading` section exists with 4–8 bullets, each a *verified, clickable* hyperlink with a clause on what it covers. Never a private note/PDF path.
5. **Prose flows for a beginner.** Every theory section opens with the plain-words idea before the formal definition; a reader never needs to look up a term to follow the next line (see *Language: beginner-friendly first*).
6. **Every dense section has a signpost.** An "In plain words" / warm-up callout where a section is unusually formal.
7. **Figures earn their place.** Each dense idea that gains from a picture has one; any figure the prose leans on is referenced with `@fig-…`.
8. **Cross-refs intact.** `@sec-…`/`@fig-…`/`@eq-…` all resolve after render (no "Could not resolve reference").
