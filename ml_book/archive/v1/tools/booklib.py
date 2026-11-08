"""Write the Machine Learning Reference as a Quarto book."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOKS = ROOT / "books"

VOLUMES: list[dict] = []


def volume(number: int, title: str, stage: str, audience: str, question: str) -> dict:
    vol = {
        "number": number,
        "id": f"vol{number:02d}",
        "title": title,
        "stage": stage,
        "audience": audience,
        "question": question,
        "chapters": [],
    }
    VOLUMES.append(vol)
    return vol


def chapter(vol: dict, slug: str, title: str, body: str) -> None:
    vol["chapters"].append({"slug": slug, "title": title, "body": body.strip() + "\n"})


def _table_cell(text: str) -> str:
    # A raw pipe splits a markdown table. Keep already-escaped pipes intact.
    return text.replace("\\|", "\x00").replace("|", "\\|").replace("\x00", "\\|")


def entry(
    *,
    question: str,
    card: list[tuple[str, str]],
    theory: str,
    derivation: str,
    code: str,
    experiment: str,
    project: str,
    beginner: str,
    university: str,
    research: str,
    failures: str,
    see_also: str,
) -> str:
    rows = "\n".join(f"| {_table_cell(k)} | {_table_cell(v)} |" for k, v in card)
    return f"""**Reference question.** {question}

Use the card to look the result up. Use the rest of the chapter when you need to derive it, implement it, or decide whether the result still applies.

## Reference card

| Item | Statement |
|---|---|
{rows}

## Theory

{theory.strip()}

## Derivation

{derivation.strip()}

## Code

```python
{code.strip()}
```

## Experiment

{experiment.strip()}

## Project

{project.strip()}

## Depth tracks

**Beginner.** {beginner.strip()}

**University.** {university.strip()}

**Research.** {research.strip()}

## Failure modes

{failures.strip()}

## See also

{see_also.strip()}
"""


def _yaml_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def quarto_config() -> str:
    lines = [
        "project:",
        "  type: book",
        "  output-dir: _build",
        "",
        "book:",
        '  title: "Machine Learning Reference"',
        '  subtitle: "A guide from first principles to research: theory, code, experiments, and projects"',
        '  author: "Personal curriculum"',
        "  date: today",
        "  chapters:",
        "    - index.qmd",
        "    - 00_how_to_study.qmd",
        "    - glossary.qmd",
        "    - upstream.qmd",
    ]
    for vol in VOLUMES:
        part = f"Volume {vol['number']:02d}. {vol['title']}"
        lines.append(f"    - part: {_yaml_string(part)}")
        lines.append("      chapters:")
        lines.append(f"        - {vol['id']}/index.qmd")
        for ch in vol["chapters"]:
            lines.append(f"        - {vol['id']}/{ch['slug']}.qmd")
    lines += [
        "",
        "bibliography: references.bib",
        "",
        "format:",
        "  html:",
        "    theme: cosmo",
        "    css: shared/reference.css",
        "    toc: true",
        "    toc-depth: 3",
        "    number-sections: true",
        "    number-depth: 3",
        "    code-copy: true",
        "    code-overflow: wrap",
        "    smooth-scroll: true",
        "    link-external-newwindow: true",
        "  pdf:",
        "    documentclass: scrreprt",
        "    papersize: a4",
        "    toc: true",
        "    number-sections: true",
        "    colorlinks: true",
        "    include-in-header:",
        "      text: |",
        "        \\usepackage{booktabs}",
        "        \\usepackage{longtable}",
        "        \\usepackage{amsmath,amssymb}",
        "  epub:",
        "    toc: true",
        "    number-sections: true",
        "",
    ]
    return "\n".join(lines)


def volume_index(vol: dict) -> str:
    rows = "\n".join(
        f"| {i:02d} | {_table_cell(ch['title'])} | `{vol['id']}/{ch['slug']}.qmd` |"
        for i, ch in enumerate(vol["chapters"], start=1)
    )
    return f"""---
title: "Volume {vol['number']:02d}. {vol['title']}"
---

# Volume {vol['number']:02d}. {vol['title']}

**Stage.** {vol['stage']}

**Audience.** {vol['audience']}

**Question this volume answers.** {vol['question']}

This volume is a reference, not a short tour. Each chapter opens with a card you can look up later, then gives the theory, a reference implementation, an experiment with a decision rule, and a project with a rubric. Read in order the first time. After that, enter through the reference card of the chapter that matches the problem in front of you.

## How to use the four layers

| Layer | What you should be able to do when you finish it |
|---|---|
| Theory | State the definition, the assumptions, and the result, including what the result does not say. |
| Code | Implement the core object in a few dozen lines and check it against a known case. |
| Experiment | Change one factor, record a metric with uncertainty, and say what outcome would change your mind. |
| Project | Produce an artifact someone else can rerun: code, figure, table, and a one-page lab note. |

## Depth tracks

Every chapter ends with three tracks.

- **Beginner.** Do the reference card, the code, and the smallest experiment. Skip optional proofs.
- **University.** Do the derivation, the full experiment, and the project rubric.
- **Research.** Read the failure modes, change an assumption, and write what would have to be true for the method to fail on a new distribution.

## Chapters

| No. | Chapter | File |
|---|---|---|
{rows}

## Lab note

Keep one markdown file per experiment, `notes/YYYY-MM-DD-<slug>.md`, with the hypothesis, the command you ran, the metric, and the decision. The series is a reference only if the experiments you run are attached to it.
"""


def write_all() -> None:
    BOOKS.mkdir(parents=True, exist_ok=True)
    (BOOKS / "shared").mkdir(exist_ok=True)
    config = quarto_config()
    (BOOKS / "_quarto.yml").write_text(config, encoding="utf-8")
    (BOOKS / "config").mkdir(exist_ok=True)
    for name, fmt in (
        ("_quarto-html.yml", "html"),
        ("_quarto-pdf.yml", "pdf"),
        ("_quarto-epub.yml", "epub"),
    ):
        # Same source list. Build one format at a time with `quarto render --to`.
        (BOOKS / "config" / name).write_text(
            config + f"# Build with: quarto render --to {fmt}\n",
            encoding="utf-8",
        )
    for vol in VOLUMES:
        folder = BOOKS / vol["id"]
        folder.mkdir(exist_ok=True)
        (folder / "index.qmd").write_text(volume_index(vol), encoding="utf-8")
        for ch in vol["chapters"]:
            text = f"---\ntitle: {_yaml_string(ch['title'])}\n---\n\n# {ch['title']}\n\n{ch['body']}"
            (folder / f"{ch['slug']}.qmd").write_text(text, encoding="utf-8")
    print(f"Wrote {sum(len(v['chapters']) for v in VOLUMES)} chapters in {len(VOLUMES)} volumes.")
