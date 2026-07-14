"""
Regenerate the readable Jupyter notebooks in docs/ from the source .md files.

Each markdown SECTION (every header, split on the fly) becomes its own cell,
so the rendered notebook has clean spacing and reads one concept at a time.
ASCII diagrams stay inside fenced code blocks so they render monospace.

Usage:  .venv/bin/python docs/build_notebooks.py
"""
import json
import re
import pathlib

DOCS = pathlib.Path(__file__).resolve().parent
ROOT = DOCS.parent
HEADER = re.compile(r"^#{1,6} ")


def split_cells(md_text: str) -> list[str]:
    """Split markdown into one cell per header; drop `---` rules; trim blanks.
    Tracks ``` fences so headers inside code blocks don't trigger a split."""
    cells, cur, in_fence = [], [], False
    for line in md_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            cur.append(line)
            continue
        if not in_fence:
            if stripped == "---":            # horizontal rule -> cell boundary already
                continue
            if HEADER.match(line):           # a new header starts a new cell
                if any(x.strip() for x in cur):
                    cells.append(cur)
                cur = [line]
                continue
        cur.append(line)
    if any(x.strip() for x in cur):
        cells.append(cur)

    out = []
    for c in cells:
        while c and not c[0].strip():
            c.pop(0)
        while c and not c[-1].strip():
            c.pop()
        out.append("\n".join(c))
    return out


def to_ipynb(cells: list[str]) -> dict:
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": c.splitlines(keepends=True)}
            for c in cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    DOCS.mkdir(exist_ok=True)
    for name in ("ARCHITECTURE", "INTERVIEW"):
        cells = split_cells((ROOT / f"{name}.md").read_text())
        (DOCS / f"{name}.ipynb").write_text(json.dumps(to_ipynb(cells), indent=1))
        print(f"{name}.ipynb  ->  {len(cells)} cells")


if __name__ == "__main__":
    main()
