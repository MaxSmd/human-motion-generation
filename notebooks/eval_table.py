"""Generate a LaTeX comparison table (MoMask, MARDM, RMG, MARDM-mini) and render to PDF.

Bold = best per column. Underline = second-best. Diversity uses
"closer to GT (9.503) is better"; the others use min/max as the arrows indicate.

Usage:
    python notebooks/eval_table.py ~/rmg-runs/mardm-eval-latest/eval/results.json
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

# GT diversity reference for the "Div→ closest-to-ideal" ranking.
GT_DIVERSITY = 9.503

# RMG paper baseline numbers (HumanML3D test split, Guo evaluator).
BASELINES = [
    ("MoMask [2024]",    {"FID": 0.045, "R@1": 0.521, "Diversity": None,  "MultiModality": 1.241}),
    ("MARDM [2025]",     {"FID": 0.114, "R@1": 0.500, "Diversity": None,  "MultiModality": 2.231}),
    ("RMG (paper)",      {"FID": 0.043, "R@1": 0.525, "Diversity": 9.555, "MultiModality": 2.748}),
]

COLS  = ["FID",            "R@1",        "Diversity",       "MultiModality"]
HEADS = ["FID$\\downarrow$", "R@1$\\uparrow$", "Div$\\rightarrow$", "MMod$\\uparrow$"]
RULES = ["min",            "max",        "target",          "max"]


def rank_indices(values, mode):
    pairs = [(i, v) for i, v in values if v is not None]
    if not pairs:
        return None, None
    if mode == "min":
        pairs.sort(key=lambda x: x[1])
    elif mode == "max":
        pairs.sort(key=lambda x: -x[1])
    else:  # target
        pairs.sort(key=lambda x: abs(x[1] - GT_DIVERSITY))
    best   = pairs[0][0]
    second = pairs[1][0] if len(pairs) > 1 else None
    return best, second


def fmt(val, role):
    if val is None:
        return "--"
    s = f"{val:.3f}"
    if role == "best":
        return rf"\textbf{{{s}}}"
    if role == "second":
        return rf"\underline{{{s}}}"
    return s


def load_ours(path: Path):
    data = json.loads(path.read_text())
    rows = []
    for omega in sorted(data.keys(), key=float):
        m = data[omega]
        rp = m.get("r_precision") or [None]
        rows.append((
            f"MARDM-mini (ours, $w={omega}$)",
            {
                "FID":           m.get("fid"),
                "R@1":           rp[0] if rp else None,
                "Diversity":     m.get("diversity"),
                "MultiModality": m.get("multimodality"),
            },
        ))
    return rows


def build_tex(rows):
    bests, seconds = {}, {}
    for col, mode in zip(COLS, RULES):
        col_vals = [(i, r[1].get(col)) for i, r in enumerate(rows)]
        b, s = rank_indices(col_vals, mode)
        bests[col], seconds[col] = b, s

    body_lines = []
    n_baselines = len(BASELINES)
    for i, (name, metrics) in enumerate(rows):
        cells = [name]
        for col in COLS:
            role = "best" if bests[col] == i else ("second" if seconds[col] == i else "normal")
            cells.append(fmt(metrics.get(col), role))
        line = " & ".join(cells) + r" \\"
        body_lines.append(line)
        # divider between baseline rows and our row(s)
        if i == n_baselines - 1 and len(rows) > n_baselines:
            body_lines.append(r"\midrule")

    table = (
        "\\begin{tabular}{l|cccc}\n"
        "\\toprule\n"
        "\\multirow{2}{*}{\\textbf{Method}} & "
        "\\multicolumn{4}{c}{\\textbf{HumanML3D Format}} \\\\\n"
        "\\cmidrule(lr){2-5}\n"
        " & " + " & ".join(HEADS) + " \\\\\n"
        "\\midrule\n"
        + "\n".join(body_lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    doc = (
        "\\documentclass[border=4pt]{standalone}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{multirow}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\begin{document}\n"
        + table +
        "\\end{document}\n"
    )
    return table, doc


def main(results_path: Path) -> None:
    ours = load_ours(results_path)
    rows = list(BASELINES) + ours
    table, doc = build_tex(rows)

    out_dir = results_path.parent
    tex_path  = out_dir / "comparison.tex"
    doc_path  = out_dir / "_comparison_doc.tex"
    pdf_path  = out_dir / "comparison.pdf"

    tex_path.write_text(table)
    doc_path.write_text(doc)

    print("\n=== LaTeX table ===\n")
    print(table)

    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        print(
            "\npdflatex not found — only wrote the .tex source.\n"
            "Install on macOS with: brew install --cask basictex\n"
            "Then `eval $(/usr/libexec/path_helper)` (or restart your shell)\n"
            "and re-run this script."
        )
        return

    try:
        subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-output-directory", str(out_dir), str(doc_path)],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print("\npdflatex failed:\n", e.stdout.decode()[-2000:], file=sys.stderr)
        return

    standalone_pdf = out_dir / "_comparison_doc.pdf"
    if standalone_pdf.exists():
        standalone_pdf.replace(pdf_path)
    for ext in (".aux", ".log"):
        leftover = out_dir / ("_comparison_doc" + ext)
        if leftover.exists():
            leftover.unlink()
    doc_path.unlink(missing_ok=True)

    print(f"\nWrote: {tex_path}")
    print(f"Wrote: {pdf_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: eval_table.py <results.json>")
    main(Path(sys.argv[1]).expanduser())
