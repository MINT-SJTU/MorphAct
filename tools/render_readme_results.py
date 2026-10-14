"""Render the README's result graphics using only the Python standard library."""

from collections import Counter
from decimal import Decimal
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = (("LIBERO", "libero"), ("Meta-World", "metaworld"))
WIDTH, HEIGHT = 1024, 354
COL_WIDTHS = (108, 138, 96, 96, 96, 96, 102, 128, 116)
INK, MUTED, ACCENT = "#25323A", "#59636E", "#14745A"


RESULTS = {
    "LIBERO": {
        "columns": ["Spatial", "Object", "Goal", "Long", "Mean"],
        "rows": [
            ["X-VLA", "Baseline", "95.40", "96.60", "96.00", "84.20", "93.05", "—"],
            ["X-VLA", "MorphAct", "98.67", "99.87", "97.07", "93.20", "97.20", "0.0300"],
            ["π0", "Baseline", "96.80", "98.80", "95.80", "85.20", "94.15", "—"],
            ["π0", "MorphAct", "98.00", "99.60", "96.60", "92.40", "96.65", "0.0517"],
            ["π0.5", "Baseline", "98.80", "98.20", "98.00", "92.40", "96.85", "—"],
            ["π0.5", "MorphAct", "98.33", "99.67", "98.87", "97.53", "98.60", "0.0525"],
        ],
    },
    "Meta-World": {
        "columns": ["Easy", "Medium", "Hard", "Very Hard", "Avg."],
        "rows": [
            ["X-VLA", "Baseline", "72.50", "46.36", "48.30", "55.00", "55.54", "—"],
            ["X-VLA", "MorphAct", "83.99", "58.18", "55.28", "55.00", "63.11", "1.8400"],
            ["π0", "Baseline", "77.90", "51.80", "53.30", "20.00", "50.75", "—"],
            ["π0", "MorphAct", "82.44", "71.36", "56.39", "58.00", "67.05", "0.6342"],
            ["π0.5", "Baseline", "70.50", "45.00", "38.30", "41.00", "48.70", "—"],
            ["π0.5", "MorphAct", "85.77", "59.85", "66.39", "45.67", "64.42", "0.3970"],
        ],
    },
}


def render(headers, rows, benchmark, language):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
             f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
             f'<title id="title">{escape(benchmark)} results</title>',
             '<desc id="desc">Baseline and MorphAct success rates grouped by backbone. '
             'All source values are retained. Gains are absolute percentage-point differences.</desc>',
             '<style>text{font-family:Arial,Helvetica,"PingFang SC",sans-serif;'
             'font-variant-numeric:tabular-nums}</style>',
             f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#FFFFFF"/>']
    edges = [24]
    for width in COL_WIDTHS:
        edges.append(edges[-1] + width)

    def rectangle(x, y, width, height, fill):
        parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}"/>')

    def text(x, y, value, size=18, weight=400, fill=INK, anchor="end", kind=None):
        semantic = f' data-kind="{kind}"' if kind else ""
        parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
                     f'dominant-baseline="middle" font-size="{size}" font-weight="{weight}" '
                     f'fill="{fill}"{semantic}>{escape(value)}</text>')

    labels = headers[:]
    labels[6] += " ↑"
    labels.append("Gain" if language == "en" else "提升")
    rectangle(24, 18, 976, 44, "#35444B")
    for col in (6, 8):
        rectangle(edges[col], 18, COL_WIDTHS[col], 44, ACCENT)
    for col, label in enumerate(labels):
        if col < 2:
            text(edges[col] + 16, 40, label, 16, 600, "#FFFFFF", "start")
        else:
            text((edges[col] + edges[col + 1]) / 2, 40, label, 16, 600, "#FFFFFF", "middle")

    for group, (baseline, morphact) in enumerate(zip(rows[::2], rows[1::2])):
        top = 62 + group * 96
        rectangle(edges[1], top + 42, 868, 42, "#EDF5EF")
        text(78, top + 42, baseline[0], 20, 600, INK, "middle")
        for offset, row in enumerate((baseline, morphact)):
            y = top + 21 + offset * 42
            is_morphact = offset == 1
            text(edges[1] + 16, y, row[1], 17, 600 if is_morphact else 400,
                 ACCENT if is_morphact else MUTED, "start")
            for col in range(2, 8):
                winner = col < 7 and Decimal(row[col]) > Decimal((baseline, morphact)[1 - offset][col])
                mean = col == 6
                text(edges[col + 1] - 16, y, row[col], 21 if mean and is_morphact else 18,
                     700 if winner else 400,
                     ACCENT if mean and is_morphact else INK if is_morphact or winner else MUTED,
                     kind="value")
            gain = f"{Decimal(morphact[6]) - Decimal(baseline[6]):+.2f}" if is_morphact else "—"
            text(edges[9] - 16, y, gain, 20 if is_morphact else 18,
                 700 if is_morphact else 400, ACCENT if is_morphact else MUTED, kind="gain")

    parts.append("</svg>")
    svg = "\n".join(parts) + "\n"
    parsed = ET.fromstring(svg)
    drawn_values = [node.text for node in parsed.iter() if node.get("data-kind") == "value"]
    if Counter(drawn_values) != Counter(value for row in rows for value in row[2:]):
        raise ValueError("Rendered numeric values differ from the source table")
    return svg


def main():
    for benchmark, slug in BENCHMARKS:
        data = RESULTS[benchmark]
        for language in ("en", "zh"):
            headers = (["Backbone", "Method"] if language == "en" else ["骨干", "方法"])
            headers += data["columns"] + ["Var. (pp²)" if language == "en" else "方差 (pp²)"]
            target = ROOT / "assets" / f"results-{slug}-{language}.svg"
            target.write_text(render(headers, data["rows"], benchmark, language), encoding="utf-8")
            print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
