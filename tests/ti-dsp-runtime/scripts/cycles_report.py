"""Generate an HTML regression report comparing two DSP cycles.csv files."""

import argparse
import csv
import html
import os
import sys
from pathlib import Path


def parse_csv(path: str) -> dict[str, int]:
    """Parse cycles.csv (header row + one data row) into {name: cycles}."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        values = next(reader, None)
    if not headers or not values:
        return {}
    result = {}
    for name, val in zip(headers, values):
        name = name.strip()
        val = val.strip()
        if name and val:
            try:
                result[name] = int(val)
            except ValueError:
                pass
    return result


def _status(pct_change: float | None, threshold: float) -> tuple[str, str]:
    """Return (label, css-class) for a row."""
    if pct_change is None:
        return "NEW", "new"
    if pct_change > threshold:
        return "REGRESSED", "regressed"
    if pct_change < -threshold:
        return "IMPROVED", "improved"
    return "OK", "ok"


def build_html(
    current: dict[str, int],
    previous: dict[str, int],
    threshold: float,
    build_number: str,
    git_commit: str,
) -> str:
    rows = []
    for name, cur_cycles in current.items():
        prev_cycles = previous.get(name)
        if prev_cycles is not None and prev_cycles > 0:
            pct = (cur_cycles - prev_cycles) / prev_cycles * 100.0
        else:
            pct = None
        label, css = _status(pct, threshold)
        rows.append((label, name, cur_cycles, prev_cycles, pct, css))

    # Sort: REGRESSED first, then alphabetical by name
    def sort_key(r):
        order = {"REGRESSED": 0, "NEW": 1, "IMPROVED": 2, "OK": 3}
        return (order.get(r[0], 9), r[1].lower())

    rows.sort(key=sort_key)

    def fmt_cycles(v: int | None) -> str:
        if v is None:
            return "&mdash;"
        return f"{v:,}"

    def fmt_change(pct: float | None) -> str:
        if pct is None:
            return "&mdash;"
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.1f}%"

    tbody = ""
    for label, name, cur, prev, pct, css in rows:
        tbody += (
            f'    <tr class="{html.escape(css)}">'
            f"<td>{html.escape(name)}</td>"
            f"<td class='num'>{fmt_cycles(cur)}</td>"
            f"<td class='num'>{fmt_cycles(prev)}</td>"
            f"<td class='num'>{fmt_change(pct)}</td>"
            f"<td class='status'>{html.escape(label)}</td>"
            f"</tr>\n"
        )

    has_prev = bool(previous)
    baseline_note = (
        f"Baseline: build {html.escape(build_number)} (previous successful)"
        if has_prev
        else "No baseline available — all rows show as NEW"
    )

    commit_short = git_commit[:12] if git_commit else "unknown"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DSP Cycles Regression Report</title>
<style>
  body {{ font-family: sans-serif; margin: 2em; }}
  h1 {{ font-size: 1.4em; }}
  .meta {{ color: #555; font-size: 0.9em; margin-bottom: 1em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
  th {{ background: #f0f0f0; }}
  td.num {{ text-align: right; font-family: monospace; }}
  td.status {{ font-weight: bold; }}
  tr.regressed td {{ background: #ffe0e0; }}
  tr.regressed td.status {{ color: #c00; }}
  tr.improved td {{ background: #e0ffe0; }}
  tr.improved td.status {{ color: #060; }}
  tr.new td {{ background: #fff8e0; }}
  tr.ok td {{ background: #fff; }}
</style>
</head>
<body>
<h1>DSP Cycles Regression Report</h1>
<div class="meta">
  Build: {html.escape(build_number)} &nbsp;|&nbsp;
  Commit: {html.escape(commit_short)} &nbsp;|&nbsp;
  Threshold: &plusmn;{threshold:.0f}% &nbsp;|&nbsp;
  {baseline_note}
</div>
<table>
<thead>
  <tr>
    <th>Test</th>
    <th>Current</th>
    <th>Previous</th>
    <th>Change</th>
    <th>Status</th>
  </tr>
</thead>
<tbody>
{tbody}</tbody>
</table>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="DSP cycles regression HTML report")
    parser.add_argument("--current", required=True, help="Path to current cycles.csv")
    parser.add_argument("--previous", default=None, help="Path to previous cycles.csv")
    parser.add_argument("--output", required=True, help="Output HTML path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Regression threshold in percent (default: 10)",
    )
    args = parser.parse_args()

    current = parse_csv(args.current)
    if not current:
        print(f"ERROR: could not parse {args.current}", file=sys.stderr)
        sys.exit(1)

    previous = parse_csv(args.previous) if args.previous else {}

    build_number = os.environ.get("BUILD_NUMBER", "N/A")
    git_commit = os.environ.get("GIT_COMMIT", "")

    report = build_html(current, previous, args.threshold, build_number, git_commit)
    Path(args.output).write_text(report)
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
