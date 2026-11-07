"""A minimal results table (no pandas required, so the core needs only numpy and scipy)."""
from __future__ import annotations

import csv
import html
import io
import math


class Table:
    def __init__(self, rows=None, columns=None, title: str = "", note: str = ""):
        self.rows = list(rows or [])
        self.columns = list(columns) if columns else (list(self.rows[0].keys()) if self.rows else [])
        self.title = title
        self.note = note

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r.get(key) for r in self.rows]
        return self.rows[key]

    def append(self, row: dict):
        for k in row:
            if k not in self.columns:
                self.columns.append(k)
        self.rows.append(row)

    def to_pandas(self):
        import pandas as pd
        return pd.DataFrame(self.rows, columns=self.columns)

    def to_csv(self, path=None) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=self.columns, extrasaction="ignore")
        w.writeheader()
        for r in self.rows:
            w.writerow({k: _fmt(r.get(k), csv_mode=True) for k in self.columns})
        text = buf.getvalue()
        if path:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                fh.write(text)
        return text

    def to_html(self) -> str:
        head = "".join(f"<th>{html.escape(str(c))}</th>" for c in self.columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(_fmt(r.get(c)))}</td>" for c in self.columns) + "</tr>"
            for r in self.rows)
        cap = f"<caption>{html.escape(self.title)}</caption>" if self.title else ""
        note = f"<p class='note'>{html.escape(self.note)}</p>" if self.note else ""
        return f"<table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{note}"

    def __repr__(self):
        widths = {c: max(len(str(c)), *(len(_fmt(r.get(c))) for r in self.rows)) if self.rows else len(str(c))
                  for c in self.columns}
        line = "  ".join(str(c).rjust(widths[c]) for c in self.columns)
        out = [self.title, line] if self.title else [line]
        for r in self.rows:
            out.append("  ".join(_fmt(r.get(c)).rjust(widths[c]) for c in self.columns))
        return "\n".join(out)


def _fmt(v, csv_mode=False):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if math.isnan(v):
            return "" if csv_mode else "NA"
        if csv_mode:
            return repr(float(v))
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.3e}"
        return f"{v:.3f}"
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return _fmt(v.item(), csv_mode)
    except ImportError:
        pass
    return str(v)
