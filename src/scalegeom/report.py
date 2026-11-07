"""Self-contained reports: tables, figures, and the settings that produced them."""
from __future__ import annotations

import base64
import datetime as _dt
import html
import io
import json
import os

import numpy as np

CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:980px;margin:24px auto;
padding:0 16px;color:#1f2a44;background:#fff;line-height:1.45}
h1{font-size:22px;margin-bottom:4px}h2{font-size:16px;margin-top:28px;border-bottom:1px solid #e5e7eb;padding-bottom:4px}
table{border-collapse:collapse;font-size:12px;margin:8px 0;display:block;overflow-x:auto}
caption{text-align:left;font-weight:600;padding:4px 0}
th,td{border-bottom:1px solid #eceef2;padding:4px 8px;text-align:right;white-space:nowrap}
th{background:#f6f7f9;font-weight:600}td:first-child,th:first-child{text-align:left}
.note{font-size:11px;color:#6b7280;margin-top:2px}figure{margin:14px 0}img{max-width:100%}
figcaption{font-size:12px;color:#4b5563}.meta{font-size:12px;color:#4b5563}code{font-size:11px}
"""


class Report:
    def __init__(self, title: str):
        self.title = title
        self.tables = {}
        self.figures = {}
        self.captions = {}
        self.meta = {"created": _dt.datetime.now().isoformat(timespec="seconds")}
        self.notes = []
        self.payload = {}

    def add_table(self, name, table):
        self.tables[name] = table

    def add_figure(self, name, fig, caption=""):
        self.figures[name] = fig
        self.captions[name] = caption

    def __getitem__(self, name):
        return self.tables[name]

    def __repr__(self):
        return "\n\n".join([self.title] + [repr(t) for t in self.tables.values()])

    def _png(self, fig, dpi=150):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        FigureCanvasAgg(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        return buf.getvalue()

    def to_html(self, path=None) -> str:
        parts = [f"<!doctype html><html><head><meta charset='utf-8'>"
                 f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
                 f"<title>{html.escape(self.title)}</title><style>{CSS}</style></head><body>",
                 f"<h1>{html.escape(self.title)}</h1>"]
        meta = ", ".join(f"{html.escape(str(k))}: <code>{html.escape(_short(v))}</code>"
                         for k, v in self.meta.items())
        parts.append(f"<p class='meta'>{meta}</p>")
        for n in self.notes:
            parts.append(f"<p class='note'>{html.escape(n)}</p>")
        for name, t in self.tables.items():
            parts.append(f"<h2>{html.escape(t.title or name)}</h2>{t.to_html()}")
        if self.figures:
            parts.append("<h2>Figures</h2>")
            for name, fig in self.figures.items():
                b64 = base64.b64encode(self._png(fig)).decode()
                parts.append(f"<figure><img alt='{html.escape(name)}' src='data:image/png;base64,{b64}'>"
                             f"<figcaption>{html.escape(self.captions.get(name, ''))}</figcaption></figure>")
        parts.append("<p class='note'>Method: Vaz, E. (2026). Scale-conditioned geometry: how measurement "
                     "resolution induces distance, proximity, and spatial dependence. "
                     "Geographical Analysis 58: e70061. https://doi.org/10.1111/gean.70061</p></body></html>")
        text = "\n".join(parts)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    def save(self, directory: str, figures: bool = True):
        """Write every table as CSV, figures as PNG, settings as JSON, and the HTML report."""
        os.makedirs(directory, exist_ok=True)
        for name, t in self.tables.items():
            t.to_csv(os.path.join(directory, f"{name}.csv"))
        if figures:
            for name, fig in self.figures.items():
                with open(os.path.join(directory, f"{name}.png"), "wb") as fh:
                    fh.write(self._png(fig, dpi=300))
        with open(os.path.join(directory, "settings.json"), "w", encoding="utf-8") as fh:
            json.dump(self.meta, fh, indent=2, default=_jsonable)
        self.to_html(os.path.join(directory, "report.html"))
        return directory


def _jsonable(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def _short(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, dict):
        return ", ".join(f"{k}={_short(x)}" for k, x in v.items())
    return str(v)
