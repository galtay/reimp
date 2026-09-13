"""What every report's build script shares: filling a template into a self-contained page.

A template holds the prose, with `{{name}}` placeholders for its numbers
and TeX between `\\( \\)` (inline) or `\\[ \\]` (display) for its equations,
which are rendered to MathML, so the page needs no script to show them.
Three markers: `__REPORT_CSS__` and `__REPORT_JS__`, where the styles and
chart helpers every report shares are inlined, and `__REPORT_DATA__`,
where the page's chart data is embedded as JSON.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np
from latex2mathml.converter import convert

HERE = Path(__file__).resolve().parent
PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
ASSETS = {"__REPORT_CSS__": HERE / "report.css", "__REPORT_JS__": HERE / "report.js"}
# Script and style blocks are left alone; math is found only in the prose around them.
CODE = re.compile(r"(<script\b.*?</script>|<style\b.*?</style>)", re.S)
MATH = re.compile(r"\\\((.+?)\\\)|\\\[(.+?)\\\]", re.S)


def native(value):
    """numpy scalars as JSON values."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def mathml(text: str) -> str:
    """`text` with each `\\( tex \\)` and `\\[ tex \\]` outside scripts and styles as MathML."""

    def one(match: re.Match) -> str:
        inline, display = match.groups()
        tex = html.unescape(inline if inline is not None else display).strip()
        out = convert(tex, display="inline" if inline is not None else "block")
        # The converter passes a command it does not know through as text.
        if "\\" in out:
            raise ValueError(f"TeX the converter does not understand: {tex!r}")
        return out

    parts = CODE.split(text)
    return "".join(part if CODE.fullmatch(part) else MATH.sub(one, part) for part in parts)


def render(data: dict, facts: dict[str, str], template: str) -> str:
    """The template filled, its math rendered, the shared assets inlined, the data embedded."""
    missing = sorted(set(PLACEHOLDER.findall(template)) - set(facts))
    if missing:
        raise KeyError(f"template placeholders without a value: {missing}")
    page = mathml(PLACEHOLDER.sub(lambda m: html.escape(facts[m.group(1)]), template))
    for marker, path in ASSETS.items():
        page = page.replace(marker, path.read_text().strip())
    payload = json.dumps(data, default=native, allow_nan=False, separators=(",", ":"))
    return page.replace("__REPORT_DATA__", payload.replace("</", "<\\/"))
