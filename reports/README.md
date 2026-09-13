# Reports

Self-contained HTML pages, each explaining one part of reimp at a glance.
Open them in a browser; they need nothing but their fonts from Google Fonts.

| report | what it covers | rebuild |
|---|---|---|
| [`shared_data.html`](shared_data.html) | the TCGA cohort, its six quantifications and library statistics, the default gene selection, the 5-fold patient-level cross-validation, and how evenly projects fall across the folds | `uv run python reports/build/shared_data.py` |
| [`baselines.html`](baselines.html) | every shared probe on the PCA baseline at 4 to 256 dimensions, pooled over the folds and fold by fold, beside what each metric gives with no embedding | `uv run python reports/build/baselines.py` |

Every number on a page is computed from the data by its script in `build/`:
the script fills its template's `{{name}}` placeholders and embeds the chart
data as JSON. Rebuild a page after the dataset or the splits change rather
than editing it; `uv run pytest` builds each report against the miniature
test dataset.

The baselines script first fits and scores whatever baseline is missing or
stale under `out/` (a few minutes from scratch, seconds after), and
`--rerun` recomputes all of them. Pages share their styles and chart
helpers, `build/report.css` and `build/report.js`, which `build/reportkit.py`
inlines into each.
