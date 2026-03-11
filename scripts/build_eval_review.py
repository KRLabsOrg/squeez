"""Build a self-contained HTML review app from eval examples JSON.

Usage:
    python scripts/build_eval_review.py \
        --examples eval_examples.json \
        --output eval_review.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Squeez Eval Review</title>
  <style>
    :root {{
      --bg: #f4f1e8;
      --panel: #fffdf7;
      --ink: #1f1c16;
      --muted: #6d665b;
      --line: #d7cdbd;
      --accent: #005f73;
      --accent-soft: #dff4f6;
      --good: #2a9d8f;
      --bad: #bc4749;
      --warn: #c77d00;
      --mono: "Iosevka", "SFMono-Regular", ui-monospace, monospace;
      --sans: "IBM Plex Sans", "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff8df 0%, transparent 35%),
        linear-gradient(135deg, #efe8da 0%, #f7f4ec 55%, #ebe2d3 100%);
      min-height: 100vh;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 360px 1fr;
      min-height: 100vh;
    }}
    .sidebar {{
      border-right: 1px solid var(--line);
      background: rgba(255, 253, 247, 0.86);
      backdrop-filter: blur(10px);
      padding: 18px;
      overflow: auto;
      position: sticky;
      top: 0;
      height: 100vh;
    }}
    .main {{
      padding: 24px;
      overflow: auto;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 1.45rem;
      letter-spacing: -0.03em;
    }}
    .subtitle {{
      color: var(--muted);
      margin-bottom: 18px;
      font-size: 0.95rem;
    }}
    .controls {{
      display: grid;
      gap: 10px;
      margin-bottom: 18px;
    }}
    .controls label {{
      display: grid;
      gap: 6px;
      font-size: 0.88rem;
      color: var(--muted);
    }}
    input, select {{
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: white;
      font: inherit;
      color: var(--ink);
    }}
    .sample-list {{
      display: grid;
      gap: 10px;
    }}
    .sample-item {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 14px;
      padding: 12px;
      cursor: pointer;
      transition: 160ms ease;
    }}
    .sample-item:hover {{
      transform: translateY(-1px);
      border-color: var(--accent);
    }}
    .sample-item.active {{
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(0, 95, 115, 0.12);
      background: linear-gradient(180deg, #fff, var(--accent-soft));
    }}
    .sample-title {{
      font-weight: 600;
      font-size: 0.92rem;
      line-height: 1.25;
      margin-bottom: 8px;
    }}
    .sample-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      font-size: 0.78rem;
    }}
    .pill {{
      border-radius: 999px;
      padding: 3px 8px;
      background: #f2ece0;
      color: var(--muted);
    }}
    .pill.good {{ background: #dff6ee; color: #1d6c63; }}
    .pill.bad {{ background: #fde7e7; color: #8d2f30; }}
    .pill.warn {{ background: #fff0d7; color: #8a5900; }}
    .nav {{
      display: flex;
      gap: 8px;
      margin-bottom: 18px;
    }}
    button {{
      border: 0;
      background: var(--accent);
      color: white;
      border-radius: 10px;
      padding: 10px 14px;
      font: inherit;
      cursor: pointer;
    }}
    button.secondary {{
      background: #dfd6c5;
      color: #3c3427;
    }}
    .header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }}
    .header h2 {{
      margin: 0;
      font-size: 1.4rem;
      letter-spacing: -0.03em;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 20px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: var(--panel);
    }}
    .stat .label {{
      font-size: 0.78rem;
      color: var(--muted);
      margin-bottom: 4px;
    }}
    .stat .value {{
      font-size: 1.18rem;
      font-weight: 700;
    }}
    .panes {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-bottom: 16px;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--panel);
      overflow: hidden;
    }}
    .panel.full {{ grid-column: 1 / -1; }}
    .panel-head {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      font-size: 0.88rem;
      font-weight: 700;
      letter-spacing: 0.01em;
      background: #fbf7ef;
    }}
    .panel-body {{
      padding: 14px;
    }}
    .task {{
      white-space: pre-wrap;
      line-height: 1.45;
    }}
    .code {{
      white-space: pre-wrap;
      font-family: var(--mono);
      font-size: 0.86rem;
      line-height: 1.45;
      max-height: 360px;
      overflow: auto;
    }}
    .lines {{
      display: grid;
      gap: 8px;
    }}
    .line {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 9px 10px;
      font-family: var(--mono);
      font-size: 0.82rem;
      line-height: 1.4;
      background: white;
      white-space: pre-wrap;
    }}
    .line.pred {{ border-left: 4px solid var(--accent); }}
    .line.ref {{ border-left: 4px solid var(--good); }}
    .empty {{
      color: var(--muted);
      font-style: italic;
    }}
    .footer-note {{
      color: var(--muted);
      font-size: 0.85rem;
      margin-top: 8px;
    }}
    @media (max-width: 1100px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--line); }}
      .panes {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1>Squeez Review</h1>
      <div class="subtitle">Browse per-sample predictions, gold labels, and scoring details.</div>
      <div class="controls">
        <label>Search
          <input id="search" placeholder="task / prediction / gold">
        </label>
        <label>Sort
          <select id="sort">
            <option value="index">Original order</option>
            <option value="span_f1_asc">Lowest span F1</option>
            <option value="span_f1_desc">Highest span F1</option>
            <option value="fuzzy_f1_asc">Lowest fuzzy F1</option>
            <option value="fuzzy_f1_desc">Highest fuzzy F1</option>
            <option value="rouge_desc">Highest ROUGE-L</option>
            <option value="compression_desc">Highest compression</option>
          </select>
        </label>
        <label>Filter
          <select id="filter">
            <option value="all">All samples</option>
            <option value="bad_exact">Exact match = 0</option>
            <option value="bad_fuzzy">Fuzzy F1 &lt; 0.5</option>
            <option value="false_negative">False negatives</option>
            <option value="false_positive">False positives</option>
            <option value="empty_gold">Gold empty</option>
            <option value="nonempty_gold">Gold non-empty</option>
          </select>
        </label>
      </div>
      <div id="summary" class="footer-note"></div>
      <div id="sample-list" class="sample-list"></div>
    </aside>
    <main class="main">
      <div class="nav">
        <button class="secondary" id="prev-btn">Previous</button>
        <button class="secondary" id="next-btn">Next</button>
      </div>
      <div id="detail"></div>
    </main>
  </div>
  <script>
    const data = __DATA__;
    let view = data.map((sample, index) => ({{ ...sample, _index: index }}));
    let active = 0;

    function metricClass(value) {{
      if (value >= 0.8) return "good";
      if (value >= 0.5) return "warn";
      return "bad";
    }}

    function escapeHtml(text) {{
      return text
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}

    function truncate(text, maxLen = 110) {{
      return text.length <= maxLen ? text : text.slice(0, maxLen - 1) + "…";
    }}

    function applyFilters() {{
      const search = document.getElementById("search").value.trim().toLowerCase();
      const filter = document.getElementById("filter").value;
      const sort = document.getElementById("sort").value;

      view = data
        .map((sample, index) => ({{ ...sample, _index: index }}))
        .filter((sample) => {{
          const m = sample.metrics;
          const haystack = [
            sample.task,
            sample.tool_output,
            ...sample.predicted_lines,
            ...sample.reference_lines
          ].join("\\n").toLowerCase();
          if (search && !haystack.includes(search)) return false;
          if (filter === "bad_exact" && m.exact_match !== 0) return false;
          if (filter === "bad_fuzzy" && m.fuzzy_span_f1 >= 0.5) return false;
          if (filter === "false_negative" && m.empty_category !== "false_negative") return false;
          if (filter === "false_positive" && m.empty_category !== "false_positive") return false;
          if (filter === "empty_gold" && sample.reference_lines.length !== 0) return false;
          if (filter === "nonempty_gold" && sample.reference_lines.length === 0) return false;
          return true;
        }});

      const sorters = {{
        span_f1_asc: (a, b) => a.metrics.span_f1 - b.metrics.span_f1,
        span_f1_desc: (a, b) => b.metrics.span_f1 - a.metrics.span_f1,
        fuzzy_f1_asc: (a, b) => a.metrics.fuzzy_span_f1 - b.metrics.fuzzy_span_f1,
        fuzzy_f1_desc: (a, b) => b.metrics.fuzzy_span_f1 - a.metrics.fuzzy_span_f1,
        rouge_desc: (a, b) => b.metrics.rouge_l - a.metrics.rouge_l,
        compression_desc: (a, b) => b.metrics.compression - a.metrics.compression,
      }};
      if (sorters[sort]) view.sort(sorters[sort]);
      else view.sort((a, b) => a._index - b._index);

      if (active >= view.length) active = Math.max(view.length - 1, 0);
      renderList();
      renderDetail();
    }}

    function renderList() {{
      const list = document.getElementById("sample-list");
      const summary = document.getElementById("summary");
      summary.textContent = `${{view.length}} visible / ${{data.length}} total`;
      list.innerHTML = view.map((sample, i) => {{
        const m = sample.metrics;
        const title = truncate(sample.task.replace(/\\s+/g, " ").trim() || "(empty task)");
        return `
          <div class="sample-item ${{i === active ? "active" : ""}}" data-index="${{i}}">
            <div class="sample-title">${{escapeHtml(title)}}</div>
            <div class="sample-meta">
              <span class="pill ${{metricClass(m.span_f1)}}">F1 ${{m.span_f1.toFixed(3)}}</span>
              <span class="pill ${{metricClass(m.fuzzy_span_f1)}}">Fuzzy ${{m.fuzzy_span_f1.toFixed(3)}}</span>
              <span class="pill">EM ${{m.exact_match.toFixed(0)}}</span>
              <span class="pill">ROUGE ${{m.rouge_l.toFixed(3)}}</span>
            </div>
          </div>
        `;
      }}).join("");

      list.querySelectorAll(".sample-item").forEach(node => {{
        node.addEventListener("click", () => {{
          active = Number(node.dataset.index);
          renderList();
          renderDetail();
        }});
      }});
    }}

    function renderLines(lines, cls) {{
      if (!lines.length) return '<div class="empty">No lines.</div>';
      return `<div class="lines">${{lines.map(line => `<div class="line ${{cls}}">${{escapeHtml(line)}}</div>`).join("")}}</div>`;
    }}

    function renderDetail() {{
      const root = document.getElementById("detail");
      if (!view.length) {{
        root.innerHTML = '<div class="empty">No samples match the current filters.</div>';
        return;
      }}
      const sample = view[active];
      const m = sample.metrics;
      root.innerHTML = `
        <div class="header">
          <div>
            <h2>Sample ${{active + 1}} / ${{view.length}}</h2>
            <div class="subtitle">${{escapeHtml(sample.task.split("\\n")[0] || "(empty task)")}}</div>
          </div>
        </div>
        <div class="stats">
          <div class="stat"><div class="label">Span F1</div><div class="value">${{m.span_f1.toFixed(3)}}</div></div>
          <div class="stat"><div class="label">Fuzzy F1</div><div class="value">${{m.fuzzy_span_f1.toFixed(3)}}</div></div>
          <div class="stat"><div class="label">Exact Match</div><div class="value">${{m.exact_match.toFixed(0)}}</div></div>
          <div class="stat"><div class="label">ROUGE-L</div><div class="value">${{m.rouge_l.toFixed(3)}}</div></div>
          <div class="stat"><div class="label">Compression</div><div class="value">${{m.compression.toFixed(3)}}</div></div>
          <div class="stat"><div class="label">Empty Category</div><div class="value" style="font-size:.95rem">${{escapeHtml(m.empty_category)}}</div></div>
        </div>
        <div class="panel full">
          <div class="panel-head">Task</div>
          <div class="panel-body task">${{escapeHtml(sample.task)}}</div>
        </div>
        <div class="panel full">
          <div class="panel-head">Raw Prompt</div>
          <div class="panel-body code">${{escapeHtml(sample.prompt || "")}}</div>
        </div>
        <div class="panes">
          <div class="panel">
            <div class="panel-head">Predicted Lines</div>
            <div class="panel-body">${{renderLines(sample.predicted_lines, "pred")}}</div>
          </div>
          <div class="panel">
            <div class="panel-head">Reference Lines</div>
            <div class="panel-body">${{renderLines(sample.reference_lines, "ref")}}</div>
          </div>
          <div class="panel full">
            <div class="panel-head">Tool Output</div>
            <div class="panel-body code">${{escapeHtml(sample.tool_output)}}</div>
          </div>
        </div>
        <div class="footer-note">
          Strict span metrics use exact line matching. Fuzzy metrics use one-to-one line matches at overlap ≥ 0.5.
        </div>
      `;
    }}

    document.getElementById("search").addEventListener("input", applyFilters);
    document.getElementById("filter").addEventListener("change", applyFilters);
    document.getElementById("sort").addEventListener("change", applyFilters);
    document.getElementById("prev-btn").addEventListener("click", () => {{
      if (!view.length) return;
      active = (active - 1 + view.length) % view.length;
      renderList();
      renderDetail();
    }});
    document.getElementById("next-btn").addEventListener("click", () => {{
      if (!view.length) return;
      active = (active + 1) % view.length;
      renderList();
      renderDetail();
    }});
    applyFilters();
  </script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build HTML review app for eval examples")
    parser.add_argument("--examples", required=True, help="Path to examples JSON from squeez eval")
    parser.add_argument("--output", required=True, help="Path to HTML output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    with open(args.examples) as f:
        examples = json.load(f)

    html_doc = HTML_TEMPLATE.replace("__DATA__", json.dumps(examples))
    Path(args.output).write_text(html_doc)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
