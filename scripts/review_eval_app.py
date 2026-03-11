"""Interactive Streamlit app for reviewing squeez evaluation examples.

Usage:
    streamlit run scripts/review_eval_app.py -- --examples eval_examples.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import streamlit as st


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review squeez eval examples in Streamlit")
    parser.add_argument("--examples", default=None, help="Path to eval examples JSON")
    return parser


def _load_examples(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Examples file must contain a JSON list")
    return data


def _metric(example: dict[str, Any], key: str, default: float = 0.0) -> float:
    metrics = example.get("metrics") or {}
    value = metrics.get(key, default)
    return float(value)


def _error_text(example: dict[str, Any]) -> str:
    return str(example.get("error") or "")


def _empty_category(example: dict[str, Any]) -> str:
    metrics = example.get("metrics") or {}
    return str(metrics.get("empty_category") or "")


def _matches_filter(example: dict[str, Any], mode: str) -> bool:
    if mode == "all":
        return True
    if mode == "errors":
        return bool(_error_text(example))
    if mode == "bad_exact":
        return _metric(example, "exact_match") == 0.0
    if mode == "bad_fuzzy":
        return _metric(example, "fuzzy_span_f1") < 0.5
    if mode == "false_negative":
        return _empty_category(example) == "false_negative"
    if mode == "false_positive":
        return _empty_category(example) == "false_positive"
    if mode == "empty_gold":
        return len(example.get("reference_lines") or []) == 0
    if mode == "nonempty_gold":
        return len(example.get("reference_lines") or []) > 0
    return True


def _search_blob(example: dict[str, Any]) -> str:
    parts = [
        str(example.get("task") or ""),
        str(example.get("tool_output") or ""),
        str(example.get("prompt") or ""),
        "\n".join(example.get("predicted_lines") or []),
        "\n".join(example.get("reference_lines") or []),
        _error_text(example),
    ]
    return "\n".join(parts).lower()


def _sort_key(example: dict[str, Any], sort_mode: str) -> Any:
    if sort_mode == "lowest_span_f1":
        return (_metric(example, "span_f1"),)
    if sort_mode == "highest_span_f1":
        return (-_metric(example, "span_f1"),)
    if sort_mode == "lowest_fuzzy_f1":
        return (_metric(example, "fuzzy_span_f1"),)
    if sort_mode == "highest_fuzzy_f1":
        return (-_metric(example, "fuzzy_span_f1"),)
    if sort_mode == "highest_rouge":
        return (-_metric(example, "rouge_l"),)
    if sort_mode == "highest_compression":
        return (-_metric(example, "compression"),)
    return (int(example.get("_index", 0)),)


def _line_block(lines: list[str], title: str, height: int = 220) -> None:
    st.markdown(f"**{title}**")
    if not lines:
        st.caption("No lines")
        return
    st.code("\n".join(lines), language=None, height=height)


def main() -> int:
    parser = build_parser()
    args, _unknown = parser.parse_known_args()

    st.set_page_config(
        page_title="Squeez Eval Review",
        page_icon=":mag:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Squeez Eval Review")
    st.caption("Inspect prompt, tool output, predictions, gold labels, and scoring side by side.")

    examples_path = args.examples
    with st.sidebar:
        st.header("Data")
        examples_path = st.text_input("Examples JSON", value=examples_path or "")
        uploaded = st.file_uploader("Or upload examples JSON", type=["json"])

    try:
        if uploaded is not None:
            examples = json.load(uploaded)
        else:
            examples = _load_examples(examples_path or None)
    except Exception as exc:
        st.error(f"Failed to load examples: {exc}")
        return 1

    if not examples:
        st.info("Provide an eval examples JSON file via --examples or the sidebar.")
        return 0

    for index, example in enumerate(examples):
        example["_index"] = index

    with st.sidebar:
        st.header("Filters")
        query = st.text_input("Search", value="", placeholder="task / prompt / prediction / gold")
        filter_mode = st.selectbox(
            "Filter",
            [
                "all",
                "errors",
                "bad_exact",
                "bad_fuzzy",
                "false_negative",
                "false_positive",
                "empty_gold",
                "nonempty_gold",
            ],
            index=0,
        )
        sort_mode = st.selectbox(
            "Sort",
            [
                "original",
                "lowest_span_f1",
                "highest_span_f1",
                "lowest_fuzzy_f1",
                "highest_fuzzy_f1",
                "highest_rouge",
                "highest_compression",
            ],
            index=0,
        )
        show_only_errors = st.checkbox("Only samples with request/runtime errors", value=False)

    filtered = []
    lowered = query.strip().lower()
    for example in examples:
        if show_only_errors and not _error_text(example):
            continue
        if lowered and lowered not in _search_blob(example):
            continue
        if not _matches_filter(example, filter_mode):
            continue
        filtered.append(example)

    filtered.sort(key=lambda item: _sort_key(item, sort_mode))

    if not filtered:
        st.warning("No samples match the current filters.")
        return 0

    left, right = st.columns([1.1, 2.4], gap="large")

    with left:
        st.subheader("Samples")
        st.caption(f"{len(filtered)} visible / {len(examples)} total")
        labels = []
        for item in filtered:
            if _error_text(item):
                labels.append(
                    f"[ERR] #{item['_index']} {str(item.get('task') or '').splitlines()[0][:70]}"
                )
            else:
                labels.append(
                    f"#{item['_index']} F1={_metric(item, 'span_f1'):.3f} "
                    f"Fuzzy={_metric(item, 'fuzzy_span_f1'):.3f} "
                    f"{str(item.get('task') or '').splitlines()[0][:55]}"
                )
        selected_label = st.selectbox("Choose sample", labels, index=0)
        selected = filtered[labels.index(selected_label)]

        st.markdown("**Quick metrics**")
        if _error_text(selected):
            st.error(_error_text(selected))
        else:
            st.write(
                {
                    "span_f1": _metric(selected, "span_f1"),
                    "fuzzy_span_f1": _metric(selected, "fuzzy_span_f1"),
                    "exact_match": _metric(selected, "exact_match"),
                    "rouge_l": _metric(selected, "rouge_l"),
                    "compression": _metric(selected, "compression"),
                    "empty_category": _empty_category(selected),
                }
            )

    with right:
        st.subheader(f"Sample #{selected['_index']}")
        metric_cols = st.columns(6)
        metric_cols[0].metric("Span F1", f"{_metric(selected, 'span_f1'):.3f}")
        metric_cols[1].metric("Fuzzy F1", f"{_metric(selected, 'fuzzy_span_f1'):.3f}")
        metric_cols[2].metric("Exact", f"{_metric(selected, 'exact_match'):.0f}")
        metric_cols[3].metric("ROUGE-L", f"{_metric(selected, 'rouge_l'):.3f}")
        metric_cols[4].metric("Compression", f"{_metric(selected, 'compression'):.3f}")
        metric_cols[5].metric("Empty Cat.", _empty_category(selected) or "-")

        if _error_text(selected):
            st.error(_error_text(selected))

        st.markdown("### Task")
        st.text(selected.get("task") or "")

        prompt_col, parse_col = st.columns(2, gap="large")
        with prompt_col:
            st.markdown("### Raw Prompt")
            st.code(selected.get("prompt") or "", language=None, height=280)
        with parse_col:
            st.markdown("### Parsed Output")
            st.code(selected.get("tool_output") or "", language=None, height=280)

        pred_col, gold_col = st.columns(2, gap="large")
        with pred_col:
            _line_block(selected.get("predicted_lines") or [], "Predicted Lines", height=260)
        with gold_col:
            _line_block(selected.get("reference_lines") or [], "Reference Lines", height=260)

        if selected.get("metrics"):
            st.markdown("### Full Metrics")
            st.json(selected["metrics"], expanded=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
