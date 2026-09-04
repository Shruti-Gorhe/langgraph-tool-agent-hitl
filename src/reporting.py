import csv
from datetime import datetime
from pathlib import Path
from config import (
    RESULTS_DIR,
    FIXED_CHUNK_SIZE, FIXED_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP,
    BREAKPOINT_PERCENTILE, TOP_K, ENSEMBLE_WEIGHTS, RRF_C,
    EMBED_MODEL,
)


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def comparison_table(agg_rows: list[dict]) -> str:
    header = (
        "| Strategy      | Method   | HR@1  | HR@3  | HR@5  | MRR@1 | MRR@3 | MRR@5 |"
        " Avg ms | Avg chars |"
    )
    sep = (
        "|:--------------|:---------|------:|------:|------:|------:|------:|------:|"
        "-------:|----------:|"
    )
    lines = [header, sep]
    for r in agg_rows:
        lines.append(
            f"| {r['strategy']:<13} | {r['method']:<8} |"
            f" {_pct(r['hit_rate@1']):>5} | {_pct(r['hit_rate@3']):>5} | {_pct(r['hit_rate@5']):>5} |"
            f" {_pct(r['mrr@1']):>5} | {_pct(r['mrr@3']):>5} | {_pct(r['mrr@5']):>5} |"
            f" {r['avg_query_ms']:>6} | {r['avg_retrieved_chars']:>9} |"
        )
    return "\n".join(lines)


def style_table(per_question_rows: list[dict], agg_rows: list[dict]) -> str:
    from collections import defaultdict
    buckets: dict[tuple, list] = defaultdict(list)
    for r in per_question_rows:
        key = (r["strategy"], r["method"], r["query_style"])
        buckets[key].append(r)

    header = (
        "| Strategy      | Method   | Style     | HR@5  | MRR@5 | n |"
    )
    sep = "|:--------------|:---------|:----------|------:|------:|--:|"
    lines = [header, sep]
    for r in agg_rows:
        for style in ("keyword", "paraphrase"):
            group = buckets[(r["strategy"], r["method"], style)]
            if not group:
                continue
            n = len(group)
            hr5 = sum(x["hit@5"] for x in group) / n
            mrr5 = sum(x["rr@5"] for x in group) / n
            lines.append(
                f"| {r['strategy']:<13} | {r['method']:<8} | {style:<9} |"
                f" {_pct(hr5):>5} | {_pct(mrr5):>5} | {n:>1} |"
            )
    return "\n".join(lines)


def findings(agg_rows: list[dict], per_question_rows: list[dict]) -> str:
    by_key = {(r["strategy"], r["method"]): r for r in agg_rows}
    lines = []

    best_hr5 = max(agg_rows, key=lambda r: r["hit_rate@5"])
    best_mrr5 = max(agg_rows, key=lambda r: r["mrr@5"])
    lines.append(
        f"**Best HR@5:** {best_hr5['strategy']} / {best_hr5['method']} "
        f"({_pct(best_hr5['hit_rate@5'])})."
    )
    lines.append(
        f"**Best MRR@5:** {best_mrr5['strategy']} / {best_mrr5['method']} "
        f"({_pct(best_mrr5['mrr@5'])})."
    )

    lines.append("")
    lines.append("**Hybrid vs semantic delta (MRR@5):**")
    for strategy in ("fixed", "semantic", "hierarchical"):
        sem = by_key.get((strategy, "semantic"))
        hyb = by_key.get((strategy, "hybrid"))
        if sem and hyb:
            delta = hyb["mrr@5"] - sem["mrr@5"]
            direction = "+" if delta >= 0 else ""
            lines.append(
                f"- {strategy}: {direction}{_pct(delta)} "
                f"(semantic {_pct(sem['mrr@5'])} → hybrid {_pct(hyb['mrr@5'])})"
            )

    from collections import defaultdict
    style_buckets: dict[tuple, list] = defaultdict(list)
    for r in per_question_rows:
        style_buckets[(r["strategy"], r["method"], r["query_style"])].append(r)

    lines.append("")
    lines.append("**Keyword vs paraphrase gap (MRR@5 across all strategies):**")
    for method in ("semantic", "hybrid"):
        kw_scores = []
        pp_scores = []
        for strategy in ("fixed", "semantic", "hierarchical"):
            kw = style_buckets[(strategy, method, "keyword")]
            pp = style_buckets[(strategy, method, "paraphrase")]
            if kw:
                kw_scores.extend(r["rr@5"] for r in kw)
            if pp:
                pp_scores.extend(r["rr@5"] for r in pp)
        kw_avg = sum(kw_scores) / len(kw_scores) if kw_scores else 0
        pp_avg = sum(pp_scores) / len(pp_scores) if pp_scores else 0
        lines.append(
            f"- {method}: keyword {_pct(kw_avg)} vs paraphrase {_pct(pp_avg)}"
        )

    lines.append("")
    lines.append("**Chunk size context:**")
    for r in agg_rows:
        lines.append(
            f"- {r['strategy']} avg chunk chars: {r['avg_retrieved_chars']}"
        )

    return "\n".join(lines)


def write_artifacts(
    agg_rows: list[dict],
    per_question_rows: list[dict],
    ingest_stats: dict,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = RESULTS_DIR / "retrieval_metrics.csv"
    if agg_rows:
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            writer.writeheader()
            writer.writerows(agg_rows)

    pq_path = RESULTS_DIR / "per_question.csv"
    if per_question_rows:
        with open(pq_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_question_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_question_rows)

    report_path = RESULTS_DIR / "comparison.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# RAG Retrieval Benchmark — Comparison Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")

        f.write("## Configuration\n\n")
        f.write(f"```\n")
        f.write(f"EMBED_MODEL           = {EMBED_MODEL}\n")
        f.write(f"FIXED_CHUNK_SIZE      = {FIXED_CHUNK_SIZE}\n")
        f.write(f"FIXED_CHUNK_OVERLAP   = {FIXED_CHUNK_OVERLAP}\n")
        f.write(f"CHILD_CHUNK_SIZE      = {CHILD_CHUNK_SIZE}\n")
        f.write(f"CHILD_CHUNK_OVERLAP   = {CHILD_CHUNK_OVERLAP}\n")
        f.write(f"BREAKPOINT_PERCENTILE = {BREAKPOINT_PERCENTILE}\n")
        f.write(f"TOP_K                 = {TOP_K}\n")
        f.write(f"ENSEMBLE_WEIGHTS      = {ENSEMBLE_WEIGHTS}\n")
        f.write(f"RRF_C                 = {RRF_C}\n")
        f.write(f"```\n\n")

        if ingest_stats:
            f.write("## Corpus & Ingestion Stats\n\n")
            f.write("| Strategy      | Chunks | Avg chars | Sections | Ingest s |\n")
            f.write("|:--------------|-------:|----------:|---------:|---------:|\n")
            for strat, s in ingest_stats.items():
                f.write(
                    f"| {strat:<13} | {s['n_chunks']:>6} | {s['avg_chunk_chars']:>9} |"
                    f" {s['n_sections']:>8} | {s['ingest_secs']:>8} |\n"
                )
            f.write("\n")

        f.write("## Headline Metrics\n\n")
        f.write(comparison_table(agg_rows))
        f.write("\n\n")

        f.write("## Query Style Breakdown\n\n")
        f.write(style_table(per_question_rows, agg_rows))
        f.write("\n\n")

        f.write("## Findings\n\n")
        f.write(findings(agg_rows, per_question_rows))
        f.write("\n\n")

        f.write("## Limitations\n\n")
        f.write(
            "- **10 documents (~75–150 chunks per strategy).** Absolute Hit Rates read high; "
            "relative ordering between strategies and methods is the valid comparison.\n"
            "- **No PDF in the measured corpus.** The loader supports PDF with page metadata; "
            "all reported numbers come from Markdown.\n"
            "- **Chunk size varies with strategy** (500/80 fixed vs 300/50 hierarchical vs "
            "percentile-based semantic). This is inherent to the strategies as designed and "
            "is disclosed via avg_chunk_chars above.\n"
            "- **Retrieval-only.** No generation step or LLM judge — the study measures "
            "retrieval quality, not answer quality, which is exactly reproducible.\n"
        )

    return report_path
