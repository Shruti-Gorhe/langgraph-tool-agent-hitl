import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_TOKEN"] = ""

import warnings
warnings.filterwarnings("ignore")

import argparse
from config import STRATEGIES, METHODS

parser = argparse.ArgumentParser(description="RAG chunking benchmark")
parser.add_argument(
    "--skip-ingest",
    action="store_true",
    help="Skip ingestion and use the existing Chroma index.",
)
parser.add_argument(
    "--strategies",
    nargs="+",
    choices=list(STRATEGIES),
    default=list(STRATEGIES),
    help="Strategies to ingest / benchmark (default: all three).",
)
parser.add_argument(
    "--methods",
    nargs="+",
    choices=list(METHODS),
    default=list(METHODS),
    help="Retrieval methods to benchmark (default: both).",
)
args = parser.parse_args()

from src.embeddings import get_embeddings
from src.pipeline import ingest, set_embeddings_ref
from src.retrievers import build_retriever
from src.evaluation import load_questions, evaluate
from src.reporting import write_artifacts

print("Loading embedding model...")
emb = get_embeddings()
set_embeddings_ref(emb)
print("  sentence-transformers/all-MiniLM-L6-v2  384-dim  offline\n")

if not args.skip_ingest:
    print(f"Ingesting strategies: {args.strategies}")
    stats = ingest(args.strategies)
    print()
    print(f"  {'strategy':<15} {'chunks':>6}  {'avg chars':>9}  {'sections':>8}  {'secs':>6}")
    print(f"  {'-'*15} {'-'*6}  {'-'*9}  {'-'*8}  {'-'*6}")
    for s, v in stats.items():
        print(f"  {s:<15} {v['n_chunks']:>6}  {v['avg_chunk_chars']:>9}  {v['n_sections']:>8}  {v['ingest_secs']:>6}")
    print()
else:
    print("Skipping ingestion — using existing index.\n")
    stats = {}

questions = load_questions()
combos = [(s, m) for s in args.strategies for m in args.methods]
print(f"Running benchmark: {len(combos)} combos × {len(questions)} questions = {len(combos)*len(questions)} retrievals\n")

all_pq = []
all_agg = []

for strategy, method in combos:
    retriever = build_retriever(strategy, method, emb)
    rows, agg = evaluate(strategy, method, questions, retriever)
    all_pq.extend(rows)
    agg["strategy"] = strategy
    agg["method"] = method
    all_agg.append(agg)
    print(
        f"  [{strategy:<13} / {method:<8}]  "
        f"HR@1={agg['hit_rate@1']:.2f}  HR@3={agg['hit_rate@3']:.2f}  HR@5={agg['hit_rate@5']:.2f}  "
        f"MRR@1={agg['mrr@1']:.2f}  MRR@5={agg['mrr@5']:.2f}  "
        f"avg {agg['avg_query_ms']:.0f}ms"
    )

print()
report_path = write_artifacts(all_agg, all_pq, stats)
print(f"Results written to  results/retrieval_metrics.csv")
print(f"                    results/per_question.csv")
print(f"                    {report_path}")
