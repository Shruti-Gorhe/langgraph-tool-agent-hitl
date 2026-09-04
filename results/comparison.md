# RAG Retrieval Benchmark — Comparison Report

Generated: 2026-09-04T15:52:47

## Configuration

```
EMBED_MODEL           = sentence-transformers/all-MiniLM-L6-v2
FIXED_CHUNK_SIZE      = 500
FIXED_CHUNK_OVERLAP   = 80
CHILD_CHUNK_SIZE      = 300
CHILD_CHUNK_OVERLAP   = 50
BREAKPOINT_PERCENTILE = 95
TOP_K                 = 5
ENSEMBLE_WEIGHTS      = (0.5, 0.5)
RRF_C                 = 60
```

## Corpus & Ingestion Stats

| Strategy      | Chunks | Avg chars | Sections | Ingest s |
|:--------------|-------:|----------:|---------:|---------:|
| fixed         |     85 |       371 |       84 |     1.29 |
| semantic      |     30 |       988 |       26 |     0.27 |
| hierarchical  |    167 |       183 |       87 |     0.31 |

## Headline Metrics

| Strategy      | Method   | HR@1  | HR@3  | HR@5  | MRR@1 | MRR@3 | MRR@5 | Avg ms | Avg chars |
|:--------------|:---------|------:|------:|------:|------:|------:|------:|-------:|----------:|
| fixed         | semantic | 80.0% | 85.0% | 85.0% | 80.0% | 82.5% | 82.5% |    3.1 |       387 |
| fixed         | hybrid   | 70.0% | 85.0% | 85.0% | 70.0% | 76.7% | 76.7% |    3.6 |       399 |
| semantic      | semantic |  0.0% | 75.0% | 90.0% |  0.0% | 25.0% | 28.2% |    1.9 |       834 |
| semantic      | hybrid   |  0.0% | 90.0% | 90.0% |  0.0% | 42.5% | 42.5% |    2.6 |      1174 |
| hierarchical  | semantic | 55.0% | 80.0% | 90.0% | 55.0% | 66.7% | 69.2% |    1.9 |       217 |
| hierarchical  | hybrid   | 70.0% | 80.0% | 85.0% | 70.0% | 75.0% | 76.2% |    2.6 |       223 |

## Query Style Breakdown

| Strategy      | Method   | Style     | HR@5  | MRR@5 | n |
|:--------------|:---------|:----------|------:|------:|--:|
| fixed         | semantic | keyword   | 90.0% | 85.0% | 10 |
| fixed         | semantic | paraphrase | 80.0% | 80.0% | 10 |
| fixed         | hybrid   | keyword   | 90.0% | 85.0% | 10 |
| fixed         | hybrid   | paraphrase | 80.0% | 68.3% | 10 |
| semantic      | semantic | keyword   | 100.0% | 29.8% | 10 |
| semantic      | semantic | paraphrase | 80.0% | 26.7% | 10 |
| semantic      | hybrid   | keyword   | 100.0% | 46.7% | 10 |
| semantic      | hybrid   | paraphrase | 80.0% | 38.3% | 10 |
| hierarchical  | semantic | keyword   | 90.0% | 70.8% | 10 |
| hierarchical  | semantic | paraphrase | 90.0% | 67.5% | 10 |
| hierarchical  | hybrid   | keyword   | 90.0% | 80.0% | 10 |
| hierarchical  | hybrid   | paraphrase | 80.0% | 72.5% | 10 |

## Findings

**Best HR@5:** semantic / semantic (90.0%).
**Best MRR@5:** fixed / semantic (82.5%).

**Hybrid vs semantic delta (MRR@5):**
- fixed: -5.8% (semantic 82.5% → hybrid 76.7%)
- semantic: +14.3% (semantic 28.2% → hybrid 42.5%)
- hierarchical: +7.1% (semantic 69.2% → hybrid 76.2%)

**Keyword vs paraphrase gap (MRR@5 across all strategies):**
- semantic: keyword 61.9% vs paraphrase 58.1%
- hybrid: keyword 70.6% vs paraphrase 59.7%

**Chunk size context:**
- fixed avg chunk chars: 387
- fixed avg chunk chars: 399
- semantic avg chunk chars: 834
- semantic avg chunk chars: 1174
- hierarchical avg chunk chars: 217
- hierarchical avg chunk chars: 223

## Limitations

- **10 documents (~75–150 chunks per strategy).** Absolute Hit Rates read high; relative ordering between strategies and methods is the valid comparison.
- **No PDF in the measured corpus.** The loader supports PDF with page metadata; all reported numbers come from Markdown.
- **Chunk size varies with strategy** (500/80 fixed vs 300/50 hierarchical vs percentile-based semantic). This is inherent to the strategies as designed and is disclosed via avg_chunk_chars above.
- **Retrieval-only.** No generation step or LLM judge — the study measures retrieval quality, not answer quality, which is exactly reproducible.
