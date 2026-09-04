from pathlib import Path

KB_DIR = Path("knowledge_base")
DATA_DIR = Path("data")
STORAGE_DIR = Path("storage")
RESULTS_DIR = Path("results")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

FIXED_CHUNK_SIZE = 500
FIXED_CHUNK_OVERLAP = 80

CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 50

BREAKPOINT_PERCENTILE = 95

PARENT_HEADERS = [("##", "section"), ("###", "subsection")]

TOP_K = 5
TOP_K_VALUES = (1, 3, 5)

ENSEMBLE_WEIGHTS = (0.5, 0.5)
RRF_C = 60

STRATEGIES = ("fixed", "semantic", "hierarchical")
METHODS = ("semantic", "hybrid")
