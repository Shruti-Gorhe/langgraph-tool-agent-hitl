import re
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document


def _h1_title(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)", text, re.MULTILINE)
    return m.group(1).strip() if m else fallback


def _load_markdown(path: Path) -> Document:
    raw = TextLoader(str(path), encoding="utf-8").load()[0]
    raw.metadata["source"] = path.name
    raw.metadata["doc_type"] = "markdown"
    raw.metadata["doc_title"] = _h1_title(raw.page_content, path.stem)
    raw.metadata["page"] = -1
    return raw


def _load_pdf(path: Path) -> Document:
    pages = PyPDFLoader(str(path)).load()
    combined = "\n".join(p.page_content for p in pages)
    boundaries = []
    offset = 0
    for i, p in enumerate(pages):
        boundaries.append({"page": i, "offset": offset})
        offset += len(p.page_content) + 1
    doc = Document(
        page_content=combined,
        metadata={
            "source": path.name,
            "doc_type": "pdf",
            "doc_title": _h1_title(combined, path.stem),
            "page": 0,
            "page_boundaries": str(boundaries),
        },
    )
    return doc


def load_corpus(directory: Path) -> list[Document]:
    docs = []
    for path in sorted(directory.iterdir()):
        if path.suffix == ".md":
            docs.append(_load_markdown(path))
        elif path.suffix == ".pdf":
            docs.append(_load_pdf(path))
    return docs
