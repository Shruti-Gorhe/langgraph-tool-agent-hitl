import re
import warnings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from config import (
    FIXED_CHUNK_SIZE,
    FIXED_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    BREAKPOINT_PERCENTILE,
    PARENT_HEADERS,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_experimental.text_splitter import SemanticChunker


def _heading_offset_map(text: str) -> list[tuple[int, str]]:
    result = []
    for m in re.finditer(r"^#{1,3}\s+(.+)", text, re.MULTILINE):
        result.append((m.start(), m.group(1).strip()))
    return result


def _section_for_offset(offset_map: list[tuple[int, str]], pos: int) -> str:
    section = ""
    for off, heading in offset_map:
        if off <= pos:
            section = heading
        else:
            break
    return section


def _section_by_prefix(offset_map: list[tuple[int, str]], text: str, chunk_text: str) -> str:
    prefix = chunk_text.strip()[:40].lower()
    for off, heading in offset_map:
        norm = heading.lower()
        if norm in prefix or prefix in norm:
            return heading
    pos = text.find(chunk_text.strip()[:30])
    if pos >= 0:
        return _section_for_offset(offset_map, pos)
    return ""


def attach_metadata(
    chunks: list[Document],
    strategy: str,
    source: str,
    doc_title: str,
    doc_type: str,
    extra: dict | None = None,
) -> list[Document]:
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = source
        chunk.metadata["doc_title"] = doc_title
        chunk.metadata["doc_type"] = doc_type
        chunk.metadata["chunk_strategy"] = strategy
        chunk.metadata["chunk_id"] = f"{source}__{strategy}__{i:04d}"
        chunk.metadata.setdefault("section", "")
        chunk.metadata.setdefault("subsection", "")
        chunk.metadata.setdefault("page", -1)
        chunk.metadata.setdefault("parent_id", "")
        chunk.metadata.setdefault("parent_title", "")
        chunk.metadata["char_len"] = len(chunk.page_content)
        if extra:
            for k, v in extra.items():
                chunk.metadata.setdefault(k, v)
        for k in list(chunk.metadata.keys()):
            if chunk.metadata[k] is None:
                chunk.metadata[k] = ""
    return chunks


def chunk_fixed(doc: Document, embeddings: HuggingFaceEmbeddings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=FIXED_CHUNK_SIZE,
        chunk_overlap=FIXED_CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents([doc])
    offset_map = _heading_offset_map(doc.page_content)
    for chunk in chunks:
        start = chunk.metadata.get("start_index", 0)
        chunk.metadata["section"] = _section_for_offset(offset_map, start)
    return attach_metadata(
        chunks, "fixed", doc.metadata["source"], doc.metadata["doc_title"], doc.metadata["doc_type"]
    )


def chunk_semantic(doc: Document, embeddings: HuggingFaceEmbeddings) -> list[Document]:
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=BREAKPOINT_PERCENTILE,
    )
    chunks = splitter.create_documents([doc.page_content])
    offset_map = _heading_offset_map(doc.page_content)
    for chunk in chunks:
        chunk.metadata["section"] = _section_by_prefix(
            offset_map, doc.page_content, chunk.page_content
        )
    return attach_metadata(
        chunks, "semantic", doc.metadata["source"], doc.metadata["doc_title"], doc.metadata["doc_type"]
    )


def chunk_hierarchical(doc: Document, embeddings: HuggingFaceEmbeddings) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=PARENT_HEADERS,
        strip_headers=False,
    )
    parents = header_splitter.split_text(doc.page_content)

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )

    all_children = []
    source = doc.metadata["source"]
    doc_title = doc.metadata["doc_title"]
    doc_type = doc.metadata["doc_type"]

    for j, parent in enumerate(parents):
        parent_id = f"{source}__parent__{j:04d}"
        parent_title = parent.metadata.get("section", parent.metadata.get("subsection", ""))
        section = parent.metadata.get("section", "")
        subsection = parent.metadata.get("subsection", "")
        children = child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parent_id"] = parent_id
            child.metadata["parent_title"] = parent_title
            child.metadata["section"] = section
            child.metadata["subsection"] = subsection
        all_children.extend(children)

    return attach_metadata(all_children, "hierarchical", source, doc_title, doc_type)
