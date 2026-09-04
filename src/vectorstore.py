from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from config import STORAGE_DIR


def _collection_name(strategy: str) -> str:
    return f"kb_{strategy}"


def _persist_dir() -> str:
    path = STORAGE_DIR / "chroma"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_store(strategy: str, embeddings: HuggingFaceEmbeddings) -> Chroma:
    return Chroma(
        collection_name=_collection_name(strategy),
        embedding_function=embeddings,
        persist_directory=_persist_dir(),
    )


def reset_collection(strategy: str, embeddings: HuggingFaceEmbeddings) -> Chroma:
    store = get_store(strategy, embeddings)
    existing = store.get()
    if existing["ids"]:
        store.delete(ids=existing["ids"])
    return store


def read_all_chunks(store: Chroma) -> list[Document]:
    result = store.get(include=["documents", "metadatas"])
    docs = []
    for text, meta in zip(result["documents"], result["metadatas"]):
        docs.append(Document(page_content=text, metadata=meta or {}))
    return docs
