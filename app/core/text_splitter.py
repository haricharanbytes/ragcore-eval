import logging

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

logger = logging.getLogger(__name__)


def split_documents(documents: list[LCDocument]) -> list[LCDocument]:
    """Split a list of loaded Documents into chunks"""
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Add a running chunk_index per source document, since split_documents
    counts_by_source: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        idx = counts_by_source.get(source, 0)
        chunk.metadata["chunk_index"] = idx
        counts_by_source[source] = idx + 1

    logger.info(
        "Split %d document section(s) into %d chunk(s) (size=%d, overlap=%d)",
        len(documents), len(chunks), settings.chunk_size, settings.chunk_overlap,
    )
    return chunks