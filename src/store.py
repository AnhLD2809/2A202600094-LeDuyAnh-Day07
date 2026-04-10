from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            # TODO: initialize chromadb client + collection
            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=self._collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # TODO: build a normalized stored record for one document
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata,
            "embedding": self._embedding_fn(doc.content)
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # TODO: run in-memory similarity search over provided records
        from .chunking import compute_similarity
        query_emb = self._embedding_fn(query)
        scored = []
        for record in records:
            score = compute_similarity(query_emb, record["embedding"])
            rec = record.copy()
            rec["score"] = score
            scored.append(rec)
            
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        # TODO: embed each doc and add to store
        if self._use_chroma:
            ids = [doc.id for doc in docs]
            documents = [doc.content for doc in docs]
            metadatas = [doc.metadata for doc in docs]
            embeddings = [self._embedding_fn(doc.content) for doc in docs]
            if ids:
                self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        else:
            for doc in docs:
                self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        # TODO: embed query, compute similarities, return top_k
        if self._use_chroma:
            query_emb = self._embedding_fn(query)
            results = self._collection.query(query_embeddings=[query_emb], n_results=top_k)
            ret = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    ret.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1.0 - results["distances"][0][i] if results.get("distances") else 0.0
                    })
            return ret
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        # TODO
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # TODO: filter by metadata, then search among filtered chunks
        metadata_filter = metadata_filter or {}
        if self._use_chroma:
            query_emb = self._embedding_fn(query)
            kwargs = {"query_embeddings": [query_emb], "n_results": top_k}
            if metadata_filter:
                kwargs["where"] = metadata_filter
            results = self._collection.query(**kwargs)
            ret = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    ret.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1.0 - results["distances"][0][i] if results.get("distances") else 0.0
                    })
            return ret
        else:
            filtered_store = self._store
            if metadata_filter:
                filtered_store = []
                for rec in self._store:
                    match = True
                    for k, v in metadata_filter.items():
                        if rec["metadata"].get(k) != v:
                            match = False
                            break
                    if match:
                        filtered_store.append(rec)
            return self._search_records(query, filtered_store, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        # TODO: remove all stored chunks where metadata['doc_id'] == doc_id
        if self._use_chroma:
            before_count = self._collection.count()
            self._collection.delete(ids=[doc_id])
            # some docs could be chunked with ID ending in _1 etc, but if doc_id is exactly the id
            # actually we check id first, but might also do where={"doc_id": doc_id} if there is metadata
            try:
                self._collection.delete(where={"doc_id": doc_id})
            except Exception:
                pass 
            return self._collection.count() < before_count
        else:
            before_len = len(self._store)
            self._store = [rec for rec in self._store if rec["id"] != doc_id and rec["metadata"].get("doc_id") != doc_id]
            return len(self._store) < before_len
