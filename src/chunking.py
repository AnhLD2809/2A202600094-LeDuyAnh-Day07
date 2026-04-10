from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        # TODO: split into sentences, group into chunks
        import re
        if not text.strip():
            return []
        
        # Split but keep separators by using capturing groups
        parts = re.split(r'(\. |\! |\? |\.\n)', text)
        sentences = []
        for i in range(0, len(parts), 2):
            part = parts[i]
            sep = parts[i+1] if i+1 < len(parts) else ""
            combined = (part + sep).strip()
            if combined:
                sentences.append(combined)

        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunks.append(" ".join(sentences[i : i + self.max_sentences_per_chunk]))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        # TODO: implement recursive splitting strategy
        text = text.strip()
        if not text:
            return []
        return self._split(text, self.separators.copy())

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # TODO: recursive helper used by RecursiveChunker.chunk
        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[i:i+self.chunk_size])
            return chunks

        sep = remaining_separators[0]
        if sep == "":
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[i:i+self.chunk_size])
            return chunks

        parts = current_text.split(sep)
        if len(parts) == 1:
            return self._split(current_text, remaining_separators[1:])

        final_chunks = []
        current_chunk = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            if len(part) > self.chunk_size:
                if current_chunk:
                    final_chunks.append(current_chunk)
                    current_chunk = ""
                final_chunks.extend(self._split(part, remaining_separators[1:]))
            else:
                temp = (current_chunk + sep + part).strip() if current_chunk else part
                if len(temp) <= self.chunk_size:
                    current_chunk = temp
                else:
                    if current_chunk:
                        final_chunks.append(current_chunk)
                    current_chunk = part

        if current_chunk:
            final_chunks.append(current_chunk)
            
        return final_chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    # TODO: implement cosine similarity formula
    dot_product = _dot(vec_a, vec_b)
    norm_a = _dot(vec_a, vec_a) ** 0.5
    norm_b = _dot(vec_b, vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # TODO: call each chunker, compute stats, return comparison dict
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=20)
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=3)
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)

        fixed_chunks = fixed_chunker.chunk(text)
        sentence_chunks = sentence_chunker.chunk(text)
        recursive_chunks = recursive_chunker.chunk(text)

        return {
            "fixed_size": {
                "count": len(fixed_chunks),
                "avg_length": sum(len(c) for c in fixed_chunks) / len(fixed_chunks) if fixed_chunks else 0,
                "chunks": fixed_chunks
            },
            "by_sentences": {
                "count": len(sentence_chunks),
                "avg_length": sum(len(c) for c in sentence_chunks) / len(sentence_chunks) if sentence_chunks else 0,
                "chunks": sentence_chunks
            },
            "recursive": {
                "count": len(recursive_chunks),
                "avg_length": sum(len(c) for c in recursive_chunks) / len(recursive_chunks) if recursive_chunks else 0,
                "chunks": recursive_chunks
            }
        }
