"""
Vector database for RAG capabilities in Osyllabi.

This module provides a lightweight SQLite-based vector database for storing
and retrieving embeddings for Retrieval-Augmented Generation.
"""
import json
import pandas as pd
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from osyllabi.utils.log import log
from osyllabi.utils.vector.operations import cosine_similarity

# Default table names
DEFAULT_CHUNKS_TABLE = "chunks"
DEFAULT_VECTORS_TABLE = "vectors"


class VectorDatabase:
    """
    In-memory vector database using pandas DataFrames.
    Stores documents, chunks, and embeddings for quick access.
    """

    def __init__(self, db_path=None):
        """
        Initialize database.
        
        Args:
            db_path: Optional path to database file (not used in in-memory implementation)
        """
        # Each row will contain: doc_id, chunk_id, text, source, metadata, embedding (np.array)
        self.data = pd.DataFrame(columns=[
            "doc_id", "chunk_id", "text", "source", "metadata", "embedding"
        ])
        self._doc_counter = 0
        self._chunk_counter = 0
        # Store path for potential future persistence
        self.db_path = db_path

    def add_document(
        self,
        text_chunks: List[str],
        vectors: List[List[float]],
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        dedup: bool = True
    ) -> List[int]:
        if len(text_chunks) != len(vectors):
            raise ValueError("Number of chunks and vectors must match")

        doc_id = self._doc_counter
        self._doc_counter += 1
        chunk_ids = []

        for i, (chunk, vec) in enumerate(zip(text_chunks, vectors)):
            if not chunk.strip():
                continue
            chunk_id = self._chunk_counter
            self._chunk_counter += 1

            # Check dedup
            if dedup:
                existing = self.data[
                    (self.data["text"] == chunk) & (self.data["source"] == source)
                ]
                if len(existing) > 0:
                    chunk_ids.append(int(existing.iloc[0]["chunk_id"]))
                    continue

            row = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": chunk,
                "source": source if source else "",
                "metadata": json.dumps(metadata) if metadata else None,
                "embedding": np.array(vec, dtype=np.float32)
            }
            self.data = pd.concat([self.data, pd.DataFrame([row])], ignore_index=True)
            chunk_ids.append(chunk_id)
        return chunk_ids

    def remove_document(self, source: str) -> None:
        doc_rows = self.data[self.data["source"] == source]
        if len(doc_rows) > 0:
            doc_ids = doc_rows["doc_id"].unique()
            self.data = self.data[~self.data["doc_id"].isin(doc_ids)]

    def get_document_sources(self) -> List[str]:
        return list(self.data["source"].dropna().unique())

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        threshold: float = 0.0,
        source_filter: Optional[Union[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        if isinstance(source_filter, str):
            source_filter = [source_filter]
        df = self.data.copy()
        if source_filter:
            df = df[df["source"].isin(source_filter)]
        if df.empty:
            return []

        qvec = np.array(query_vector, dtype=np.float32)
        embeddings = df["embedding"].to_list()

        # Cosine similarity
        sims = []
        for emb in embeddings:
            dot = np.dot(qvec, emb)
            norm_prod = np.linalg.norm(qvec) * np.linalg.norm(emb)
            similarity = float(dot / (norm_prod + 1e-8))
            sims.append(similarity)

        df["similarity"] = sims
        df = df[df["similarity"] >= threshold]
        df = df.sort_values("similarity", ascending=False).head(top_k)

        results = []
        for _, row in df.iterrows():
            results.append({
                "id": int(row["chunk_id"]),
                "text": row["text"],
                "source": row["source"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "similarity": float(row["similarity"]),
                "chunk_index": int(row["chunk_id"])
            })
        return results

    def get_stats(self) -> Dict[str, Any]:
        return {
            "document_count": int(self.data["doc_id"].nunique()),
            "chunk_count": int(len(self.data)),
            "embedding_dimension": (
                len(self.data["embedding"].iloc[0]) if not self.data.empty else 0
            )
        }

    def close(self) -> None:
        # No-op for in-memory
        pass
