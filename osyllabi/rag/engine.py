"""
Retrieval-Augmented Generation engine for curriculum generation.

This module provides the main RAG engine that coordinates vector storage,
retrieval, and context assembly for curriculum generation.
"""
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from osyllabi.utils.log import log
from osyllabi.utils.decorators.singleton import singleton
from osyllabi.rag.database import VectorDatabase
from osyllabi.rag.embedding import EmbeddingGenerator
from osyllabi.rag.chunking import TextChunker


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for curriculum content.
    
    This class integrates document processing, embedding generation,
    vector storage, and retrieval to support RAG for curriculum generation.
    """
    
    def __init__(
        self,
        run_id: Optional[str] = None,
        base_dir: Optional[Union[str, Path]] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Initialize the RAG engine.
        
        Args:
            run_id: Unique identifier for this generation run
            base_dir: Base directory for vector storage
            embedding_model: Name of the embedding model to use
            chunk_size: Size of text chunks in tokens
            chunk_overlap: Overlap between consecutive chunks in tokens
        """
        # Generate run ID if not provided
        self.run_id = run_id or str(int(time.time()))
        self.base_dir = Path(base_dir) if base_dir else Path.cwd() / "output" / self.run_id
        
        # Create vector database directory
        self.vector_dir = self.base_dir / "vectors"
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        
        # Create vector database and components
        db_path = self.vector_dir / "vector.db"
        self.vector_db = VectorDatabase(db_path)
        self.embedder = EmbeddingGenerator(model_name=embedding_model)
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        
        # Store configuration
        self.config = {
            "run_id": self.run_id,
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "created_at": time.time()
        }
        self._save_config()
        
        log.info(f"Initialized RAG engine for run {self.run_id} with model {embedding_model}")
        
    def _save_config(self) -> None:
        """Save RAG configuration to disk."""
        config_path = self.vector_dir / "metadata.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def add_document(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Process and add a document to the vector store.
        
        Args:
            text: Document text content
            metadata: Additional document metadata
            
        Returns:
            int: Number of chunks added to the database
        """
        if not text or not text.strip():
            log.debug("Skipping empty document")
            return 0
            
        # Prepare metadata
        doc_metadata = metadata or {}
        
        # Clean and chunk the text
        chunks = self.chunker.chunk_text(text)
        if not chunks:
            log.debug("No chunks generated from document")
            return 0
            
        # Generate embeddings
        embeddings = self.embedder.embed_texts(chunks)
        
        # Store in database
        chunk_ids = self.vector_db.add_document(chunks, embeddings, doc_metadata)
        
        log.debug(f"Added document with {len(chunks)} chunks to vector database")
        return len(chunks)
        
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: The search query
            top_k: Maximum number of results to return
            threshold: Minimum similarity score to include (0-1)
            
        Returns:
            List of dictionaries with retrieved chunks and metadata
        """
        log.debug(f"Retrieving context for query: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Search for similar chunks
        results = self.vector_db.search(
            query_embedding, 
            top_k=top_k,
            threshold=threshold
        )
        
        log.debug(f"Retrieved {len(results)} relevant chunks")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG engine and database.
        
        Returns:
            Dict with statistics and metadata
        """
        return {
            "run_id": self.run_id,
            "document_count": self.vector_db.get_document_count(),
            "chunk_count": self.vector_db.get_chunk_count(),
            "embedding_model": self.config["embedding_model"],
            "chunk_size": self.config["chunk_size"],
            "created_at": self.config["created_at"]
        }
    
    @classmethod
    def load(cls, run_id: str, base_dir: Optional[Union[str, Path]] = None) -> 'RAGEngine':
        """
        Load an existing RAG engine by run ID.
        
        Args:
            run_id: ID of the run to load
            base_dir: Base directory where run data is stored
            
        Returns:
            RAGEngine: Initialized RAG engine
            
        Raises:
            FileNotFoundError: If the run doesn't exist
        """
        base_path = Path(base_dir) if base_dir else Path.cwd() / "output"
        run_dir = base_path / run_id
        
        if not (run_dir / "vectors" / "vector.db").exists():
            raise FileNotFoundError(f"No RAG data found for run {run_id}")
            
        # Load configuration
        config_path = run_dir / "vectors" / "metadata.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            return cls(
                run_id=run_id,
                base_dir=base_dir,
                embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
                chunk_size=config.get("chunk_size", 512),
                chunk_overlap=config.get("chunk_overlap", 50)
            )
        else:
            # Use defaults if config not found
            return cls(run_id=run_id, base_dir=base_dir)
