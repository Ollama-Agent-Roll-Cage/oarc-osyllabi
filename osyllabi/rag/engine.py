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
from osyllabi.utils.utils import check_for_ollama
from osyllabi.utils.decorators.singleton import singleton
from osyllabi.rag.database import VectorDatabase
from osyllabi.rag.embedding import EmbeddingGenerator
from osyllabi.rag.chunking import TextChunker
from osyllabi.rag.llama import LlamaDocumentLoader, setup_llama_index

@singleton
class RAGEngine:
    """
    Retrieval-Augmented Generation engine for curriculum content.
    
    This class integrates document processing, embedding generation,
    vector storage, and retrieval to support RAG for curriculum generation.
    """
    
    @classmethod 
    def _reset_singleton(cls):
        """Reset the singleton instance (for testing only)"""
        if hasattr(cls, "_instance"):
            delattr(cls, "_instance")
    
    def __init__(
        self,
        run_id: Optional[str] = None,
        base_dir: Optional[Union[str, Path]] = None,
        embedding_model: str = "llama3.1:latest",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        create_dirs: bool = True
    ):
        """
        Initialize the RAG engine.
        
        Args:
            run_id: Unique identifier for this generation run
            base_dir: Base directory for vector storage
            embedding_model: Name of the embedding model to use
            chunk_size: Size of text chunks in tokens
            chunk_overlap: Overlap between consecutive chunks in tokens
            create_dirs: Whether to create directories if they don't exist
            
        Raises:
            RuntimeError: If Ollama is not available
        """
        # Ensure Ollama is available before proceeding
        check_for_ollama(raise_error=True)
        
        # Initialize configuration first
        self.config = {
            "run_id": run_id or str(int(time.time())),
            "embedding_model": embedding_model,  # Use the provided model directly
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "created_at": time.time()
        }

        # Use config values consistently
        self.run_id = self.config["run_id"]
        self.base_dir = Path(base_dir) if base_dir else Path.cwd() / "output" / self.run_id
        self.vector_dir = self.base_dir / "vectors"

        # Create components using config values
        db_path = self.vector_dir / "vector.db"
        self.vector_db = VectorDatabase(db_path)
        self.embedder = EmbeddingGenerator(model_name=self.config["embedding_model"])
        self.chunker = TextChunker(
            chunk_size=self.config["chunk_size"],
            overlap=self.config["chunk_overlap"]
        )
        
        # Save config to disk if directories should be created
        if create_dirs:
            self.vector_dir.mkdir(parents=True, exist_ok=True)
            self._save_config()
        
        log.info(f"Initialized RAG engine for run {self.run_id} with model {self.config['embedding_model']}")
        
        # Initialize LlamaIndex components
        try:
            setup_llama_index()
            self.document_loader = LlamaDocumentLoader()
            log.info("LlamaIndex document loader enabled")
        except Exception as e:
            log.warning(f"LlamaIndex integration initialization failed: {e}")
            self.document_loader = None

        
    def _save_config(self) -> None:
        """Save RAG configuration to disk."""
        self.vector_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        config_path = self.vector_dir / "metadata.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
            

    def add_document(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> int:
        """
        Process and add a document to the vector store.
        
        Args:
            text: Document text content
            metadata: Additional document metadata
            source: Source identifier (file, URL, etc.)
            
        Returns:
            int: Number of chunks added to the database
            
        Raises:
            RuntimeError: If vectorization or database operation fails
        """
        if not text or not text.strip():
            log.debug("Skipping empty document")
            return 0
            
        # Prepare metadata
        doc_metadata = metadata or {}
        if source:
            doc_metadata['source'] = source
        
        # Clean and chunk the text
        chunks = self.chunker.chunk_text(text)
        if not chunks:
            log.debug("No chunks generated from document")
            return 0
            
        # Generate embeddings - will raise RuntimeError if Ollama is not available
        try:
            embeddings = self.embedder.embed_texts(chunks)
        except Exception as e:
            log.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError(f"Failed to generate embeddings for document: {e}")
        
        # Store in database
        try:
            chunk_ids = self.vector_db.add_document(chunks, embeddings, doc_metadata, source=source)
            
            # Update document count
            if hasattr(self, 'stats'):
                self.stats["documents_added"] = self.stats.get("documents_added", 0) + 1
                self.stats["chunks_added"] = self.stats.get("chunks_added", 0) + len(chunks)
            
            log.debug(f"Added document with {len(chunks)} chunks to vector database")
            return len(chunks)
        except Exception as e:
            log.error(f"Failed to add document to database: {e}")
            raise RuntimeError(f"Failed to add document to database: {e}")
        

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        threshold: float = 0.0,
        source_filter: Optional[Union[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: The search query
            top_k: Maximum number of results to return
            threshold: Minimum similarity score to include (0-1)
            source_filter: Optional filter for specific sources
            
        Returns:
            List of dictionaries with retrieved chunks and metadata
            
        Raises:
            RuntimeError: If embedding generation or search fails
        """
        log.debug(f"Retrieving context for query: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        # Generate query embedding - will raise RuntimeError if Ollama is not available
        try:
            query_embedding = self.embedder.embed_text(query)
        except Exception as e:
            log.error(f"Failed to generate embedding for query: {e}")
            raise RuntimeError(f"Failed to generate embedding for query: {e}")
        
        # Search for similar chunks
        try:
            results = self.vector_db.search(
                query_embedding, 
                top_k=top_k,
                threshold=threshold,
                source_filter=source_filter
            )
            
            # Update query count
            if hasattr(self, 'stats'):
                self.stats["queries_performed"] = self.stats.get("queries_performed", 0) + 1
            
            log.debug(f"Retrieved {len(results)} relevant chunks")
            return results
        except Exception as e:
            log.error(f"Vector search failed: {e}")
            raise RuntimeError(f"Failed to search for relevant context: {e}")
    

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG engine and database.
        
        Returns:
            Dict with statistics and metadata
        """
        db_stats = self.vector_db.get_stats()
        
        return {
            "run_id": self.run_id,
            "document_count": db_stats["document_count"],
            "chunk_count": db_stats["chunk_count"],
            "embedding_model": self.config["embedding_model"],
            "chunk_size": self.config["chunk_size"],
            "embedding_dimension": self.embedder.get_embedding_dimension(),
            "created_at": self.config["created_at"],
            "query_count": self.stats.get("queries_performed", 0) if hasattr(self, "stats") else 0
        }
    

    def purge(self) -> None:
        """
        Remove all data from the vector database.
        
        This operation cannot be undone.
        """
        log.warning(f"Purging all data from RAG engine {self.run_id}")
        
        # Close current connection
        self.vector_db.close()
        
        # Remove database file
        db_path = self.vector_dir / "vector.db"
        if db_path.exists():
            db_path.unlink()
        
        # Recreate database
        self.vector_db = VectorDatabase(db_path)
        
        # Reset stats
        self.stats = {
            "documents_added": 0,
            "chunks_added": 0,
            "queries_performed": 0
        }
        
        log.info("Database purged successfully")
    

    @classmethod
    def load(
        cls,
        run_id: str,
        base_dir: Optional[Union[str, Path]] = None,
        create_dirs: bool = True
    ) -> "RAGEngine":
        """
        Load an existing RAG engine by run ID.
        
        Args:
            run_id: ID of the run to load
            base_dir: Base directory where run data is stored
            create_dirs: Whether to create directories if they don't exist
            
        Returns:
            RAGEngine: Initialized RAG engine
            
        Raises:
            FileNotFoundError: If the run doesn't exist
            RuntimeError: If Ollama is not available
        """
        # First check if Ollama is available
        check_for_ollama()
        
        base_path = Path(base_dir) if base_dir else Path.cwd() / "output"
        run_dir = base_path / run_id
        db_path = run_dir / "vectors" / "vector.db"
        
        if not db_path.exists():
            raise FileNotFoundError(f"No RAG data found for run {run_id}")
            
        # Load configuration
        config_path = run_dir / "vectors" / "metadata.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Create new instance with loaded config
            engine = cls(
                run_id=config.get("run_id", run_id),  # Use config run_id if available
                base_dir=base_dir,
                embedding_model=config.get("embedding_model"),  # Use loaded model
                chunk_size=config.get("chunk_size", 512),
                chunk_overlap=config.get("chunk_overlap", 50),
                create_dirs=create_dirs
            )
            return engine
        else:
            # Use defaults if config not found
            return cls(run_id=run_id, base_dir=base_dir, create_dirs=create_dirs)
    

    def add_document_with_llama(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add document using LlamaIndex document loaders.
        
        Args:
            file_path: Path to file
            metadata: Optional metadata to associate with document
            
        Returns:
            Number of chunks added
            
        Raises:
            RuntimeError: If LlamaIndex is not available
        """
        if not self.document_loader:
            try:
                self.document_loader = LlamaDocumentLoader()
            except Exception as e:
                raise RuntimeError(f"Failed to initialize document loader: {e}")
        
        # Load document using LlamaIndex
        documents = self.document_loader.load(file_path)
        
        # Process each document
        total_chunks = 0
        for doc in documents:
            source = Path(file_path).name
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata.update(doc.get("metadata", {}))
            
            # Extract content and add to database
            text = doc.get("content", "")
            chunks_added = self.add_document(
                text=text,
                metadata=doc_metadata,
                source=source
            )
            
            total_chunks += chunks_added
            
        return total_chunks
