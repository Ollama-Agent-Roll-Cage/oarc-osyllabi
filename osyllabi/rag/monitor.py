"""
Performance monitoring for RAG operations.

This module provides functionality to track and analyze the performance
of the RAG system to help with optimization and debugging.
"""
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from osyllabi.utils.log import log


class RAGMonitor:
    """
    Monitor performance metrics for RAG operations.
    
    This class tracks metrics such as latency, hit rates, and response
    quality to help optimize the RAG system.
    """
    
    def __init__(self, log_path: Optional[Union[str, Path]] = None):
        """
        Initialize the RAG monitor.
        
        Args:
            log_path: Optional path to store metrics
        """
        self.metrics = {
            "retrieval": {
                "count": 0,
                "total_time": 0.0,
                "chunk_count": 0,
                "hit_count": 0
            },
            "embedding": {
                "count": 0,
                "total_time": 0.0,
                "chunk_count": 0
            },
            "documents": {
                "count": 0,
                "total_chunks": 0
            }
        }
        
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            
        self.query_history = []
    
    def start_retrieval(self) -> int:
        """
        Start timing a retrieval operation.
        
        Returns:
            int: Retrieval ID for this operation
        """
        retrieval_id = self.metrics["retrieval"]["count"] + 1
        self.metrics["retrieval"]["count"] += 1
        
        # Store start time in the history
        self.query_history.append({
            "id": retrieval_id,
            "start_time": time.time(),
            "query": None,
            "results": [],
            "duration": 0.0
        })
        
        return retrieval_id
    
    def record_retrieval(
        self,
        retrieval_id: int,
        query: str,
        results: List[Dict[str, Any]],
        duration: Optional[float] = None
    ) -> None:
        """
        Record results from a retrieval operation.
        
        Args:
            retrieval_id: Retrieval operation ID
            query: The query used
            results: List of retrieved results
            duration: Optional duration (calculated if not provided)
        """
        # Find the right history entry
        entry = None
        for e in self.query_history:
            if e["id"] == retrieval_id:
                entry = e
                break
                
        if not entry:
            log.warning(f"No retrieval operation found with ID {retrieval_id}")
            return
        
        # Calculate duration if not provided
        if duration is None:
            duration = time.time() - entry["start_time"]
        
        # Update history entry
        entry["query"] = query
        entry["results"] = [{
            "id": r.get("id"),
            "source": r.get("source"),
            "similarity": r.get("similarity", 0)
        } for r in results]
        entry["duration"] = duration
        
        # Update aggregate metrics
        self.metrics["retrieval"]["total_time"] += duration
        self.metrics["retrieval"]["chunk_count"] += len(results)
        if results:
            self.metrics["retrieval"]["hit_count"] += 1
            
        # Log if path is specified
        if self.log_path:
            self._save_metrics()
    
    def record_embedding(
        self,
        chunk_count: int,
        duration: float
    ) -> None:
        """
        Record metrics from an embedding operation.
        
        Args:
            chunk_count: Number of chunks embedded
            duration: Duration of the embedding operation
        """
        self.metrics["embedding"]["count"] += 1
        self.metrics["embedding"]["total_time"] += duration
        self.metrics["embedding"]["chunk_count"] += chunk_count
        
        # Log if path is specified
        if self.log_path:
            self._save_metrics()
    
    def record_document_addition(self, chunk_count: int) -> None:
        """
        Record metrics for document addition.
        
        Args:
            chunk_count: Number of chunks added
        """
        self.metrics["documents"]["count"] += 1
        self.metrics["documents"]["total_chunks"] += chunk_count
        
        # Log if path is specified
        if self.log_path:
            self._save_metrics()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        # Calculate derived metrics
        metrics = self.metrics.copy()
        
        # Calculate averages
        if metrics["retrieval"]["count"] > 0:
            metrics["retrieval"]["avg_time"] = (
                metrics["retrieval"]["total_time"] / 
                metrics["retrieval"]["count"]
            )
            metrics["retrieval"]["avg_chunks"] = (
                metrics["retrieval"]["chunk_count"] / 
                metrics["retrieval"]["count"]
            )
            metrics["retrieval"]["hit_rate"] = (
                metrics["retrieval"]["hit_count"] / 
                metrics["retrieval"]["count"]
            )
        
        if metrics["embedding"]["count"] > 0:
            metrics["embedding"]["avg_time"] = (
                metrics["embedding"]["total_time"] / 
                metrics["embedding"]["count"]
            )
            metrics["embedding"]["avg_chunks"] = (
                metrics["embedding"]["chunk_count"] / 
                metrics["embedding"]["count"]
            )
        
        return metrics
    
    def get_recent_queries(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent query history.
        
        Args:
            count: Number of recent queries to return
            
        Returns:
            List[Dict[str, Any]]: Recent query history
        """
        return sorted(
            self.query_history,
            key=lambda x: x.get("start_time", 0),
            reverse=True
        )[:count]
    
    def _save_metrics(self) -> None:
        """Save metrics to the log file."""
        if not self.log_path:
            return
            
        try:
            with open(self.log_path, 'w') as f:
                json.dump({
                    "metrics": self.get_metrics(),
                    "recent_queries": self.get_recent_queries(5)
                }, f, indent=2)
        except Exception as e:
            log.error(f"Failed to save metrics: {e}")
