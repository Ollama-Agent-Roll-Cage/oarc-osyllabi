"""
Vector math operations for RAG functionality.

This module provides advanced vector mathematics operations using scikit-learn
and other libraries to support retrieval-augmented generation capabilities.
"""
import numpy as np
from typing import Any, List, Optional, Tuple, Union, Dict

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from sklearn.preprocessing import normalize as sk_normalize

from osyllabi.utils.log import log
from osyllabi.utils.deps import DependencyManager

# Import FAISS directly
import faiss

# Define GPU capability flag here directly
FAISS_GPU_ENABLED = DependencyManager._is_faiss_gpu_installed()


def cosine_similarity(v1: Union[List[float], np.ndarray], v2: Union[List[float], np.ndarray]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        float: Cosine similarity score
        
    Raises:
        ValueError: If vectors have different dimensions
    """
    # Convert to numpy arrays if needed
    if not isinstance(v1, np.ndarray):
        v1 = np.array(v1)
    if not isinstance(v2, np.ndarray):
        v2 = np.array(v2)
        
    if v1.shape != v2.shape:
        raise ValueError(f"Vector dimensions do not match: {v1.shape} vs {v2.shape}")
        
    # Calculate cosine similarity
    dot = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Handle zero vectors
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    return float(dot / (norm_v1 * norm_v2))


def normalize_vector(vector: Union[List[float], np.ndarray]) -> List[float]:
    """
    Normalize a vector to unit length using scikit-learn.
    
    Args:
        vector: Vector to normalize
        
    Returns:
        List[float]: Normalized vector
    """
    v = np.array(vector, dtype=np.float32).reshape(1, -1)
    
    # Use scikit-learn's normalize which is optimized for L2 normalization
    normalized = sk_normalize(v, norm='l2')
    
    return normalized.flatten().tolist()


def mean_vector(vectors: List[List[float]]) -> List[float]:
    """
    Calculate the mean of multiple vectors.
    
    Args:
        vectors: List of vectors
        
    Returns:
        List[float]: Mean vector
        
    Raises:
        ValueError: If no vectors provided or vectors have different dimensions
    """
    if not vectors:
        raise ValueError("No vectors provided")
    
    # Convert to numpy array
    np_vectors = np.array(vectors, dtype=np.float32)
    
    # Calculate mean
    mean = np.mean(np_vectors, axis=0)
    
    return mean.tolist()


def concatenate_vectors(vectors: List[List[float]], 
                       weights: Optional[List[float]] = None) -> List[float]:
    """
    Concatenate multiple vectors with optional weighting.
    
    Args:
        vectors: List of vectors to concatenate
        weights: Optional weights for each vector
        
    Returns:
        List[float]: Concatenated vector
        
    Raises:
        ValueError: If weights are provided but don't match vector count
    """
    if not vectors:
        return []
    
    if weights and len(weights) != len(vectors):
        raise ValueError("Number of weights must match number of vectors")
    
    # Apply weights if provided
    if weights:
        weighted_vectors = []
        for vec, weight in zip(vectors, weights):
            weighted_vectors.append([v * weight for v in vec])
        vectors = weighted_vectors
    
    # Concatenate
    result = []
    for vec in vectors:
        result.extend(vec)
    
    return result


def reduce_dimensions(vectors: List[List[float]], target_dims: int) -> List[List[float]]:
    """
    Reduce dimensionality of vectors using PCA.
    
    Args:
        vectors: List of vectors to reduce
        target_dims: Target number of dimensions
        
    Returns:
        List[List[float]]: Reduced dimension vectors
        
    Raises:
        ValueError: If no vectors provided or target_dims is invalid
    """
    if not vectors:
        raise ValueError("No vectors provided")
        
    if target_dims < 1:
        raise ValueError("Target dimensions must be at least 1")
        
    # Convert to numpy array
    np_vectors = np.array(vectors, dtype=np.float32)
    
    # Use scikit-learn's PCA for dimension reduction
    pca = PCA(n_components=min(target_dims, np_vectors.shape[1]))
    reduced = pca.fit_transform(np_vectors)
    
    # Convert back to list format
    return reduced.tolist()


def batch_cosine_similarity(query_vector: Union[List[float], np.ndarray],
                           vectors: List[List[float]]) -> List[float]:
    """
    Compute cosine similarity between a query vector and multiple vectors.
    
    Args:
        query_vector: Query vector
        vectors: List of vectors to compare against
        
    Returns:
        List[float]: List of similarity scores
    """
    if not vectors:
        return []
        
    # Convert to numpy arrays
    q_vec = np.array(query_vector, dtype=np.float32).reshape(1, -1)
    all_vecs = np.array(vectors, dtype=np.float32)
    
    # Compute similarities
    similarities = sk_cosine_similarity(q_vec, all_vecs)[0]
    
    return similarities.tolist()


def create_faiss_index(vectors: List[List[float]], use_gpu: bool = True) -> Any:
    """
    Create a FAISS index for efficient similarity search.
    
    Args:
        vectors: Vectors to index
        use_gpu: Whether to use GPU acceleration if available
        
    Returns:
        FAISS index object or None if FAISS is not available
        
    Raises:
        ValueError: If vectors format is invalid
    """
    if not vectors:
        raise ValueError("No vectors provided to index")
        
    # Convert to numpy array
    np_vectors = np.array(vectors, dtype=np.float32)
    
    # Get dimensionality
    d = np_vectors.shape[1]
    
    # Create L2 index
    index = faiss.IndexFlatL2(d)
    
    # Use GPU if requested and available
    if use_gpu and FAISS_GPU_ENABLED:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        log.info("Using GPU-accelerated FAISS index")
    elif use_gpu and not FAISS_GPU_ENABLED:
            raise RuntimeError("GPU acceleration requested but FAISS GPU support not available")
    
    # Add vectors to the index
    index.add(np_vectors)
    
    return index


def faiss_search(index: Any, query_vector: List[float], k: int = 5) -> Tuple[List[float], List[int]]:
    """
    Search a FAISS index for similar vectors.
    
    Args:
        index: FAISS index created with create_faiss_index
        query_vector: Query vector
        k: Number of results to return
        
    Returns:
        Tuple of (distances, indices)
    """
    # Convert query to numpy array
    q_vec = np.array([query_vector], dtype=np.float32)
    
    # Search the index
    distances, indices = index.search(q_vec, k)
    
    # Convert to Python lists
    return distances[0].tolist(), indices[0].tolist()


def weighted_average_vectors(vectors: List[List[float]], weights: List[float]) -> List[float]:
    """
    Calculate the weighted average of vectors.
    
    Args:
        vectors: List of vectors to average
        weights: Weight for each vector (must sum to 1.0)
        
    Returns:
        List[float]: Weighted average vector
        
    Raises:
        ValueError: If vectors or weights are invalid
    """
    if not vectors or not weights:
        raise ValueError("Empty vectors or weights provided")
        
    if len(vectors) != len(weights):
        raise ValueError("Number of vectors must match number of weights")
        
    # Check weights approximately sum to 1.0
    weight_sum = sum(weights)
    if not 0.99 <= weight_sum <= 1.01:
        log.warning(f"Weights sum to {weight_sum}, not 1.0. Normalizing.")
        weights = [w / weight_sum for w in weights]
    
    # Convert to numpy arrays
    np_vectors = np.array(vectors, dtype=np.float32)
    np_weights = np.array(weights, dtype=np.float32).reshape(-1, 1)
    
    # Calculate weighted average
    weighted_avg = np.sum(np_vectors * np_weights, axis=0)
    
    return weighted_avg.tolist()


def find_diverse_vectors(vectors: List[List[float]], count: int = 3, 
                        min_similarity_threshold: float = 0.7) -> List[int]:
    """
    Find a diverse subset of vectors by maximizing distance between them.
    
    Args:
        vectors: List of vectors to select from
        count: Number of diverse vectors to select
        min_similarity_threshold: Minimum similarity threshold
        
    Returns:
        List[int]: Indices of the selected diverse vectors
    """
    if not vectors:
        return []
        
    if count > len(vectors):
        count = len(vectors)
        
    # Convert to numpy array
    np_vectors = np.array(vectors, dtype=np.float32)
    
    # Start with the first vector
    selected_indices = [0]
    
    # Iteratively add the most diverse vector
    while len(selected_indices) < count:
        max_min_distance = -1
        next_index = -1
        
        # For each candidate vector
        for i in range(len(np_vectors)):
            if i in selected_indices:
                continue  # Skip already selected vectors
                
            # Calculate minimum distance to any selected vector
            min_distance = float('inf')
            for j in selected_indices:
                sim = cosine_similarity(np_vectors[i], np_vectors[j])
                distance = 1.0 - sim  # Convert similarity to distance
                min_distance = min(min_distance, distance)
            
            # If this vector is more diverse than current best
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                next_index = i
        
        # Add most diverse vector
        if next_index != -1:
            selected_indices.append(next_index)
    
    return selected_indices


def detect_outliers(vectors: List[List[float]], threshold: float = 1.5) -> List[int]:
    """
    Detect outlier vectors based on distance from centroid.
    
    Args:
        vectors: List of vectors to analyze
        threshold: Standard deviation threshold for outlier detection
        
    Returns:
        List[int]: Indices of outlier vectors
    """
    if not vectors or len(vectors) < 2:
        return []
        
    # Convert to numpy array
    np_vectors = np.array(vectors, dtype=np.float32)
    
    # Calculate centroid
    centroid = np.mean(np_vectors, axis=0)
    
    # Calculate distances from centroid using Euclidean distance
    distances = np.linalg.norm(np_vectors - centroid, axis=1)
    
    # Convert distances to z-scores for better outlier detection
    std_dist = np.std(distances)
    
    if std_dist == 0:  # Avoid division by zero if all vectors are identical
        return []
    
    outlier_indices = []
    for i, dist in enumerate(distances):
        if dist > threshold * std_dist:
            outlier_indices.append(i)
    
    return outlier_indices


def compute_vector_stats(vectors: List[List[float]]) -> Dict[str, Any]:
    """
    Compute statistical properties of a collection of vectors.
    
    Args:
        vectors: List of vectors to analyze
        
    Returns:
        Dict: Statistical properties including:
            - dim: Dimensionality
            - mean: Mean vector
            - variance: Variance across dimensions
            - min_magnitude: Minimum vector magnitude
            - max_magnitude: Maximum vector magnitude
            - avg_magnitude: Average vector magnitude
    """
    if not vectors:
        return {
            "dim": 0,
            "count": 0,
            "mean": [],
            "variance": [],
            "min_magnitude": 0,
            "max_magnitude": 0,
            "avg_magnitude": 0
        }
        
    # Convert to numpy array
    np_vectors = np.array(vectors, dtype=np.float32)
    
    # Compute statistics
    dim = np_vectors.shape[1]
    mean_vec = np.mean(np_vectors, axis=0)
    variance = np.var(np_vectors, axis=0)
    
    # Compute magnitudes
    magnitudes = np.linalg.norm(np_vectors, axis=1)
    min_mag = float(np.min(magnitudes))
    max_mag = float(np.max(magnitudes))
    avg_mag = float(np.mean(magnitudes))
    
    return {
        "dim": int(dim),
        "count": len(vectors),
        "mean": mean_vec.tolist(),
        "variance": variance.tolist(),
        "min_magnitude": min_mag,
        "max_magnitude": max_mag,
        "avg_magnitude": avg_mag
    }
