# Retrieval-Augmented Generation (RAG) System

This document explains how Osyllabi uses RAG to enhance curriculum generation by leveraging vector embeddings to retrieve relevant context.

## Architecture Overview

```mermaid
flowchart TD
    subgraph ResourceProcessing ["Resource Processing"]
        Input[Resource Content] --> Chunking[Text Chunker]
        Chunking --> Embedding[Vector Embedding]
        Embedding --> Storage[Vector Database]
    end
    
    subgraph QueryProcessing ["Query Processing"] 
        Query[Agent Query] --> QueryForm[Query Formulator]
        QueryForm --> QueryEmbed[Embed Query]
        QueryEmbed --> VectorSearch[Vector Search]
        Storage --> VectorSearch
        VectorSearch --> Ranking[Similarity Ranking]
        Ranking --> ContextAssembly[Context Assembly]
    end
    
    subgraph Integration ["Prompt Integration"]
        ContextAssembly --> Prompt[Augmented Prompt]
        Prompt --> LLM[AI Generation]
        LLM --> Output[Enhanced Content]
    end
    
    subgraph Optimization ["System Optimization"]
        direction TB
        
        GPU{GPU Available?} -->|Yes| FGPU[FAISS-GPU]
        GPU -->|No| FCPU[FAISS-CPU]
        
        subgraph Monitoring ["System Monitoring"]
            Monitor[RAG Monitor]
            Stats[(Performance Stats)]
            Monitor --> Stats
        end
        
        FGPU & FCPU -.-> VectorSearch
    end
    
    subgraph Dependencies ["Required Dependencies"]
        direction TB
        Ollama[Ollama API] -.-> Embedding & LLM
        LangChain[LangChain] -.-> Chunking
        SQLite[(SQLite)] -.-> Storage
        FAISS[FAISS] -.-> VectorSearch
    end
```

## Required Dependencies

Osyllabi's RAG system has strict dependency requirements:

1. **Ollama**: Mandatory for embedding generation and inferencing
   - Must be installed and running at system startup
   - No fallback mechanism - system will exit if unavailable
   - Visit [Ollama Download Page](https://ollama.ai/download) for installation

2. **LangChain**: Required for text chunking functionality
   - Used for optimal document segmentation
   - Installed automatically with Osyllabi

3. **SQLite**: Used for the vector database (included in Python standard library)

4. **FAISS**: Automatically used for efficient vector search when available
   - CPU version used by default
   - GPU version used when hardware is available

## System Architecture

The RAG system is composed of several integrated components:

### 1. RAG Engine

The central coordinator for the entire RAG process:

```python
# Initialize RAG engine
rag_engine = RAGEngine(
    run_id="curriculum_12345",             # Unique ID for this session
    embedding_model="llama3",              # Model for embedding generation
    chunk_size=512,                        # Size of text chunks
    chunk_overlap=50                       # Overlap between chunks
)

# Add documents to knowledge base
doc_count = rag_engine.add_document(
    text="Machine learning is a field of study...",
    metadata={"source": "textbook", "chapter": "introduction"}
)

# Retrieve relevant context
context = rag_engine.retrieve(
    query="How to explain gradient descent to beginners?",
    top_k=5,                               # Return top 5 most relevant chunks
    threshold=0.7                          # Minimum similarity threshold
)
```

### 2. Document Processing Pipeline

The process of converting raw documents into searchable vectors:

1. **Text Chunking**: Splits documents into optimal segments

   ```python
   chunker = TextChunker(chunk_size=512, overlap=50)
   chunks = chunker.chunk_text(document_text)
   ```

2. **Embedding Generation**: Creates vector representations

   ```python
   embedder = EmbeddingGenerator(model_name="llama3")
   vectors = embedder.embed_texts(chunks)
   ```

3. **Vector Storage**: Persists embeddings for retrieval

   ```python
   vector_db = VectorDatabase("./vectors.db")
   vector_db.add_document(chunks, vectors, metadata)
   ```

### 3. Query Processing Pipeline

The process of finding relevant context for a query:

1. **Query Formulation**: Creates effective retrieval queries

   ```python
   formulator = QueryFormulator()
   optimized_query = formulator.formulate_query(
       topic="Machine Learning",
       query_type="learning_path"
   )
   ```

2. **Vector Search**: Finds similar content

   ```python
   query_embedding = embedder.embed_text(optimized_query)
   results = vector_db.search(
       query_embedding,
       top_k=5,
       threshold=0.7
   )
   ```

3. **Context Assembly**: Formats retrieved chunks

   ```python
   assembler = ContextAssembler(format_style="markdown")
   formatted_context = assembler.assemble_context(results)
   ```

### 4. Agent Integration

Agents can leverage RAG to enhance their capabilities:

```python
# Create RAG-enhanced agent
agent = RAGEnhancedAgent("learning_path_agent", rag_engine=rag_engine)

# Create enhanced prompt with relevant context
enhanced_prompt = agent.create_enhanced_prompt(
    base_prompt="Create a learning path for {topic} at {skill_level} level.",
    topic="Machine Learning",
    query_type="learning_path",
    skill_level="Beginner"
)

# Use enhanced prompt for generation
learning_path = ollama_client.generate(enhanced_prompt)
```

## Performance Optimization

The system includes multiple optimization strategies:

### Vector Search Acceleration

```python
# Detect GPU and use FAISS with acceleration
has_gpu, gpu_info = detect_gpu()

# Create optimized index
vectors = [doc.embedding for doc in documents]
index = create_faiss_index(vectors, use_gpu=has_gpu)

# Perform optimized search
distances, indices = faiss_search(index, query_vector, k=5)
```

### Performance Monitoring

```python
# Initialize monitor
monitor = RAGMonitor(log_path="./rag_metrics.json")

# Record retrieval operations
retrieval_id = monitor.start_retrieval()
results = rag_engine.retrieve(query, top_k=5)
monitor.record_retrieval(retrieval_id, query, results)

# Get performance metrics
metrics = monitor.get_metrics()
print(f"Average retrieval time: {metrics['retrieval']['avg_time']:.3f}s")
print(f"Hit rate: {metrics['retrieval']['hit_rate']*100:.1f}%")
```

## Configuration Options

System behavior can be customized through environment variables:

```python
# RAG Configuration
RAG_CONFIG = {
    'embedding_model': os.getenv('OSYLLABUS_EMBEDDING_MODEL', 'llama3'),
    'chunk_size': int(os.getenv('OSYLLABUS_CHUNK_SIZE', '512')),
    'chunk_overlap': int(os.getenv('OSYLLABUS_CHUNK_OVERLAP', '50')),
    'retrieval_top_k': int(os.getenv('OSYLLABUS_RETRIEVAL_TOP_K', '5')),
    'similarity_threshold': float(os.getenv('OSYLLABUS_SIMILARITY_THRESHOLD', '0.7'))
}

# FAISS Configuration
FAISS_CONFIG = {
    'use_gpu': os.getenv('OSYLLABUS_FAISS_USE_GPU', 'true').lower() in ('true', 'yes', '1')
}
```

## Error Handling

The RAG system implements comprehensive error handling:

1. **Dependency Verification**: Checks for Ollama at startup
2. **Graceful Degradation**: Falls back to simpler methods when optimal ones fail
3. **Detailed Logging**: Records performance metrics and error information
4. **Retry Mechanisms**: Automatically retries transient failures

## Future Enhancements

1. **Hybrid Search**: Combine vector retrieval with keyword-based filtering
2. **Multi-source Weighting**: Prioritize trusted sources for particular knowledge domains
3. **Contextual Feedback**: Incorporate user feedback to improve context quality
4. **Cross-run Knowledge**: Enable knowledge sharing between curriculum runs
5. **Advanced Chunking**: Semantically-aware document segmentation
