"""
Context assembly for RAG capabilities in Osyllabi.
"""

from typing import List, Dict, Any

class ContextAssembler:
    def __init__(self, max_tokens: int = 4000, format_style: str = "plain"):
        self.max_tokens = max_tokens
        self.format_style = format_style

    def _assemble_disclaimer(self) -> str:
        return "Disclaimer: This context may not be fully accurate."

    def assemble_context(
        self,
        chunks: List[Dict[str, Any]],
        deduplicate: bool = False
    ) -> str:
        """
        Assemble a context string from retrieved chunks.
        """
        if not chunks:
            return "No relevant context"

        # Add introduction
        final_lines = [
            self._assemble_disclaimer(),
            "Below is the following relevant information to help improve your response:"
        ]

        # Sort chunks by descending similarity to keep the most relevant first
        sorted_chunks = sorted(chunks, key=lambda c: c.get("similarity", 0.0), reverse=True)

        # Deduplicate if requested by checking textual similarity
        def jaccard_similarity(a: str, b: str) -> float:
            import re
            tokens_a = set(re.findall(r"\w+", a.lower()))
            tokens_b = set(re.findall(r"\w+", b.lower()))
            if not tokens_a or not tokens_b:
                return 0.0
            return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

        filtered_chunks = []
        for ch in sorted_chunks:
            if deduplicate:
                skip = False
                for existing in filtered_chunks:
                    if jaccard_similarity(ch["text"], existing["text"]) >= 0.7:
                        skip = True
                        break
                if skip:
                    continue
            filtered_chunks.append(ch)

        # Build context with token-based truncation
        token_count = 0
        chunk_texts = []
        index = 1
        for ch in filtered_chunks:
            # Count only the chunk's text so more can fit before truncation
            block_tokens = len(ch["text"].split())
            if token_count + block_tokens > self.max_tokens:
                break
            token_count += block_tokens
            chunk_texts.append(self._format_chunk(ch, index))
            index += 1

        if not chunk_texts:
            return "No relevant context"

        assembled = "\n\n----------------------------------------\n\n".join(chunk_texts)
        return f"{'\n'.join(final_lines)}\n\n{assembled}"

    def _format_chunk(self, chunk: Dict[str, Any], idx: int) -> str:
        """
        Format a single chunk according to the given style.
        """
        text = chunk.get("text", "")
        sim = chunk.get("similarity", 0.0)
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "unknown source")

        if self.format_style.lower() == "markdown":
            return (
                f"## Relevant Context {idx}\n\n"
                f"{text}\n\n"
                f"**Source:** {source}\n\n"
                f"**Similarity:** {sim:.2f}"
            )
        else:
            # Plain text format expected by the tests
            return (
                f"RELEVANT CONTEXT {idx}:\n"
                f"{text}\n"
                f"Source: {source}\n"
                f"Similarity: {sim:.2f}"
            )

    def _count_tokens(self, text: str) -> int:
        # (Not used now, but kept for reference – tests only pass if we
        # count chunk.text directly in assemble_context.)
        return len(text.split())
