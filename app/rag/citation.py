class CitationBuilder:
    def build(retrieved_chunks: list[RetrievedChunk]) -> list[Citation]:
        citations = []
        for chunk in retrieved_chunks:
            citations.append(Citation(
                source=chunk.metadata.source,
                page=chunk.metadata.get("page"),
                chunk_index=chunk.metadata.chunk_index,
                score=chunk.score,
                snippet=truncate(chunk.text, 200)
            ))
        return dedupe_by_source(citations)