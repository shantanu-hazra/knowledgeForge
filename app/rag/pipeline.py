class RAGPipeline:
    def __init__(retriever, prompt_builder, llm_chat_client, citation_builder):
        store all

    def search(query: str) -> RAGResult:
        # exposed to agent as the tool's execute function
        retrieved_chunks = retriever.retrieve(query)
        if retrieved_chunks is empty:
            return RAGResult(answer=None, context_found=False)

        prompt = prompt_builder.build_rag_prompt(query, retrieved_chunks)
        llm_response = llm_chat_client.generate(prompt)
        citations = citation_builder.build(retrieved_chunks)

        return RAGResult(
            answer=llm_response.text,
            citations=citations,
            context_found=True,
            raw_chunks=retrieved_chunks
        )

    def ingest_document(file_path, document_id) -> IngestionResult:
        delegate to IngestionPipeline.ingest(...)