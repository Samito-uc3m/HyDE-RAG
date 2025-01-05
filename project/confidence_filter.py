from llama_index.core import QueryBundle
from doc_list import build_doc_list_response


def query_with_confidence(query_str: str, retriever, confidence_threshold: float = 0.8) -> str:
    """
    Retrieve top results for the query, check the highest similarity score,
    and only call the LLM if the score >= `confidence_threshold`.
    Otherwise, return an empty string or a custom fallback.
    """

    # Retrieve the nodes (directly from the retriever)
    query_bundle = QueryBundle(query_str)
    retrieved_nodes_with_scores = retriever._retrieve(query_bundle)
    
    # Get documents for the top k nodes
    retrieved_docs = build_doc_list_response(retrieved_nodes_with_scores)
    if not retrieved_docs:
        # No documents retrieved => zero confidence
        return []

    # Only return documents with confidence >= threshold
    retrieved_docs = [doc for doc in retrieved_docs if doc.similarity >= confidence_threshold]

    # Print the similarity score of all documents
    for doc in retrieved_docs:
        print(f"Document: {doc.title}, Similarity: {doc.similarity}")

    return retrieved_docs
