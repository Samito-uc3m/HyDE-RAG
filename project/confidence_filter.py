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
    retrieved_nodes_with_scores.sort(key=lambda x: x.score if x.score else 0, reverse=True)

    # Get documents for the top k nodes
    retrieved_docs = build_doc_list_response(retrieved_nodes_with_scores)

    print("\n===== DOCUMENT LIST =====\n")
    print(retrieved_docs)

    if not retrieved_docs:
        # No documents retrieved => zero confidence
        return []

    # Check top similarity (assuming the retriever sorted them by descending similarity)
    top_score = retrieved_docs[0].similarity
    print("top score", top_score)
    if top_score is None or top_score < confidence_threshold:
        return [] # Return empty/no answer

    return retrieved_docs
