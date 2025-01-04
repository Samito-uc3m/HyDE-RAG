from llama_index.core import QueryBundle


def query_with_confidence(query_str: str, retriever, query_engine, confidence_threshold: float = 0.8) -> str:
    """
    Retrieve top results for the query, check the highest similarity score,
    and only call the LLM if the score >= `confidence_threshold`.
    Otherwise, return an empty string or a custom fallback.
    """

    # Retrieve the nodes (directly from the retriever)
    query_bundle = QueryBundle(query_str)
    retrieved_nodes_with_scores = retriever._retrieve(query_bundle)

    if not retrieved_nodes_with_scores:
        # No documents retrieved => zero confidence
        return ""

    # Check top similarity (assuming the retriever sorted them by descending similarity)
    top_score = retrieved_nodes_with_scores[0].score
    print("top score", top_score)
    if top_score is None or top_score < confidence_threshold:
        return ""  # Return empty/no answer

    # If above threshold, proceed to query with the LLM
    response = query_engine.query(query_str)
    return str(response)
