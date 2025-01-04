import json

from llama_index.core import QueryBundle


def retrieve_documents(query_str, retriever, top_k=5):
    """
    Directly call the retriever to get top_k nodes.
    Returns a list of NodeWithScore, sorted by descending similarity.
    """
    query_bundle = QueryBundle(query_str)
    nodes_with_scores = retriever._retrieve(query_bundle)
    # sort by score desc, just in case
    nodes_with_scores.sort(key=lambda x: x.score if x.score else 0, reverse=True)
    return nodes_with_scores[:top_k]


def build_doc_list_response(query_str, nodes_with_scores):
    """
    Create a JSON or structured string listing each retrieved document,
    including metadata and similarity score.
    """
    doc_list = []
    for i, nws in enumerate(nodes_with_scores):
        metadata = nws.node.metadata or {}
        doc_score = nws.score if nws.score else 0
        doc_list.append(
            {
                "index": i + 1,
                "title": metadata.get("title", "No Title"),
                "source_id": metadata.get("source", f"doc_{i}"),
                "similarity": round(doc_score, 4),
            }
        )

    response = {"query": query_str, "documents": doc_list}
    # Return as a JSON-formatted string
    return json.dumps(response, indent=2)
