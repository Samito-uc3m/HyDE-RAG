import json
from pydantic import BaseModel


class DocListResponse(BaseModel):
    index: int
    title: str
    abstract: str
    source_id: str
    similarity: float


def build_doc_list_response(nodes_with_scores) -> list[DocListResponse]:
    """
    Create a JSON or structured string listing each retrieved document,
    including metadata and similarity score.
    """
    doc_list = []
    for i, nws in enumerate(nodes_with_scores):
        metadata = nws.node.metadata or {}
        doc_score = nws.score if nws.score else 0
        doc_list.append(
            DocListResponse(
                index=i + 1,
                title=metadata.get("title", "No Title"),
                abstract=metadata.get("abstract", "No Abstract"),
                source_id=metadata.get("source", f"doc_{i}"),
                similarity=round(doc_score, 4),
            )
        )

    return doc_list
