"""
Módulo: doc_list.py

Módulo para construir una lista estructurada de documentos.
"""

from pydantic import BaseModel


class DocListResponse(BaseModel):
    """
    Modelo de datos para representar un documento con metadatos y puntaje de similitud.

    Atributos:
    ----------
    index : int
        El índice del documento en la lista.
    title : str
        El título del documento.
    abstract : str
        El resumen del documento.
    source_id : str
        El identificador de la fuente del documento.
    similarity : float
        El puntaje de similitud asociado al documento.
    """
    index: int
    title: str
    abstract: str
    source_id: str
    similarity: float


def build_doc_list_response(nodes_with_scores) -> list[DocListResponse]:
    """
    Construye una lista estructurada de documentos.

    Este método toma una lista de nodos con sus puntajes de similitud y crea
    una lista de objetos `DocListResponse`, que incluyen el índice, título,
    resumen, identificador de fuente, y puntaje de similitud.

    Parámetros:
    -----------
    nodes_with_scores : list
        Una lista de objetos que contienen nodos recuperados y sus puntajes de similitud.

    Devuelve:
    --------
    list[DocListResponse]
        Una lista de objetos `DocListResponse` con los datos estructurados de los documentos.
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
