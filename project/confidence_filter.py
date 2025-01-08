"""
Módulo: get_docs_from_query.py

Módulo para realizar querys con un umbral de confianza mínimo.
"""

from config import settings
from doc_list import build_doc_list_response
from llama_index.core import QueryBundle


def query_with_confidence(query_str: str, retriever) -> str:
    """
    Realiza una query utilizando un umbral de confianza.

    Este método recibe una query como texto y utiliza un "retriever" para
    recuperar los documentos más relevantes. Evalúa la similitud de los resultados
    recuperados y sólo devuelve aquellos cuya similitud sea mayor o igual al umbral
    de confianza especificado (`confidence_threshold`).

    Parámetros:
    -----------
    query_str : str
        El texto de la query a realizar.
    retriever : object
        Un objeto encargado de realizar las búsquedas en el índice de datos.
    confidence_threshold : float, opcional
        El umbral mínimo de confianza para aceptar un resultado (por defecto es 0.8).

    Devuelve:
    --------
    list
        Una lista de documentos cuya similitud con la consulta supera el umbral de confianza.
        Si no se encuentra ningún documento que cumpla el criterio, retorna una lista vacía.
    """

    # Retrieve the nodes (directly from the retriever)
    query_bundle = QueryBundle(query_str)
    retrieved_nodes_with_scores = retriever._retrieve(query_bundle)

    # Get documents for the top k nodes
    retrieved_docs = build_doc_list_response(retrieved_nodes_with_scores)
    if not retrieved_docs:
        # No documents retrieved => zero confidence
        return []

    # Filter documents based on confidence threshold
    retrieved_docs = [
        doc
        for doc in retrieved_docs
        if doc.similarity >= settings.RETRIEVER_CONFIDENCE_THRESHOLD
    ]

    return retrieved_docs
