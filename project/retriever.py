"""
Módulo: retriever.py

Módulo que implementa una clase para la recuperación de documentos relevantes a partir de
un almacén vectorial, utilizando un modelo de incrustaciones para consultas.
Proporciona funcionalidad para integrar recuperación basada en similitud con LlamaIndex.
"""

from typing import Any, List, Optional
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core import QueryBundle
from llama_index.core.vector_stores import VectorStoreQuery
from typing import List, Optional, Any


class VectorDBRetriever(BaseRetriever):
    """
    Recuperador de documentos basado en un almacén vectorial.

    Esta clase utiliza un modelo de incrustaciones para generar representaciones vectoriales
    de consultas y realiza búsquedas en un almacén vectorial para obtener los nodos
    más relevantes.

    Parámetros:
    -----------
    vector_store : object
        Instancia del almacén vectorial donde se encuentran los datos.
    embed_model : Any
        Modelo utilizado para generar incrustaciones de texto.
    query_mode : str, opcional
        Modo de consulta utilizado en la búsqueda (por defecto, "default").
    node_top_k : int, opcional
        Número máximo de nodos devueltos por la consulta (por defecto, 20).
    document_top_k : int, opcional
        Número máximo de documentos únicos seleccionados (por defecto, 5).
    """

    def __init__(
        self,
        vector_store,
        embed_model: Any,
        query_mode: str = "default",
        node_top_k: int = 20,
        document_top_k: int = 5,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._node_top_k = node_top_k
        self._document_top_k = document_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Recupera nodos relevantes basados en una consulta vectorial.

        Este método utiliza el modelo de incrustaciones para convertir la consulta en un
        vector, realiza una consulta en el almacén vectorial y selecciona los nodos más
        relevantes según su similitud.

        Parámetros:
        -----------
        query_bundle : QueryBundle
            Un objeto que contiene la cadena de consulta y metadatos adicionales.

        Devuelve:
        --------
        List[NodeWithScore]
            Una lista de nodos con sus respectivas puntuaciones de similitud.
        """
        
        # 1. Embed the query and retrieve the top-N nodes
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._node_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)
        
        # The vector store results should already be in descending order of similarity.
        # If not, you can manually sort by similarity here.

        # 2. Collect the top-K nodes from distinct sources
        selected_nodes: List[NodeWithScore] = []
        seen_sources = set()

        for index, node in enumerate(query_result.nodes):
            if len(selected_nodes) >= self._document_top_k:
                # If we already have enough distinct documents, break out
                break

            similarity: Optional[float] = None
            if query_result.similarities:
                similarity = query_result.similarities[index]

            source = node.metadata.get("source", "unknown_source")

            # If this node's source has not been seen yet, select it
            if source not in seen_sources:
                seen_sources.add(source)
                selected_nodes.append(NodeWithScore(node=node, score=similarity))

        return selected_nodes
