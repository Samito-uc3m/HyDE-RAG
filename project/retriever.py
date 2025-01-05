from typing import Any, List, Optional
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core import QueryBundle
from llama_index.core.vector_stores import VectorStoreQuery
from typing import List, Optional, Any


class VectorDBRetriever(BaseRetriever):
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
