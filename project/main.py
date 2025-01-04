from confidence_filter import query_with_confidence
from config import QUERY_MODE, SIMILARITY_TOP_K
from data_loader import load_documents
from embedding_setup import get_embedding_model
from llm_setup import get_llm
from query_engine import create_query_engine
from retriever import VectorDBRetriever
from vector_store_setup import (
    chunk_documents,
    create_nodes,
    create_vector_store,
    embed_and_add_nodes,
)


def main():
    # Load documents
    documents = load_documents()

    # Create vector store
    vector_store = create_vector_store()

    # Chunk documents into text nodes
    text_chunks, doc_idxs = chunk_documents(documents)
    nodes = create_nodes(documents, text_chunks, doc_idxs)

    # Setup embedding model and embed nodes
    embed_model = get_embedding_model()
    embed_and_add_nodes(nodes, embed_model, vector_store)

    # Setup LLM
    llm = get_llm()

    # Create retriever and query engine
    retriever = VectorDBRetriever(vector_store, embed_model, query_mode=QUERY_MODE, similarity_top_k=SIMILARITY_TOP_K)
    query_engine = create_query_engine(retriever, llm)

    print("im here")

    # Example queries
    query_str = "Renormalized quasiparticles in antiferromagnetic states of the Hubbard model"
    response = query_with_confidence(query_str, retriever, query_engine, confidence_threshold=0.5)
    print("Query 1:", query_str)

    if not response:
        print("Confidence < 50%. Returning no answer.")
    else:
        print("Final Answer:", str(response))

    # # Another query
    # query_str = "How does Llama 2 perform compared to other open-source models?"
    # response = query_engine.query(query_str)
    # print("Query 2:", query_str)
    # print("Response:", str(response))
    # if response.source_nodes:
    #     print("Source:", response.source_nodes[0].get_content())


if __name__ == "__main__":
    main()
