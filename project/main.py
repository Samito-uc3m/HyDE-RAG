from data_loader import load_documents
from llm_setup import get_llm
from embedding_setup import get_embedding_model
from vector_store_setup import create_vector_store, chunk_documents, create_nodes, embed_and_add_nodes
from retriever import VectorDBRetriever
from query_engine import create_query_engine
from config import SIMILARITY_TOP_K, QUERY_MODE

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
    query_str = "Can you tell me about the key concepts for safety finetuning"
    response = query_engine.query(query_str)
    print("Query 1:", query_str)
    print("Response:", str(response))

    # Another query
    query_str = "How does Llama 2 perform compared to other open-source models?"
    response = query_engine.query(query_str)
    print("Query 2:", query_str)
    print("Response:", str(response))
    if response.source_nodes:
        print("Source:", response.source_nodes[0].get_content())

if __name__ == "__main__":
    main()
