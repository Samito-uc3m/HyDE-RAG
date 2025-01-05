import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HOME"] = "/home/sam/.cache/huggingface/"
os.environ["HF_TOKEN"] = "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"


from confidence_filter import query_with_confidence
from config import QUERY_MODE, SIMILARITY_TOP_K
from correlation_filter import run_correlation_filter
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
    print("Loading documents...")
    documents = load_documents(max_docs=500)

    print("Creating vector store...")
    vector_store = create_vector_store()

    print("Splitting documents into chunks...")
    text_chunks, doc_idxs = chunk_documents(documents)

    print("Creating nodes...")
    nodes = create_nodes(documents, text_chunks, doc_idxs)

    print("Embedding and adding nodes to vector store...")
    embed_model = get_embedding_model()
    embed_and_add_nodes(nodes, embed_model, vector_store)

    print("Setting up LLM...")
    llm = get_llm()

    print("Setting up retriever...")
    retriever = VectorDBRetriever(
        vector_store=vector_store, embed_model=embed_model, query_mode=QUERY_MODE, similarity_top_k=SIMILARITY_TOP_K
    )

    # print("Creating query engine...")
    # query_engine = create_query_engine(retriever, llm)

    # EXAMPLE: Return a list of relevant docs (rather than an LLM answer)
    user_query = "Renormalized quasiparticles in antiferromagnetic states of the Hubbard model"
    # user_query = "Fc Barcelona financial fair play problems with Dani Olmo and Pau Victor"
    print(f"\nUser Query: {user_query}")
    confidence_threshold = 0.5

    response = query_with_confidence(user_query, retriever, confidence_threshold)
    print("\n===== FILTERED DOCUMENT LIST =====\n")
    print(response)

    correlation_response = run_correlation_filter(user_query, response, llm)
    print("\n===== COMPILATION & DIFFERENCES =====\n")
    print(correlation_response)


if __name__ == "__main__":
    main()
