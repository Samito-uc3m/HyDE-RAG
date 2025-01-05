import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HOME"] = "/home/sam/.cache/huggingface/"
os.environ["HF_TOKEN"] = "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"


from confidence_filter import query_with_confidence
from config import QUERY_MODE, NODE_TOP_K, DOCUMENT_TOP_K
from correlation_filter import run_correlation_filter
from query_transformer import run_query_transformation_filter
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
        vector_store=vector_store,
        embed_model=embed_model,
        query_mode=QUERY_MODE,
        node_top_k=NODE_TOP_K,
        document_top_k=DOCUMENT_TOP_K
    )

    # print("Creating query engine...")
    # query_engine = create_query_engine(retriever, llm)

    # User query
    user_query = "I am researching about the renormalized quasiparticles in antiferromagnetic states of the Hubbard model, could you please check and find any relevant documents?"
    user_query = "Estoy investigando sobre las quasi-partículas renormalizadas en estados antiferromagnéticos del modelo de Hubbard. ¿Podrías, por favor, buscar y encontrar documentos relevantes?"
    transformed_query = run_query_transformation_filter(user_query, llm)

    if "Output: " in transformed_query.raw["choices"][0]["text"]:
        # Find the last index of the string "Output: "
        transformed_query = transformed_query.raw["choices"][0]["text"][transformed_query.raw["choices"][0]["text"].rfind('Output: "') + len('Output: "'): -1]
        print("Trimmed Query", transformed_query)
    confidence_threshold = 0.8

    print("\n===== TRIMMED DOCUMENT LIST =====\n")
    response = query_with_confidence(transformed_query, retriever, confidence_threshold)
    # print(response)

    if len(response) == 0:
        print('I have not found relevant documents about the topic you are researching.')
        return

    correlation_response = run_correlation_filter(user_query, response, llm)
    print("\n===== COMPILATION & DIFFERENCES =====\n")
    print(correlation_response)


if __name__ == "__main__":
    main()
