from confidence_filter import query_with_confidence
from config import settings
from data_loader import load_documents
from embedding_setup import get_embedding_model
from language_engine import detect_language, load_language_detection_model
from llm_setup import get_llm
from query_transformer import run_query_transformation_filter
from response_maker import run_response_maker
from retriever import VectorDBRetriever
from vector_store_setup import (
    chunk_documents,
    create_nodes,
    create_vector_store,
    embed_and_add_nodes,
)


def main():
    print("Creating vector store...")
    collection, vector_store = create_vector_store()

    print("Setting up embdding model...")
    embed_model = get_embedding_model()

    # Check if the vector store is empty
    if collection.count() == 0:
        print("Loading documents...")
        documents = load_documents(max_docs=5000)

        print("Splitting documents into chunks...")
        text_chunks, doc_idxs = chunk_documents(documents)

        print("Creating nodes...")
        nodes = create_nodes(documents, text_chunks, doc_idxs)

        print("Adding nodes to vector store...")
        embed_and_add_nodes(nodes, embed_model, vector_store)

    else:
        print(
            f"Skipping collection loading because it already contains {collection.count()} nodes."
        )

    print("Setting up LLM...")
    llm = get_llm()

    print("Setting up retriever...")
    retriever = VectorDBRetriever(
        vector_store=vector_store,
        embed_model=embed_model,
        query_mode=settings.QUERY_MODE,
        node_top_k=settings.NODE_TOP_K,
        document_top_k=settings.DOCUMENT_TOP_K,
    )

    print("Loading language detection model...")
    language_detection_model = load_language_detection_model()

    # Detect the language of the user query
    detected_language = detect_language(settings.USER_QUERY, language_detection_model)
    print(f"Detected language: {detected_language}")

    # Transform the user query
    transformed_query = run_query_transformation_filter(settings.USER_QUERY, llm)
    if "Output: " in transformed_query:
        transformed_query = transformed_query[
            transformed_query.rfind('Output: "') + len('Output: "') : -1
        ]
        print("Trimmed Query", transformed_query)

    print("\n===== TRIMMED DOCUMENT LIST =====\n")
    response = query_with_confidence(transformed_query, retriever)
    # print(response)

    correlation_response = run_response_maker(
        transformed_query, detected_language, response, llm
    )
    print("\n===== COMPILATION & DIFFERENCES =====\n")
    print(correlation_response)


if __name__ == "__main__":
    main()
