"""
Módulo: streamlit_app.py

Aplicación principal de Streamlit para un asistente de investigación basado en RAG (Retrieval-Augmented Generation).
Este módulo permite a los usuarios ingresar consultas de investigación, buscar respuestas
utilizando modelos de lenguaje y bases de datos vectoriales, desde la interfaz.
"""
import time

import streamlit as st
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


def setup():
    # --------------- Setup ---------------
    print("Creating vector store...")
    collection, vector_store = create_vector_store()

    print("Setting up embedding model...")
    embed_model = get_embedding_model()

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

    return (
        embed_model,
        llm,
        retriever,
        vector_store,
        collection,
        language_detection_model,
    )


# --------------- Database Loading Logic ---------------
def load_database(embed_model, vector_store):
    print("Loading documents...")
    documents = load_documents(max_docs=500)

    print("Splitting documents into chunks...")
    text_chunks, doc_idxs = chunk_documents(documents)

    print("Creating nodes...")
    nodes = create_nodes(documents, text_chunks, doc_idxs)

    print("Embedding and adding nodes to vector store...")
    embed_and_add_nodes(nodes, embed_model, vector_store)


def main():
    # Set the page title
    st.title("RAG-based Research Assistant")

    # Initialize session state variables
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "response" not in st.session_state:
        st.session_state.response = ""
    if "db_loaded" not in st.session_state:
        st.session_state.db_loaded = False
    if "show_success" not in st.session_state:
        st.session_state.show_success = False
    if "embed_model" not in st.session_state:
        # Setup the application
        (
            embed_model,
            llm,
            retriever,
            vector_store,
            collection,
            language_detection_model,
        ) = setup()

        # Save the setup in session state
        st.session_state.embed_model = embed_model
        st.session_state.llm = llm
        st.session_state.retriever = retriever
        st.session_state.vector_store = vector_store
        st.session_state.collection = collection
        st.session_state.language_detection_model = language_detection_model

    # Reset the success message state if not triggered by "Load Database"
    if st.session_state.show_success and not st.session_state.db_loaded:
        st.session_state.show_success = False

    # -------------------- USER INPUT --------------------
    user_query = st.text_area("Enter your line of investigation:")

    # If the user typed a new query, clear any previous response
    if user_query != st.session_state.last_query:
        st.session_state.response = ""
        st.session_state.last_query = user_query

    # ------------------- SEARCH BUTTON ------------------
    if st.button("Search"):
        if user_query.strip():
            # Call the RAG pipeline

            # Detect the language of the query
            detected_language = detect_language(
                user_query, st.session_state.language_detection_model
            )
            print(f"Detected language: {detected_language}")

            # Transform the query
            transformed_query = run_query_transformation_filter(
                user_query, st.session_state.llm
            )
            if "Output: " in transformed_query:
                transformed_query = transformed_query[
                    transformed_query.rfind('Output: "') + len('Output: "') : -1
                ]
            print(f"Transformed query: {transformed_query}")

            # Get the query response
            response = query_with_confidence(
                transformed_query, st.session_state.retriever
            )

            # Call the correlation filter
            result = run_response_maker(
                user_query, detected_language, response, st.session_state.llm
            )
            st.session_state.response = result
        else:
            st.write("Please enter a query.")

    # ------------------ DISPLAY RESPONSE ----------------
    st.markdown("### RAG Answer")
    st.write(st.session_state.response)

    # ------------------- LOAD DATABASE BUTTON -------------------
    # Use Streamlit's columns for alignment
    _, col2 = st.columns([7, 2])  # Adjust column ratio for alignment

    with col2:
        if st.button(
            "Load Database",
            disabled=st.session_state.db_loaded,
            key="load_db_button",
        ):
            if st.session_state.collection.count() == 0:
                load_database(
                    st.session_state.embed_model, st.session_state.vector_store
                )  # Load the database when button is clicked

            st.session_state.db_loaded = True
            st.session_state.show_success = True  # Trigger success message

    # ------------------ SUCCESS MESSAGE ------------------
    success_placeholder = st.empty()  # Create a placeholder for the success message

    if st.session_state.show_success:
        success_placeholder.success("Database successfully loaded!")
        time.sleep(3)  # Wait for 3 seconds
        success_placeholder.empty()  # Clear the message
        st.session_state.show_success = False


if __name__ == "__main__":
    main()
