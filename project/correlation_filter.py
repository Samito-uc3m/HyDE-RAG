"""
Módulo: correlation_filter.py

Módulo para construir prompts y ejecutar un filtro de correlación entre una consulta
del usuario y documentos recuperados. Permite identificar diferencias, brechas y similitudes 
entre la consulta y el contenido de los documentos mediante un modelo de lenguaje.
"""

from typing import List

from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
)
from doc_list import DocListResponse



def build_correlation_prompt(query_str: str, output_language: str, retrieved_docs: List[DocListResponse]) -> str:
    """
    Construye un prompt que:
    1. Resume el contenido relevante de cada documento (la compilación).
    2. Destaca diferencias o brechas entre la consulta del usuario y los documentos.

    Parámetros:
    -----------
    query_str : str
        El texto de la consulta del usuario.
    output_language : str
        El idioma en el que se debe generar la respuesta.
    retrieved_docs : List[DocListResponse]
        Una lista de documentos relevantes recuperados para la consulta.

    Devuelve:
    --------
    str
        El texto del prompt estructurado para ser usado por un modelo de lenguaje.
    """
    doc_descriptions = "\n".join(
        f"Document {i+1}: Title: '{doc.title}' | Similarity Score: {doc.similarity}\nAbstract: {doc.abstract}"
        for i, doc in enumerate(retrieved_docs)
    )

    prompt = (
        "User Query:\n"
        f"\"{query_str}\"\n\n"
        "Retrieved Documents for Comparison:\n"
        f"{doc_descriptions}\n\n"

        "Your response should synthesize the comparison in a concise, professional summary."
    )

    return prompt


def run_correlation_filter(query_str: str, output_language: str, retrieved_docs: List[DocListResponse], llm) -> ChatResponse:
    """
    Ejecuta un filtro de correlación entre la consulta del usuario y los documentos recuperados.
    
    Este método construye un prompt basado en la consulta del usuario y los documentos relevantes,
    llama a un modelo de lenguaje (LLM) para analizar la correlación y devuelve un resultado estructurado.

    Parámetros:
    -----------
    query_str : str
        El texto de la consulta del usuario.
    output_language : str
        El idioma en el que se debe generar la respuesta.
    retrieved_docs : List[DocListResponse]
        Una lista de documentos relevantes recuperados para la consulta.
    llm : object
        El modelo de lenguaje encargado de procesar el prompt y devolver el resultado.

    Devuelve:
    --------
    ChatResponse
        La respuesta estructurada generada por el modelo de lenguaje.
    """

    # Build the prompt text
    prompt_text = build_correlation_prompt(query_str, output_language, retrieved_docs)

    # Define the message roles for better alignment with the LLM API
    system_content = (
        "You are a specialized AI assistant focused on research analysis and comparison. Your task is to compare a user's research query with relevant papers, "
        "summarize key findings, highlight similarities, and note any gaps or differences. Respond clearly and concisely in {output_language}."
    )

    user_instructions = (
        "Instructions:\n"
        "1. Summarize important points from documents if they match or relate closely to the user's query.\n"
        "2. Mention the titles of matching documents and their similarity scores.\n"
        "3. Identify and explain any differences, gaps, or conflicts between the user's query and document content.\n"
        "4. If no documents closely match the query topic, clearly state 'I have not found relevant documents about the topic you are researching.'"
    )

    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=system_content
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=prompt_text
        ),
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=user_instructions
        )
    ]

    # Pass messages=list_of_dicts instead of a single string
    response = llm.chat(messages=messages).raw['choices'][0]['text']
    return response