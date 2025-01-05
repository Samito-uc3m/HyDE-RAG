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
    doc_summaries = ""
    for i, doc in enumerate(retrieved_docs):
        doc_summaries += (
            f"\nDocument {i} (Title: '{doc.title}', Abstract: '{doc.abstract}', Score: {doc.similarity}):\n"
        )

    prompt = f"""
                System: You are an assistant that compares the user's research query to the provided documents.
                Produce a compilation of key points from the documents and highlight any differences or gaps.
                Please provide your response in a single paragraph. Please provide your answer in {output_language} language.
                If the retrieved documents do not have relevant information about the query please only state
                    'I have not found relevant documents about the topic you are researching.' and there is no need to follow the instructions bellow.

                User Query:
                {query_str}

                Retrieved Documents:
                {doc_summaries}

                Instructions:
                1. Summarize the relevant points from the documents that match the user's query (the 'compilation').
                2. Identify any differences, missing details, or conflicts between the user's query and what the documents provide.
                3. Output the answer in a single paragraph where you state only the document titles, similarities and the differences found.
            """
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

    # Now wrap that prompt_text into a list of messages
    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="You are an assistant that compares the user query to the provided documents."
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=prompt_text
        ),
    ]

    # Pass messages=list_of_dicts instead of a single string
    response = llm.chat(messages=messages).raw['choices'][0]['text']
    return response
