from typing import List

from doc_list import DocListResponse
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole


def build_response_prompt(query_str: str, retrieved_docs: List[DocListResponse]) -> str:
    """
    Construye un prompt:

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
        f"Title: '{doc.title}' | Similarity Score: {doc.similarity}\nAbstract: {doc.abstract}"
        for doc in retrieved_docs
    )

    prompt = (
        "User Query:\n"
        f'"{query_str}"\n\n'
        "Retrieved Documents for Comparison:\n"
        f"{doc_descriptions}\n\n"
    )

    return prompt


def run_response_maker(
    query_str: str, output_language: str, retrieved_docs: List[DocListResponse], llm
) -> ChatResponse:
    """
    Crea la respuesta entre la consulta del usuario y los documentos recuperados.

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

    # 1. Build the prompt text, which includes indexes for each document
    prompt_text = build_response_prompt(query_str, retrieved_docs)

    # 2. System instructions describing the overall role and format
    system_content = (
        "You are a specialized AI assistant focused on research analysis and comparison. "
        "Your task is to compare the user query with the retrieved papers. "
        "If at least one document addresses the query, you must:\n"
        " - Provide a single concise paragraph summarizing similarities or differences.\n"
        " - Then list each relevant document in the following format:\n"
        "    Index. Title (Similarity): Brief summary\n\n"
        "If none of the documents address the user's query, you must respond with a short message stating that "
        "you have not found relevant documents about the topic, without searching any other source. "
        f"Please respond in {output_language}, and do not exceed one paragraph for the summary."
    )

    # 3. Additional user-facing instructions to reinforce the exact format
    user_instructions = (
        "Instructions:\n"
        "1. If there are relevant documents, write a short paragraph (max one paragraph) comparing the query and the docs.\n"
        "2. After that paragraph, list each relevant document in this structure:\n"
        "   <document_index>. <document_title> (<document_similarity>): <short_summary_from_abstract_or_document>\n\n"
        "3. If no document is relevant, respond with:\n"
        "   'I have not found relevant documents about the topic you are researching.'\n"
        "   (in the corresponding language, with no mention of other sources).\n"
        "4. No extra paragraphs beyond the single summary paragraph. Then the list or message if no docs are relevant."
    )

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_content),
        ChatMessage(role=MessageRole.USER, content=prompt_text),
        ChatMessage(role=MessageRole.SYSTEM, content=user_instructions),
    ]

    # 4. Call the LLM with the structured messages
    response_text = llm.chat(messages=messages).raw.choices[0].message.content.strip()
    return response_text
