"""
Módulo: query_transformer.py

Módulo para la transformación de querys del usuario en frases más concisas y útiles para
búsquedas en una base de datos.
"""

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
)


def build_entry_transformation_prompt(query_str: str) -> str:
    """
    Transforma la query original en una mejor para la búsqueda en la base de datos.
    Construye una plantilla de prompt que orienta al modelo a generar una versión 
    más breve y enfocada.

    Parámetros:
    -----------
    query_str : str
        La consulta original proporcionada por el usuario.

    Devuelve:
    --------
    str
        Un mensaje de prompt para el modelo que describe la tarea de transformación.
    """
    prompt = f"""
        Example:
        Input: "I am researching about LLM models and their usage in medicine"
        Output: "LLM models in medicine"

        Task:
        Transform the following user query into a concise phrase that captures the main topic without extra words or phrases.
        Return only the concise phrase without any extra commentary. Please answer in a sentence and in English.

        User Query:
        "{query_str}"
        """
    return prompt


def run_query_transformation_filter(query_str: str, llm) -> ChatResponse:
    """
    Ejecuta la transformación de la query usando un modelo de lenguaje.

    Parámetros:
    -----------
    query_str : str
        La consulta original proporcionada por el usuario.
    llm : object
        Una instancia del modelo de lenguaje utilizado para generar la respuesta.

    Devuelve:
    --------
    ChatResponse
        La respuesta generada por el modelo en forma de un objeto estructurado.
    """

    # Build the prompt text for the model
    prompt_text = build_entry_transformation_prompt(query_str)

    # Now wrap that prompt_text into a list of messages
    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(    
                "You are a text transformation assistant. Your goal is to read the user query "
                "and respond with a concise phrase that retains only the essential meaning. "
                "Do not include filler words or additional text."
            )
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=prompt_text
        ),
    ]

    # Pass messages=list_of_dicts instead of a single string
    response = llm.chat(messages=messages)
    return response
