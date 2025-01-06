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

    # Prompt with instructions
    system_content = (
        "You are a query simplification assistant for improving search efficiency. Your task is to transform user queries "
        "into concise phrases that capture the essential meaning. Exclude unnecessary words or details."
    )

    user_task = (
        "Transform the provided query into a concise search phrase. Return only the simplified phrase, in English, "
        "without additional commentary."
    )

    prompt_text = build_entry_transformation_prompt(query_str)

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
            content=user_task
        )
    ]

    # Pass messages=list_of_dicts instead of a single string
    response = llm.chat(messages=messages).raw['choices'][0]['text']
    return response
