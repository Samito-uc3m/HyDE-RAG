"""
Módulo: llm_setup.py

Módulo para la configuración de un modelo de lenguaje grande (LLM) utilizando OpenAI.
"""

from llama_index.llms.openai import OpenAI
from config import settings

def get_llm():
    """
    Inicializa y devuelve una instancia del modelo de lenguaje OpenAI.

    Devuelve:
    --------
    llama_index.llms.llama_cpp.OpenAI
        Una instancia configurada del modelo OpenAI.
    """
    llm = OpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY
    )
    return llm
