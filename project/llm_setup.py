"""
Módulo: llm_setup.py

Módulo para la configuración de un modelo de lenguaje grande (LLM) utilizando LlamaCPP.
"""

from llama_index.llms.llama_cpp import LlamaCPP
from config import (
    MODEL_URL,           # URL del modelo a utilizar
    LLM_TEMPERATURE,     # Parámetro de temperatura para el modelo, controla la aleatoriedad
    LLM_MAX_NEW_TOKENS,  # Número máximo de tokens generados en una sola llamada
    LLM_CONTEXT_WINDOW,  # Tamaño de la ventana de contexto del modelo
    LLM_MODEL_KWARGS,    # Argumentos adicionales para el modelo
    LLM_VERBOSE          # Indicador para habilitar o deshabilitar la salida detallada
)

def get_llm():
    """
    Inicializa y devuelve una instancia del modelo de lenguaje LlamaCPP.

    Devuelve:
    --------
    llama_index.llms.llama_cpp.LlamaCPP
        Una instancia configurada del modelo LlamaCPP.
    """
    llm = LlamaCPP(
        model_url=MODEL_URL,
        model_path=None,
        temperature=LLM_TEMPERATURE,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        context_window=LLM_CONTEXT_WINDOW,
        model_kwargs=LLM_MODEL_KWARGS,
        verbose=LLM_VERBOSE,
    )
    return llm
