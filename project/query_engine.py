"""
Módulo: query_engine.py

Módulo para la creación de un motor de consultas utilizando un recuperador y un modelo de lenguaje.
"""

from llama_index.core.query_engine import RetrieverQueryEngine

def create_query_engine(retriever, llm):
    """
    Inicializa y devuelve una instancia del motor de consultas que permite
    realizar búsquedas y responder preguntas basadas en un conjunto de datos.

    Parámetros:
    -----------
    retriever : object
        Una instancia de un recuperador que facilita la recuperación de documentos
        relevantes para las consultas.
    llm : object
        Una instancia de un modelo de lenguaje que se utiliza para generar
        respuestas basadas en los documentos recuperados.

    Devuelve:
    --------
    RetrieverQueryEngine
        Una instancia configurada del motor de consultas.
    """
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
    return query_engine
