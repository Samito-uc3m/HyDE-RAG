"""
Módulo: embedding_setup.py

Módulo para configurar y devolver un modelo de embeddings basado en HuggingFace.
"""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import EMBED_MODEL_NAME

def get_embedding_model():
    """
    Utiliza el nombre del modelo especificado en la configuración (`EMBED_MODEL_NAME`)
    para crear una instancia del modelo de embeddings.

    Devuelve:
    --------
    HuggingFaceEmbedding
        Una instancia del modelo de embeddings configurado.
    """
    return HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
