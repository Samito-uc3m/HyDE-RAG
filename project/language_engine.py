"""
Módulo: language_engine.py

Módulo para la detección de idiomas utilizando un modelo preentrenado de FastText.
Incluye funciones para cargar el modelo y detectar el idioma de un texto dado.
"""

import fasttext as ft
from config import FASTTEXT_LANGUAGES_MAP, MODELS_PATH

def load_language_detection_model(fasttext_model: str) -> ft.FastText:
    """
    Carga un modelo de FastText para la detección de idiomas.

    Este método utiliza la ruta especificada para cargar el modelo preentrenado
    de FastText.
    
    Parámetros:
    -----------
    fasttext_model : str
        El nombre del modelo de FastText a cargar.

    Devuelve:
    --------
    fasttext.FastText
        Una instancia del modelo de FastText cargado.
    """
    model = ft.load_model(str(MODELS_PATH / fasttext_model))
    return model


def detect_language(query: str, model: ft.FastText) -> str:
    """
    Detecta el idioma de un texto dado utilizando un modelo de FastText.

    Este método toma un texto de entrada y utiliza un modelo de FastText para
    predecir el idioma principal.

    Parámetros:
    -----------
    query : str
        El texto cuya lengua se desea detectar.
    model : fasttext.FastText
        El modelo de FastText ya cargado.

    Devuelve:
    --------
    str
        El código del idioma detectado, como 'es' (español) o 'en' (inglés).
    """
    prediction = model.predict(query, k=1)
    return FASTTEXT_LANGUAGES_MAP[prediction[0][0].replace('__label__', '')]
