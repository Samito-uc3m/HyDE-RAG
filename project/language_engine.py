"""
Modulo para deteccion de idiomas
"""
import fasttext as ft
import os
import urllib.request

# Direccion del modelo de fasttext para deteccion de idiomas
models_path = '../models/'
fasttext_model_path = 'lid.176.ftz'

# Carga modelo de fasttext 
def load_language_detection_model():
    model = ft.load_model(models_path + fasttext_model_path)
    return model

# Funcion para detectar el idioma de un texto
# Entrada: query (str) - Texto a detectar
#          model (fasttext model) - Modelo de fasttext
# Salida: 'es' o 'en' - Idioma detectado 
def detect_language(query, model):
    prediction = model.predict(query, k=1)
    return prediction[0][0].replace('__label__', '')
