"""
Módulo: data_loader.py

Módulo para cargar documentos desde el conjunto de datos 'arxiv' disponible en Kaggle.
Permite descargar el dataset, procesar su contenido y devolver una lista de documentos
con su texto y metadatos relevantes.
"""

import json
import os

import kagglehub
from config import PDF_FILE_PATH
from llama_index.readers.file import PyMuPDFReader


def load_documents(max_docs = None):
    """
    Descarga y procesa el conjunto de datos 'arxiv' desde Kaggle.

    Este método descarga el dataset, extrae sus líneas en formato JSON, y procesa cada línea
    para crear una lista de diccionarios. Cada diccionario incluye el texto combinado
    (título y abstract) y los metadatos asociados.

    Parámetros:
    -----------
    max_docs : int, opcional
        Número máximo de documentos a procesar (por defecto es None).

    Devuelve:
    --------
    list
        Una lista de diccionarios, cada uno con las claves:
        - 'text': Título y resumen combinados.
        - 'metadata': Metadatos relevantes como 'source', 'title', y 'abstract'.
    """
    
    # Download the dataset to a local folder
    data_folder = kagglehub.dataset_download("Cornell-University/arxiv")

    # The dataset includes a large JSON file: 'arxiv-metadata-oai-snapshot.json'
    file_path = os.path.join(data_folder, "arxiv-metadata-oai-snapshot.json")

    documents = []
    count = 0

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            content = json.loads(line.strip())

            # Combine title + abstract as the text
            doc_text = content["title"] + "\n\n" + content["abstract"]

            # Keep relevant metadata
            doc_metadata = {"source": content["id"], "title": content["title"], "abstract": content["abstract"]}

            # Store everything in a dict
            documents.append({"text": doc_text, "metadata": doc_metadata})
            count += 1

            # Stop after 'max_docs' documents
            if max_docs is not None and count >= max_docs:
                break

    return documents
