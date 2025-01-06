"""
Módulo: vector_store_setup.py

Módulo para configurar un almacén vectorial utilizando ChromaVectorStore.
Incluye funciones para dividir documentos en fragmentos de texto, crear nodos
con metadatos, y generar incrustaciones para su almacenamiento.
"""


from typing import Tuple
from config import CHUNK_SIZE, CHUNK_OVERLAP, DATABASE_PATH
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb
from chromadb.api.models.Collection import Collection



def create_vector_store() -> Tuple[Collection, ChromaVectorStore]:
    """
    Crea e inicializa un almacén vectorial con Chroma.

    Esta función utiliza un cliente efímero de Chroma para crear o recuperar
    una colección de datos denominada "quickstart" y devuelve un almacén
    vectorial asociado.

    Devuelve:
    --------
    ChromaVectorStore
        Una instancia del almacén vectorial de Chroma.
    """
    chroma_client = chromadb.PersistentClient(path=str(DATABASE_PATH))
    chroma_collection = chroma_client.get_or_create_collection("quickstart")
    return chroma_collection, ChromaVectorStore(chroma_collection=chroma_collection)


def chunk_documents(documents):
    """
    Divide documentos en fragmentos de texto utilizando un separador de oraciones.

    Los documentos se dividen en fragmentos basados en un tamaño de fragmento
    definido y un solapamiento opcional para mejorar la segmentación del texto.

    Parámetros:
    -----------
    documents : list
        Lista de documentos, cada uno representado como un diccionario con una
        clave "text" para el contenido.

    Devuelve:
    --------
    tuple
        Una tupla que contiene una lista de fragmentos de texto y una lista de
        índices de documento asociados.
    """
    text_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        raw_text = doc["text"]
        cur_text_chunks = text_parser.split_text(raw_text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))
    return text_chunks, doc_idxs


def create_nodes(documents, text_chunks, doc_idxs):
    """
    Crea nodos de texto a partir de fragmentos de texto y asocia metadatos.

    Parámetros:
    -----------
    documents : list
        Lista de documentos originales.
    text_chunks : list
        Lista de fragmentos de texto generados por la división de documentos.
    doc_idxs : list
        Lista de índices que vinculan fragmentos con sus documentos originales.

    Devuelve:
    --------
    list
        Lista de nodos de texto con metadatos asociados.
    """
    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(text=text_chunk)
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc["metadata"]
        nodes.append(node)
    return nodes


def embed_and_add_nodes(nodes, embed_model, vector_store, batch_size=100):
    """
    Genera embeddings para los nodos de texto y los agrega al almacén vectorial en lotes.

    Parámetros:
    -----------
    nodes : list
        Lista de nodos de texto para incrustar.
    embed_model : object
        Modelo utilizado para generar incrustaciones de texto. Debe poder
        manejar listas de textos en una sola llamada para usar el modo batch.
    vector_store : ChromaVectorStore
        Instancia del almacén vectorial donde se almacenarán los nodos incrustados.
    batch_size : int
        Tamaño del lote (batch) para procesar las incrustaciones. Por defecto, 1000.
    """

    nodes_to_save = []
    for node in nodes:
        node_embedding = embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
        node.embedding = node_embedding
        nodes_to_save.append(node)

        if len(nodes_to_save) >= batch_size:
            vector_store.add(nodes_to_save)
            nodes_to_save = []
        
    vector_store.add(nodes_to_save)
