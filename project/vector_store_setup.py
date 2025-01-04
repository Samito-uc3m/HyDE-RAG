from config import CHUNK_SIZE
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb


def create_vector_store():
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("quickstart")
    return ChromaVectorStore(chroma_collection=chroma_collection)


def chunk_documents(documents):
    text_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
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
    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(text=text_chunk)
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc["metadata"]
        nodes.append(node)
    return nodes


def embed_and_add_nodes(nodes, embed_model, vector_store):
    for node in nodes:
        node_embedding = embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
        node.embedding = node_embedding
    vector_store.add(nodes)
