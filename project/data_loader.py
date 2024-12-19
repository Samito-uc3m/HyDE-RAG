from llama_index.readers.file import PyMuPDFReader
from config import PDF_FILE_PATH

def load_documents():
    loader = PyMuPDFReader()
    documents = loader.load(file_path=PDF_FILE_PATH)
    return documents
