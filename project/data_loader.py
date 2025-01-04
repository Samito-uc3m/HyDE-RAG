import json
import os

import kagglehub
from config import PDF_FILE_PATH
from llama_index.readers.file import PyMuPDFReader


def load_documents(max_docs=500):
    """
    Download the 'arxiv' dataset from Kaggle, parse its JSON lines, and return
    a list of dict objects, each containing 'text' and 'metadata'.
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
            if count >= max_docs:
                break

    return documents
