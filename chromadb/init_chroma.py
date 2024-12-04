import chromadb
import kagglehub
import os
import json  # Adjust depending on dataset format

# Initialize persistent ChromaDB client
client = chromadb.PersistentClient(path="./chromadb/db")
collection = client.get_or_create_collection(name="arxiv")  # Avoid duplication

# Download the latest version of the dataset
data_folder = kagglehub.dataset_download("Cornell-University/arxiv")

# Initialize lists for batch processing
documents, metadatas, ids = [], [], []

# File path to the large JSON file
file_path = os.path.join(data_folder, "arxiv-metadata-oai-snapshot.json")

# Initialize the counter for files processed
file_count = 0

# Open and read the file content line by line
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Parse each line as a separate JSON object
        content = json.loads(line.strip())  # Load each JSON object
        
        # Extract data (title and abstract as an example)
        documents.append(content["title"] + "\n\n" + content["abstract"])
        metadatas.append({"source": content["id"], "title": content["title"], "abstract":content["abstract"]})
        ids.append(f"doc_{file_count}")  # Unique IDs for each documents
        file_count += 1

        # Batch size: Add to ChromaDB in chunks (500 files at a time for efficiency)
        if len(documents) >= 500:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            documents, metadatas, ids = [], [], []  # Reset lists
            print(f"Added {file_count} files to the collection.")
            break

# Add any remaining documents after the loop
if documents:
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Added final batch of {len(documents)} files.")

print(f"Successfully added {file_count} documents to the 'arxiv' collection.")

