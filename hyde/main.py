import os
import chromadb

client = chromadb.PersistentClient(path="./chromadb/db")

# Example operation
collection = collection = client.get_collection(name="arxiv")
print("Connected to ChromaDB:", collection.name)

results = collection.query(
    query_texts=["Renormalized quasiparticles in antiferromagnetic states of the Hubbard\n  model"],
    n_results=1
)
print("Query results",results)
