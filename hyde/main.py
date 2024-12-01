import os
import chromadb

client = chromadb.PersistentClient(path="./chromadb/db")

# Example operation
collection = collection = client.get_collection(name="Students")
print("Connected to ChromaDB:", collection.name)

results = collection.query(
    query_texts=["What is the student name?"],
    n_results=2
)
print("Query results",results)
