import os

os.environ["OLLAMA_HOST"] = "http://kumo01:11434/"

import ollama

import chromadb

client = chromadb.PersistentClient(path="./chromadb/db")

# Example operation
collection = client.get_collection(name="arxiv")
print("Connected to ChromaDB:", collection.name)

text_title = "Renormalized quasiparticles in antiferromagnetic states of the Hubbard model"
results = collection.query(query_texts=[text_title], n_results=2)
print("Normal Query results", results["distances"])

# Prepare system message for HYDE
system_message = {
    "role": "system",
    "content": (
        "Generate an arxiv type abstract for this theme. "
        "Do not add specific information to the abstract, "
        "make it as general as possible. I want you only to return "
        "the plain text, I dont want Title: or Abstract: in the final text."
    ),
}

user_prompt = "I am researching renormalized quasiparticles in antiferromagnetic states of the Hubbard model"

# Store all HYDE-generated abstracts and their results
all_results = []

for i in range(5):
    # Generate a HYDE abstract
    response = ollama.chat(model="llama3.2", messages=[system_message, {"role": "user", "content": user_prompt}])
    abstract = response["message"]["content"].strip()
    print(f"\nGenerated abstract {i+1}: {abstract}")

    # Query the DB with this abstract
    hyde_query_result = collection.query(query_texts=[abstract], n_results=3)
    print(hyde_query_result)

    # Store the results along with distances and documents
    for dist, doc_id, doc_text in zip(
        hyde_query_result["distances"][0], hyde_query_result["ids"][0], hyde_query_result["documents"][0]
    ):
        all_results.append((dist, doc_id, doc_text, i + 1))

# Sort results by distance
all_results_sorted = sorted(all_results, key=lambda x: x[0])

# Ensure no repeated documents in the final top 3
unique_docs = []
seen_doc_ids = set()
for dist, doc_id, doc_text, idx in all_results_sorted:
    if doc_id not in seen_doc_ids:
        unique_docs.append((dist, doc_id, doc_text, idx))
        seen_doc_ids.add(doc_id)
    if len(unique_docs) == 3:
        break

print("\n=== Top 3 Unique Overall Results ===")
for rank, (dist, doc_id, doc_text, idx) in enumerate(unique_docs, start=1):
    print(f"Rank {rank}:")
    print(f"  HYDE Abstract Index: {idx}")
    print(f"  Distance: {dist}")
    print(f"  Document ID: {doc_id}")
    print(f"  Document Text: {doc_text[:100]}...")

results_str = ""
for rank, (dist, doc_id, doc_title, idx) in enumerate(unique_docs, start=1):
    results_str += f"  Document Title: {doc_title}\n\n"

# results_str = ""

system_message = {
    "role": "system",
    "content": (
        "You are responsible for relating documents of previous investigations "
        "of the state of the art. You are assisting researchers to provide articles of their interest. "
        'Between """ """ you will have a list of docuements that are related to the question they user is asking. '
        "Your task is to answer mentioning those title names. "
        "Your answer has to be in the language that the user ask. Dont modify the titles. "
        "I want you only to return "
        "the plain text, I dont want Title: or Abstract: in the final text."
        "The following are examples of the interaction:"
        "Example 1"
        "Question: "
        "I am researching implementations of federated neural topic models."
        "Your Response:"
        "I have found several relevant articles, including “Federated topic modeling” "
        "and “Federated nonnegative matrix factorization for short texts topic modeling with mutual information.” "
        "These works focus on the implementation of federated approaches to Bayesian topic models, such as LDA or NMF, "
        "but none of them provide an implementation based on neural topic models."
        f'"""{results_str}"""'
    ),
}

response = ollama.chat(model="llama3.2", messages=[system_message, {"role": "user", "content": user_prompt}])

print(response["message"]["content"].strip())
