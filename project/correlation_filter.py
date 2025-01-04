from typing import List

from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore


def build_correlation_prompt(query_str: str, nodes_with_scores: List[NodeWithScore]) -> str:
    """
    Builds a prompt that:
    1. Summarizes the relevant content from each doc (the compilation)
    2. Highlights differences/gaps between the user's query & docs
    """
    doc_summaries = ""
    for i, nws in enumerate(nodes_with_scores):
        metadata = nws.node.metadata or {}
        source_id = metadata.get("source", f"doc_{i}")
        title = metadata.get("title", f"Document {i+1}")
        snippet = nws.node.text[:300]  # short snippet to avoid huge prompt

        doc_summaries += (
            f"\nDocument {i+1} (ID: {source_id}, Title: '{title}', Score: {nws.score:.3f}):\n" f"{snippet}\n"
        )

    prompt = f"""
                System: You are an assistant that compares the user's research query to the provided documents.
                Produce a compilation of key points from the documents and highlight any differences or gaps.

                User Query:
                {query_str}

                Retrieved Documents:
                {doc_summaries}

                Instructions:
                1. Summarize the relevant points from the documents that match the user's query (the 'compilation').
                2. Identify any differences, missing details, or conflicts between the user's query and what the documents provide.
                3. Output your answer in two sections: 'Compilation of Relevant Docs' and 'Differences / Gaps'.
            """
    return prompt


def run_correlation_filter(query_str, retriever, llm, top_k=5):
    """
    1) Retrieve docs
    2) Build correlation prompt
    3) Call the LLM
    4) Return the structured result
    """
    query_bundle = QueryBundle(query_str)
    nodes_with_scores = retriever._retrieve(query_bundle)
    nodes_with_scores.sort(key=lambda x: x.score if x.score else 0, reverse=True)
    top_docs = nodes_with_scores[:top_k]

    # Build whatever "prompt_text" you like (the old single-string prompt)
    prompt_text = build_correlation_prompt(query_str, top_docs)

    # Now wrap that prompt_text into a list of messages
    messages = [
        {"role": "system", "content": "You are an assistant that compares the user query to the provided documents."},
        {"role": "user", "content": prompt_text},
    ]

    # Pass messages=list_of_dicts instead of a single string
    response = llm.chat(messages=messages)
    return response
