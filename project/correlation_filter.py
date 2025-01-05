from typing import List

from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
)
from doc_list import DocListResponse



def build_correlation_prompt(query_str: str, retrieved_docs: List[DocListResponse]) -> str:
    """
    Builds a prompt that:
    1. Summarizes the relevant content from each doc (the compilation)
    2. Highlights differences/gaps between the user's query & docs
    """
    doc_summaries = ""
    for i, doc in enumerate(retrieved_docs):
        doc_summaries += (
            f"\nDocument {i} (Title: '{doc.title}', Abstract: '{doc.abstract}', Score: {doc.similarity}):\n"
        )

    prompt = f"""
                System: You are an assistant that compares the user's research query to the provided documents.
                Produce a compilation of key points from the documents and highlight any differences or gaps.
                If the retrieved documents do not have relevant information about the query please only state 
                    'I have not found any relevant documents.' as there is no need to follow the instructions bellow.

                User Query:
                {query_str}

                Retrieved Documents:
                {doc_summaries}

                Instructions:
                1. Summarize the relevant points from the documents that match the user's query (the 'compilation').
                2. Identify any differences, missing details, or conflicts between the user's query and what the documents provide.
                3. Output the answer in a single paragraph where you state only the document titles, similarities and the differences found.
            """
    return prompt


def run_correlation_filter(query_str: str, retrieved_docs: List[DocListResponse], llm) -> ChatResponse:
    """
    1) Build correlation prompt
    2) Call the LLM
    3) Return the structured result
    """

    # Build whatever "prompt_text" you like (the old single-string prompt)
    prompt_text = build_correlation_prompt(query_str, retrieved_docs)

    # Now wrap that prompt_text into a list of messages
    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="You are an assistant that compares the user query to the provided documents."
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=prompt_text
        ),
    ]

    # Pass messages=list_of_dicts instead of a single string
    response = llm.chat(messages=messages)
    return response
