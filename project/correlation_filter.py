from typing import List
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)
from doc_list import DocListResponse

def summarize_document(doc: DocListResponse, output_language: str, llm) -> str:
    """
    Summarizes a single document using a language model, ensuring only the summary is returned.
    
    Parameters:
    -----------
    doc : DocListResponse
        A single document retrieved for the query.
    output_language : str
        The language in which the summary should be generated.
    llm : object
        The language model for processing the prompt.
    
    Returns:
    --------
    str
        A brief 1-2 sentence summary of the document.
    """

    system_content = (
        "You are a specialized AI assistant focused on summarizing academic documents. "
        "Your task is to provide a concise, accurate summary of the provided document with a maximum of 30 words. "
        "Output only the summary text and nothing else."
    )

    prompt_text = (
        f"Title: {doc.title}\n"
        f"Abstract: {doc.abstract}\n\n"
    )

    user_instructions = (
        "Instructions:\n"
        f"1. Summarize the document in {output_language} with a parragraph of 30 words maximum.\n"
        "2. Focus on clarity, conciseness, and accuracy."
    )

    # Construct the message structure for the LLM
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_content),
        ChatMessage(role=MessageRole.USER, content=prompt_text),
        ChatMessage(role=MessageRole.SYSTEM, content=user_instructions)
    ]

    # Call the LLM to generate the summary
    response = llm.chat(messages=messages)
    summary = response.raw.choices[0].message.content.strip()
    
    return summary

def compare_summaries_with_query(query_str: str, summaries: List[str], output_language: str, llm) -> str:
    """
    Asynchronously compares the document summaries with the user query.

    Parameters:
    -----------
    query_str : str
        The user's query.
    summaries : List[str]
        Summaries of the retrieved documents.
    output_language : str
        The output language for the response.
    llm : object
        The language model for processing the prompt.

    Returns:
    --------
    str
        The language model's response comparing the summaries with the query.
    """
    doc_descriptions = "\n".join(summaries)

    system_content = (
        "You are a specialized AI assistant focused on research analysis and comparison. Your task is to compare a user's research "
        f"query with relevant papers, highlight similarities and note any gaps or differences. Please provide a clear and concise response in {output_language}."
    )

    prompt_text = (
        f"User Query:\n"
        f"\"{query_str}\"\n\n"
        f"Retrieved Documents for Comparison:\n"
        f"\"{doc_descriptions}\"\n\n"
        "Focus on clarity, conciseness, and accuracy."
    )

    user_instructions = (
        "Task:\n"
        f"1. Analyze the query and compare it with the provided documents in a single, concise paragraph in {output_language}.\n"
        "2. Clearly state whether the documents address the same topic, a related topic, or an entirely different topic.\n"
        "3. If no relevant documents are found, respond: \"I have not found relevant documents about the topic you are researching.\" Suggest trying another database if necessary."
    )

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_content),
        ChatMessage(role=MessageRole.USER, content=prompt_text),
        ChatMessage(role=MessageRole.SYSTEM, content=user_instructions)
    ]

    response = llm.chat(messages=messages).raw.choices[0].message.content.strip()
    return response

def run_correlation_filter(query_str: str, output_language: str, retrieved_docs: List[DocListResponse], llm) -> str:
    """
    Workflow of summarizing documents and comparing with the query.
    
    Parameters:
    -----------
    query_str : str
        The user's query.
    output_language : str
        The output language for the response.
    retrieved_docs : List[DocListResponse]
        A list of documents retrieved for the query.
    llm : object
        The language model for processing the prompt.
    
    Returns:
    --------
    str
        The final response comparing the query with document summaries.
    """

    # Step 1: Summarize documents asynchronously
    summaries = [summarize_document(doc, output_language, llm) for doc in retrieved_docs]
    
    # Step 2: Compare summaries with the query asynchronously
    response = compare_summaries_with_query(query_str, summaries, output_language, llm)

    # Create final response string
    output_string = f"{response}"
    for i, (doc, summary) in enumerate(zip(retrieved_docs, summaries)):
        output_string += f"{i+1}: {doc.title} | {int(doc.similarity*100)}%: {summary}\n\n"
    
    return output_string
