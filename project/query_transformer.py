from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
)


def build_entry_transformation_prompt(query_str: str) -> str:
    """
    Transforms original query into a better for searching in the database
    """
    prompt = f"""
        Example:
        Input: "I am researching about LLM models and their usage in medicine"
        Output: "LLM models in medicine"

        Task:
        Transform the following user query into a concise phrase that captures the main topic without extra words or phrases.
        Return only the concise phrase without any extra commentary.

        User Query:
        "{query_str}"
        """
    return prompt


def run_query_transformation_filter(query_str: str, llm) -> ChatResponse:
    """
    1) Build correlation prompt
    2) Call the LLM
    3) Return the structured result
    """

    # Build whatever "prompt_text" you like (the old single-string prompt)
    prompt_text = build_entry_transformation_prompt(query_str)

    # Now wrap that prompt_text into a list of messages
    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(    
                "You are a text transformation assistant. Your goal is to read the user query "
                "and respond with a concise phrase that retains only the essential meaning. "
                "Do not include filler words or additional text."
            )
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=prompt_text
        ),
    ]

    # Pass messages=list_of_dicts instead of a single string
    response = llm.chat(messages=messages)
    return response
