
from llama_index.llms.llama_cpp import LlamaCPP
from config import (
    MODEL_URL,
    LLM_TEMPERATURE,
    LLM_MAX_NEW_TOKENS,
    LLM_CONTEXT_WINDOW,
    LLM_MODEL_KWARGS,
    LLM_VERBOSE
)

def get_llm():
    llm = LlamaCPP(
        model_url=MODEL_URL,
        model_path=None,
        temperature=LLM_TEMPERATURE,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        context_window=LLM_CONTEXT_WINDOW,
        model_kwargs=LLM_MODEL_KWARGS,
        verbose=LLM_VERBOSE,
    )
    return llm
