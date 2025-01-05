from pathlib import Path

MODEL_URL = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"
# MODEL_URL = "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"
PDF_FILE_PATH = "./data/llama2.pdf"
EMBED_MODEL_NAME = "BAAI/bge-small-en"

LLM_TEMPERATURE = 0.1
LLM_MAX_NEW_TOKENS = 384
LLM_CONTEXT_WINDOW = 3900
LLM_MODEL_KWARGS = {"n_gpu_layers": 1}
LLM_VERBOSE = False

CHUNK_SIZE = 128
CHUNK_OVERLAP = 50
NODE_TOP_K = 20
DOCUMENT_TOP_K = 3
QUERY_MODE = "default"

# Languge model
MODELS_PATH = Path('./models/')
FASTTEXT_MODEL = 'lid.176.ftz'
FASTTEXT_LANGUAGES_MAP = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "ar": "Arabic",
    "tr": "Turkish",
}
