MODEL_URL = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"
# MODEL_URL = "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"
PDF_FILE_PATH = "./data/llama2.pdf"
EMBED_MODEL_NAME = "BAAI/bge-small-en"

LLM_TEMPERATURE = 0.1
LLM_MAX_NEW_TOKENS = 256
LLM_CONTEXT_WINDOW = 3900
LLM_MODEL_KWARGS = {"n_gpu_layers": 1}
LLM_VERBOSE = False

CHUNK_SIZE = 1024
SIMILARITY_TOP_K = 2
QUERY_MODE = "default"
