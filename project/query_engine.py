from llama_index.core.query_engine import RetrieverQueryEngine

def create_query_engine(retriever, llm):
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
    return query_engine
