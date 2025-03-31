from config.settings import Config
from langchain.retrievers import BM25Retriever, EnsembleRetriever

config=Config()

# FAISS 검색기 생성
faiss_retriever = config.vector_db.as_retriever(search_kwargs={"k": 2})

# BM25 검색기 생성
bm25_retriever = BM25Retriever.from_texts(config.split_texts)
bm25_retriever.k = 2

# Hybrid 검색기 생성
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)
