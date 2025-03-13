from dotenv import load_dotenv
import os
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote.document_loaders import HWPLoader

load_dotenv()

# 문서 로드
loader = HWPLoader("C:/Users/wjddn/Documents/taejung/RAG_Chatbot/volume/test.hwp")
docs = loader.load()

# 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
splits = text_splitter.split_documents(docs)


# 임베딩 & 벡터스토어 생성(create)
faiss_store = FAISS.from_documents(splits, OpenAIEmbeddings())
faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 2})

# 검색
split_texts = [doc.page_content for doc in splits]
bm25_retriever = BM25Retriever.from_texts(split_texts)
bm25_retriever.k = 2

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)

# 프롬프트 생성성
query = input("질문을 입력하세요: ")
ensemble_retriever.invoke(query)


# TODO
# 로드 한번 하면 vector store 에 저장해놓고 그걸 쓰는 방식으로 변경
# faiss_store.save_local("faiss_index")
# faiss_store = FAISS.load_local("faiss_index", OpenAIEmbeddings())
