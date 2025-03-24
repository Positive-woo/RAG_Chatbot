from dotenv import load_dotenv
import os
import pickle
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote.document_loaders import HWPLoader

load_dotenv()

# 벡터DB 및 BM25 저장 경로 설정
vector_db_path = os.environ["VECTOR_STORE_PATH"]
faiss_index_path = os.path.join(vector_db_path, "index.faiss")  # FAISS 인덱스 저장 파일
bm25_docs_path = os.path.join(vector_db_path, "bm25_docs.pkl")  # BM25용 문서 저장 파일

if os.path.exists(faiss_index_path):
    print("기존 vector db 사용")
    vector_db = FAISS.load_local(vector_db_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    # BM25용 문서 로드
    if os.path.exists(bm25_docs_path):
        with open(bm25_docs_path, "rb") as f:
            split_texts = pickle.load(f)
    else:
        raise FileNotFoundError("BM25 문서 파일이 없습니다. 벡터DB를 새로 생성해야 합니다.")

else:
    print("벡터 스토어가 없습니다. 새로운 벡터DB 생성")

    # 문서 로드 및 분할
    loader = HWPLoader(os.environ["HWP_PATH"])
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # 원본 텍스트 저장 (BM25 용)
    split_texts = [doc.page_content for doc in splits]
    with open(bm25_docs_path, "wb") as f:
        pickle.dump(split_texts, f)

    # FAISS 벡터 스토어 생성
    vector_db = FAISS.from_documents(splits, OpenAIEmbeddings())
    vector_db.save_local(vector_db_path)

# FAISS 검색기 생성
faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 1})

# BM25 검색기 생성
bm25_retriever = BM25Retriever.from_texts(split_texts)
bm25_retriever.k = 1

# Hybrid 검색기 생성
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)

# 사용자 입력 & 검색 실행
query = input("질문을 입력하세요: ")
results = ensemble_retriever.invoke(query)

# 검색된 문서 출력
for i, doc in enumerate(results):
    print(f"문서 {i+1}:")
    print(doc)
    print("\n")  # 문서 끝에 두 줄 띄우기
