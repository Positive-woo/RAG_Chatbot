from dotenv import load_dotenv
import os

import pickle
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


load_dotenv()

class Config:
    def __init__ (self):
        self.vector_db_path = os.environ["VECTOR_STORE_PATH"]
        self.txt_path = os.environ["TXT_PATH"]
        self.faiss_index_path = os.path.join(self.vector_db_path, "index.faiss")
        self.bm25_docs_path = os.path.join(self.vector_db_path, "bm25_docs.pkl")
        self.volume_path = "./volume/vector_db"
        self.vector_db_init()


    def vector_db_init(self):
        self.vector_db = FAISS.load_local(self.vector_db_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

        # BM25용 문서 로드
        if os.path.exists(self.bm25_docs_path):
            with open(self.bm25_docs_path, "rb") as f:
                self.split_texts = pickle.load(f)
        else:
            raise FileNotFoundError("BM25 문서 파일이 없습니다. 벡터DB를 새로 생성해야 합니다.")
