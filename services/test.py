from dotenv import load_dotenv
import os
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# 환경 변수 불러오기 -> 추후에 setting으로 환경변수 로드하는거 묶어서 class화
# os.environ["langchain_project"]
# os.environ["openai_api_key"]


# 문서 로드
# hwp 로드하는거 확인해서 해야함

# 문서 분할
# recursive split 할듯

# 임베딩 & 벡터스토어 생성성
# faiss에 저장하는 과정. 함수로 만들어서 저장하거나 이미 저장된 것이 있다면 로드하도록