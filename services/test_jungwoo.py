from dotenv import load_dotenv
import os
import pickle
import shutil
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote.document_loaders import HWPLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


load_dotenv()

# 벡터DB 및 BM25 저장 경로 설정
vector_db_path = os.environ["VECTOR_STORE_PATH"]
txt_path = os.environ["TXT_PATH"]
faiss_index_path = os.path.join(vector_db_path, "index.faiss")  # FAISS 인덱스 저장 파일
bm25_docs_path = os.path.join(vector_db_path, "bm25_docs.pkl")  # BM25용 문서 저장 파일
volume_path = "./volume/vector_db"

def vector_db_maker():
    print("벡터 스토어가 없습니다. 새로운 벡터DB 생성")

    # volume 파일 내부 vector_db 비우기기
    for filename in os.listdir(volume_path):
        file_path = os.path.join(volume_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    # 텍스트 파일 읽기
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    # \n\n 기준으로 자르고, 빈 줄 제거 후 Document로 감싸기
    splits = [
        Document(page_content=chunk.strip())
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]    

    # 원본 텍스트 저장 (BM25 용)
    split_texts = [doc.page_content for doc in splits]
    with open(bm25_docs_path, "wb") as f:
        pickle.dump(split_texts, f)

    # FAISS 벡터 스토어 생성
    vector_db = FAISS.from_documents(splits, OpenAIEmbeddings())
    vector_db.save_local(vector_db_path)
    
    return vector_db, split_texts

if os.path.exists(faiss_index_path):
    # 기존거 사용할지 새로 할지 int input 받기. 만약 새로 사용한다고 하면 else로 넘어가도록
    choice = int(input("기존 벡터 DB가 있습니다. 그대로 사용하시려면 1을 입력해주세요. 새로 생성하시려면 2를 누르세요. : "))
    if choice == 1:
        print("기존 vector db 사용")
        vector_db = FAISS.load_local(vector_db_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

        # BM25용 문서 로드
        if os.path.exists(bm25_docs_path):
            with open(bm25_docs_path, "rb") as f:
                split_texts = pickle.load(f)
        else:
            raise FileNotFoundError("BM25 문서 파일이 없습니다. 벡터DB를 새로 생성해야 합니다.")
    else:
        vector_db, split_texts = vector_db_maker()
else:
    vector_db, split_texts = vector_db_maker()

# FAISS 검색기 생성
faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# BM25 검색기 생성
bm25_retriever = BM25Retriever.from_texts(split_texts)
bm25_retriever.k = 2

# Hybrid 검색기 생성
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)

custom_prompt = PromptTemplate.from_template(
    """
    아래 문맥(context)을 기반으로 사용자 질문에 답변하세요.  
    관련된 정보가 여러 개일 경우 **모두 빠짐없이 정리해서 목록 형태로 제시**하세요.  
    문맥에 해당 정보가 **전혀 없으면**, "해당 정보가 문서에 없습니다."라고 정확히 답하세요.
    말투는 너가 대학 선배님이고 질문자가 신입생이라고 생각하고 친절한 말투로 해줘. 호칭은 '후배님'으로 해줘.

    문맥:
    {context}

    질문:
    {question}

    답변:
    """
)


# LLM 설정
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 파이프라인 구성
rag_chain = (
    custom_prompt
    | llm
    | StrOutputParser()
)

while True:
    # 사용자 입력 & 검색 실행

    print("대화를 종료하려면 exit , q 등을 입력해주세요.;;")

    query = input("질문을 입력하세요: ")
    if query.lower() in ["exit", "q"]:
        print("대화를 종료합니다.")
        break

    results = ensemble_retriever.invoke(query)

    # 검색된 문서 출력
    for i, doc in enumerate(results):
        print(f"문서 {i+1}:")
        print(doc)
        print("\n")  # 문서 끝에 두 줄 띄우기


    # context를 문자열로 변환
    context_text = "\n\n".join(
        [f"[문서{i+1}]\n{doc.page_content}" for i, doc in enumerate(results)]
    )

    # 결과 생성
    answer = rag_chain.invoke({"context": context_text,
                            "question": query})

    filled_prompt = custom_prompt.format(context=context_text, question=query)
    print("\n--- 입력된 프롬프트 ---")
    print(filled_prompt)


    # 답변 출력
    print("\n--- 답변 ---")
    print(answer)