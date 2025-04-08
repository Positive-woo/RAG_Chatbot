from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from services.retriever import ensemble_retriever

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

def rag_chain_invoke(query):
    results = ensemble_retriever.invoke(query)

    # context를 문자열로 변환
    context_text = "\n\n".join(
        [f"[문서{i+1}]\n{doc.page_content}" for i, doc in enumerate(results)]
    )

    # 결과 생성
    answer = rag_chain.invoke({"context": context_text, "question": query})

    return answer