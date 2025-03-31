import streamlit as st
from services.llm import rag_chain_invoke

st.title("💬 연세 대학 선배 챗봇")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 채팅 메시지 표시 (최근 메시지가 아래로 오게)
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력창 (하단 고정)
user_input = st.chat_input("메시지를 입력하세요")

if user_input:
    # 사용자 메시지 출력
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 챗봇은 항상 "헬로"라고 답함
    reply = rag_chain_invoke(user_input)
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)