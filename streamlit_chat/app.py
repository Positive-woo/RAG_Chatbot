# app.py

import streamlit as st
from chatbot import get_response

st.set_page_config(page_title="Simple Chatbot", layout="centered")
st.title("💬 나만의 챗봇")

# 세션 상태에 메시지 저장
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 메시지 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 받기
prompt = st.chat_input("메시지를 입력해 보세요!")
if prompt:
    # 사용자 메시지 저장 및 출력
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 챗봇 응답 생성 및 저장
    response = get_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
