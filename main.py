import streamlit as st
from services.llm import rag_chain_invoke

st.title("💬 연세 대학 선배 챗봇")

def render_bubble(role, content, emoji):
    if role == "user":
        align = "row-reverse"
        bubble_color = "#dcf8c6"
        tail_style = "position: absolute; right: -10px; top: 10px; width: 0; height: 0; border-top: 10px solid transparent; border-bottom: 10px solid transparent; border-left: 10px solid #dcf8c6;"
    else:
        align = "row"
        bubble_color = "#f1f0f0"
        tail_style = "position: absolute; left: -10px; top: 10px; width: 0; height: 0; border-top: 10px solid transparent; border-bottom: 10px solid transparent; border-right: 10px solid #f1f0f0;"

    st.markdown(f"""
    <div style="display: flex; flex-direction: {align}; align-items: flex-start; margin: 10px 0;">
        <div style="width: 30px; height: 30px; font-size: 20px; text-align: center; line-height: 30px; border-radius: 50%; background-color: #eeeeee; margin: 5px;">
            {emoji}
        </div>
        <div style="position: relative; background-color: {bubble_color}; padding: 10px 15px; border-radius: 15px; max-width: 80%;">
            <div style="{tail_style}"></div>
            <div>{content}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 채팅 표시
for msg in st.session_state.chat_history:
    render_bubble(msg["role"], msg["content"])

# 사용자 입력
user_input = st.chat_input("메시지를 입력하세요")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    render_bubble("user", user_input, "🧑🏻‍💻")

    reply = rag_chain_invoke(user_input)
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    render_bubble("assistant", reply, "💁🏻")


# # 채팅 메시지 표시 (최근 메시지가 아래로 오게)
# for msg in st.session_state.chat_history:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # 사용자 입력창 (하단 고정)
# user_input = st.chat_input("메시지를 입력하세요")

# if user_input:
#     # 사용자 메시지 출력
#     st.session_state.chat_history.append({"role": "user", "content": user_input})
#     with st.chat_message("user", avatar = "🧑🏻‍💻"):
#         st.markdown(user_input)

#     reply = rag_chain_invoke(user_input)
#     st.session_state.chat_history.append({"role": "assistant", "content": reply})
#     with st.chat_message("assistant", avatar = "💁🏻"):
#         st.markdown(reply)