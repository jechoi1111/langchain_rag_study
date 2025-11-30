import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY" 

st.title("Chatbot")

# 고정 하고 싶은 값이 있을때 세션 스테이트 사용
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "무엇을 도와드릴까요?"
            }
    ]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature = 1)

if prompt := st.chat_input(): # 메시지를 입력하면 prompt에 저장
    st.session_state.messages.append({"role": "user", "content": prompt}) # 사용자 메시지 추가
    st.chat_message("user").write(prompt)
    response = chat.invoke(prompt)
    message = response.content

    st.session_state.messages.append({"role": "assistant", "content": message}) # 챗봇 메시지 추가
    st.chat_message("assistant").write(message)