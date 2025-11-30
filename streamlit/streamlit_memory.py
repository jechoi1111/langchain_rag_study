import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

# Set the OpenAI API key from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"

#Streamlit에서는 @st.cache_resource를 통해 한번 실행한 자원을 리로드 시에 재실행하지 않도록 캐시메모리에 저장할 수 있습니다.
@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path) # PDF 문서 로드
    return loader.load_and_split() # PDF 문서를 여러 청크로 분할

# Create a vector store from the document chunks
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0) # 문서를 여러 청크로 분할

    split_docs = text_splitter.split_documents(_docs)
    vectorstore = Chroma.from_documents(split_docs, GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')) # 문서를 벡터 스토어에 저장
    return vectorstore

# Initialize the LangChain components
@st.cache_resource
def initialize_components(selected_model):
    file_path = r"/Users/choejeong-eun/dev/langchain_rag_study/streamlit/대한민국 헌법.pdf"
    pages = load_and_split_pdf(file_path) # PDF 문서를 여러 청크로 분할
    vectorstore = create_vector_store(pages) # 문서를 벡터 스토어에 저장
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # 벡터 스토어를 검색 가능하게 함 (3개의 유사 문장 출력)

    # Define the contextualize question prompt
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    # prompt template 정의
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt), # 시스템 메시지 추가 (채팅 이력 추가)
            MessagesPlaceholder("history"), # 채팅 이력 추가 (이전 메시지 출력)
            ("human", "{input}"), # 사용자 질문 추가 (사용자 질문 출력)
        ]
    )

    # 질문과 답변 프롬프트 정의
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer perfect. please use imogi with the answer.
    대답은 한국어로 하고, 존댓말을 써줘.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt), # 시스템 메시지 추가 (채팅 이력 추가)
            MessagesPlaceholder("history"), # 채팅 이력 추가 (이전 메시지 출력)
            ("human", "{input}"), # 사용자 질문 추가 (사용자 질문 출력)
        ]
    )

    llm = ChatGoogleGenerativeAI(model=selected_model) # 모델 선택
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt) # 채팅 이력 추가
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt) # 질문과 답변 체인 생성
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain) # 채팅 이력 추가와 질문과 답변 체인 생성
    return rag_chain # RAG 체인 반환

# Streamlit UI
st.header("헌법 Q&A 챗봇 💬 📚")
option = st.selectbox("Select GPT Model", ("gemini-2.5-flash", "gemini-2.5-pro"))
rag_chain = initialize_components(option) # RAG 체인 생성
chat_history = StreamlitChatMessageHistory(key="chat_messages") # 채팅 이력 추가

conversational_rag_chain = RunnableWithMessageHistory( # 채팅 이력 추가
    rag_chain, # RAG 체인 추가
    lambda session_id: chat_history, # 채팅 이력 추가
    input_messages_key="input", # 사용자 질문 추가
    history_messages_key="history", # 채팅 이력 추가
    output_messages_key="answer", # 답변 추가
)


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", 
                                     "content": "헌법에 대해 무엇이든 물어보세요!"}]

for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)


if prompt_message := st.chat_input("Your question"):
    st.chat_message("human").write(prompt_message)
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke(
                {"input": prompt_message},
                config)
            
            answer = response['answer']
            st.write(answer)
            with st.expander("참고 문서 확인"):
                for doc in response['context']:
                    st.markdown(doc.metadata['source'], help=doc.page_content)
