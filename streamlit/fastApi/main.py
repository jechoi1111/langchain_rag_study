import os
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException # FastAPI 설정
from fastapi.staticfiles import StaticFiles # 정적 파일 제공
from fastapi.responses import FileResponse # 정적 파일 제공
from fastapi.middleware.cors import CORSMiddleware # CORS 설정

import bs4 # BeautifulSoup 설정
import traceback # 오류 추적
from pydantic import BaseModel # Pydantic 모델 설정

from langchain_classic.chains import create_retrieval_chain # RAG 체인 설정
from langchain_classic.chains.combine_documents import create_stuff_documents_chain # 문서 처리 체인 설정
from langchain_community.vectorstores import FAISS # FAISS 벡터 스토어 설정
from langchain_community.document_loaders import WebBaseLoader # WebBaseLoader 설정
from langchain_core.prompts import ChatPromptTemplate # ChatPromptTemplate 설정
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings # Google Generative AI 설정
from langchain_text_splitters import RecursiveCharacterTextSplitter # RecursiveCharacterTextSplitter 설정

# Load environment variables
load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 모든 도메인 허용
    allow_credentials=True, # 인증 정보 허용
    allow_methods=["*"], # 모든 메서드 허용
    allow_headers=["*"], # 모든 헤더 허용
)

# 정적 파일 제공 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

# Google API 키는 .env 파일에서 관리합니다.
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

embeddings = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001', google_api_key=google_api_key)
llm = ChatGoogleGenerativeAI(google_api_key=google_api_key, model = "gemini-2.5-flash")

class URLInput(BaseModel):
    url: str

class QueryInput(BaseModel):
    query: str

# 전역 변수로 RAG 체인을 관리합니다.
rag_chain = None

# home page
@app.get("/")
async def root():
    return FileResponse('static/index.html')

@app.post("/process_url")
async def process_url(url_input: URLInput):
    global rag_chain
    try:
        loader = WebBaseLoader(
            web_paths=(url_input.url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("newsct_article _article_body",)
                )
            ),
        )
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # 문서를 여러 청크로 분할
        splits = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings) # 문서를 벡터 스토어에 저장
        retriever = vectorstore.as_retriever() # 벡터 스토어를 검색 가능하게 함

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return {"message": "URL processed successfully"}
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in process_url: {error_trace}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query(query_input: QueryInput):
    global rag_chain
    if not rag_chain:
        raise HTTPException(status_code=400, detail="Please process a URL first")
    try:
        result = rag_chain.invoke({"input": query_input.query})
        return {"answer": result["answer"]}
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in query: {error_trace}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__": # 메인 함수 실행
    import uvicorn # uvicorn 실행
    uvicorn.run(app, host="0.0.0.0", port=8000) # uvicorn 실행 (0.0.0.0:8000 포트에서 실행)
