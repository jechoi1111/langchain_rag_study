# RAG 구성 요소 실습 - Embeddings

## Embeddings란?

- 텍스트 데이터를 수치 데이터로 변환하는 역할

## 개념

- Embedding 과정에서 문장의 의미, 주제, 감정 등 다양한 정보를 담기 위해 고차원의 Vector로 변환

## 원리

- 대부분의 경우 대용량의 말뭉치를 통해 사전학습된 모델을 통해 쉽게 임베딩

## 종류

- 유료 임베딩 모델
  - 사용하기 편리하지만 비용 발생
  - API 통신 이용하므로 보안 우려
  - 한국어 포함 많은 언어 임베딩 지원
  - GPU 없이도 빠른 임베딩
- 로컬 임베딩 모델
  - 무료지만 다소 어려운 사용
  - 오픈소스 모델 사용하므로 보안 우수
  - 모델마다 지원 언어가 다름
  - GPU 없을 시, 느린 임베딩

### Gemini

```
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -- 베딩 모델 API 호출 --
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",     # Gemini 전용 Embedding 모델
)

#  -- PDF 문서 로드 --
loader = PyPDFLoader(r"/content/drive/MyDrive/Langchain  RAG AI 챗봇 완전정복/활용문서/[이슈리포트 2022-2호] 혁신성장 정책금융 동향.pdf")
pages = loader.load()

# -- PDF 문서를 여러 청크로 분할 --
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

texts = text_splitter.split_documents(pages)

# Gemini 임베딩 모델로 청크들을 임베딩 변환하기
embeddings = embeddings_model.embed_documents([i.page_content for i in texts])
len(embeddings), len(embeddings[0])

# -- 문장 유사도 계산 --

examples= embeddings_model.embed_documents(
     [
        "안녕하세요",
        "제 이름은 홍두깨입니다.",
        "이름이 무엇인가요?",
        "랭체인은 유용합니다.",
     ]
 )

# -- 예시 질문과 답변 임베딩 --
embedded_query_q = embeddings_model.embed_query("이 대화에서 언급된 이름은 무엇입니까?")
embedded_query_a = embeddings_model.embed_query("이 대화에서 언급된 이름은 홍길동입니다.")

# -- 벡터간의 유사도 계산 --

from numpy import dot # 벡터간의 내적합
from numpy.linalg import norm # 길이
import numpy as np

def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))

print(cos_sim(embedded_query_q, embedded_query_a))
print(cos_sim(embedded_query_a, examples [1]))
print(cos_sim(embedded_query_a, examples [3]))

-- 답변 --
0.8048620547570755
0.7336600213611468
0.6100377530273482
```

### 오픈소스

```
from langchain_community.embeddings import HuggingFaceEmbeddings

#HuggingfaceEmbedding 함수로 Open source 임베딩 모델 로드
model_name = "jhgan/ko-sroberta-multitask" # 한글 임베딩 모델
ko_embedding= HuggingFaceEmbeddings(
    model_name=model_name
)

examples = ko_embedding.embed_documents(
     [
        "안녕하세요",
        "제 이름은 홍두깨입니다.",
        "이름이 무엇인가요?",
        "랭체인은 유용합니다.",
     ]
 )

embedded_query_q = ko_embedding.embed_query("이 대화에서 언급된 이름은 무엇입니까?")
embedded_query_a = ko_embedding.embed_query("이 대화에서 언급된 이름은 홍길동입니다.")

print(cos_sim(embedded_query_q, embedded_query_a))
print(cos_sim(embedded_query_q, examples[1]))
print(cos_sim(embedded_query_q, examples[3]))

-- 답변 --
0.6070005852394463
0.2947341657162066
0.2757840706251745
```
