# RAG 구성 요소 실습 - Text Splitters

## Text Spiltter란?

- RAG는 Document Loader로 불러온 문서를 벡터 임베딩으로 변환하여 벡터DB에 저장하고 이를 활용
- LLM에게 문서를 그대로 입력하여 답변하도록 하면 입력값 길이 제한으로 인해 오류가 발생할 수 있음

= 문서를 여러 개의 조각(Chunk)로 분할하여 벡터 DB에 저장하고, 이를 RAG 시스템에 활용

## Text Spiltter 종류

기본 Splitter와 Recursive Splitter가 있음.
대부분의 경우 Recursive Splitter 사용.

- CharacterTextSplitter : 구분자 1개 기준으로 분할하므로,
  max_token을 지키지 못하는 경우 발생
- RecursiveCharacterTextSplitter : 줄바꿈, 마침표, 쉼표 순으로 재귀적으로 분할하므로,
  max_token 지켜 분할

## Text Spiltter 매개변수

- Chunk_overlap이라는 매개변수는 텍스트 분할 시, 앞뒤로 조금씩 겹치게 만들어 문맥을 더 많이 포함하도록 함.

## Semantic Chunker

- 문자 분할기(Character)나 재귀적 분할기(Recursive)처럼 기계적으로 분할하지 않고, 유사성 기반으로 분할.

```
# 시멘틱 청크
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

loader = PyPDFLoader(r"/content/drive/MyDrive/Langchain  RAG AI 챗봇 완전정복/활용문서/[이슈리포트 2022-2호] 혁신성장 정책금융 동향.pdf")
pages = loader.load()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",     # Gemini 전용 Embedding 모델
    google_api_key="API_KEY"
)

text_splitter = SemanticChunker(embeddings)

texts = text_splitter.split_documents(pages)

print("-"*100)
print("[첫번째 청크]")
print(texts[0].page_content)
print("-"*100)
print("[두번째 청크]")
print(texts[1].page_content)

-- 결과 --
[첫번째 청크]
혁신성장 정책금융 동향 : ICT 산업을 중심으로
  CIS이슈리포트 2022-2호 | 1 |
<요  약>▶혁신성장 정책금융기관*은 혁신성장산업 영위기업을 발굴·지원하기 위한 정책금융 가이드라인**에 따라 혁신성장 기술분야에 대한 금융지원을 강화하고 있음     * 산업은행, 기업은행, 수출입은행, 신용보증기금, 기술보증기금, 중소벤처기업진흥공단, 무역보험공사 등 11개 기관...
----------------------------------------------------------------------------------------------------
[두번째 청크]
| 2 | CIS이슈리포트 2022-2호
▶혁신성장 ICT 산업의 정책금융 공급규모 및 공급속도를 종합적으로 분석한 결과, 차세대무선통신미디어, 능동형컴퓨팅(이상 정보통신 테마), 차세대반도체(전기전자 테마) 및 객체탐지(센서측정 테마) 기술분야로 혁신성장 정책금융이 집중되고 있음[ICT 산업 내 주요 기술분야 혁신성장 정책금융 공급 현황]                                                            (단위: 억 원, %)...
```
