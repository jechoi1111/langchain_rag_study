# Langchain의 이해와 활용

## Langchain이란?

- LLM 어플리케이션을 보조하는 프레임워크.
- 다양한 언어 모델과 도구를 쉽게 통합할 수 있으며, 유연성과 재사용성이 높아 많은 기업에서 활용하고 있음.

## Langchain은 왜 필요할까?

- LLM의 발전과 함께 중요해진 프롬프트 엔지니어링을 보조할 수 있음.
- 프로픔트 엔지니어링의 번거로움을 줄이기 위해 Prompt Template이라는 모듈 지원.

  - 프롬프트 엔지니어링 : LLM에게 어떻게 질문하는가?

- RAG도 사용자의 질문에 힌트 문장을 합하여 LLM에게 전달하는 일종의 프롬프트 엔지니어링. (사용자의 질문과 유사 문장 합침)

## Langchain의 아키텍처

- 프롬프트 엔지니어링 뿐만 아니라 RAG, Agent 등의 시스템을 만들기 위한 모듈을 모두 갖춤.

  - Models : OpenAI, Google, Antropic
  - Prompts : Prompt Template, Partial
  - Example Selectors : Dynamic Example Selector
  - Document Loader : PDF, PPTX, Word
  - Text Splitters : Recursive, HTML
  - Vector Stores : Chroma, FAISS, Qdrant
  - Output Parsers : CSV, JSON
  - Tools : Web Search, SQL, Pandas

### 가장 핵심적인 컴포넌트

1. LLM : 초거대 언어모델로, Langchain의 엔진과 같은 역할 (GPT-4, Gemini...)
2. Prompts : 초거대 언어모델에게 지시하는 명령문 (Prompt Template, Chat Prompt Template...)
3. Index : LLM이 문서를 쉽게 탐색할 수 있도록 구조화 (Document Loaders, Text Splitters, Vector Store, Retriever...)
4. Memory : 채팅 이력을 저장하여, 이를 기반으로 대화 가능하게 함. (ConverstatinoBufferMemory, Entity Memory)
5. Chain : LLM 사슬을 형성하여 연속적인 LLM 호출 (LLM Chain, QA Chain, Summarization Chain)
6. Agnets : LLM이 스스로 어떤 작업을 수행할지 계획하고 수행 (Webserach Agent, SQL Agent...)

## Langchain을 활용한 RAG 구축

- 어떤 형태의 문서이든지 Langchain을 활용해 RAG 시스템을 구축한다면, 아래 흐름을 따라 구축

1. Document Loader(문서 업로드) : PyPDFLoader를 활용한 문서 가져오기
2. Text Splitter(문서 분할) : PDF 문서를 여러 문서로 분할
3. Embed to Vectorstore(문서 임베딩) : LLM이 이핼할 수 있도록 문서 수치화
4. VectorStore Retriever(임베딩 검색) : 질문과 연관성이 높은 문서 추출
5. QA Chain(답변 생성)
