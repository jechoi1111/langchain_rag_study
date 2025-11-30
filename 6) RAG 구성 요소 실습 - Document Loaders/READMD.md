# RAG 구성 요소 실습 - Document Loaders

## Document Loaders란?

- 주어진 문서를 RAG에서 활용하기 용이한 형태(Document 객체)로 변환하는 역할
- Document 객체는 문서의 내용을 담은 Page_content와 메타데이터로 이뤄진 Dictionary임.

### PDF

```
#PyPDFLoader 불러오기
from langchain_community.document_loaders import PyPDFLoader

# PDF파일 불러올 객체 PyPDFLoader 선언
loader = PyPDFLoader(r"/content/drive/MyDrive/[이슈리포트 2022-2호] 혁신성장 정책금융 동향.pdf")

# PDF파일 로드 및 페이지별로 자르기
pages = loader.load_and_split()
print(pages[5].page_content)

-- 답변 --
| 6 | CIS이슈리포트 2022-2호
▶(주요품목① : 5G 이동통신) 정보통신 테마 내 기술분야 중 혁신성장 정책금융 공급규모가 가장 큰 차세대무선통신미디어 분야의 경우 4G/5G 기술품목의 정책금융 공급 비중이 가장 높은 것으로 확인됨[차세대무선통신미디어 분야 내 기술품목별 혁신성장 정책금융 공급액 추이](단위: 억 원)
▶5G 이동통신 시스템은 ITU(International Telecommunication Union)가 정의한 5세대 이동통신 규격을 만족시키는 무선 이동통신 네트워크 기술로, 2019년부터 국내 서비스를 시작함￮4G 이동통신 시스템(LTE)과 비교할 때 전송속도의 향상(1Gbps→20Gbps), 이동성 향상(350km/h→500km/h에서 끊김없는 데이터 전송 가능), 최대 연결가능 기기수 증가(10만 대 →100만 대 이상), 데이터 전송지연 감소(10ms→1ms) 등의 향상된 기능을 제공함￮5G는 전송속도 향상, 다수기기 접속 및 지연시간 단축을 위해 ①밀리미터파 통신이 가능한 주파수 확장, ②스몰셀(Small cell)을 도입한 기지국, ③다중안테나 송수신(Massive MIMO), ④네트워크 슬라이싱(Network Slicing) 등의 기술을 도입함[5G 주요 요소기술 특징]
자료: 삼정 KPMG
```

```
print(pages[5].metadata)

-- 답변 --
{'producer': 'Hancom PDF 1.3.0.538', 'creator': 'Hancom PDF 1.3.0.538', 'creationdate': '2022-07-29T09:03:16+09:00', 'author': 'kmd kdy', 'moddate': '2022-07-29T09:03:16+09:00', 'pdfversion': '1.4', 'source': '/content/drive/MyDrive/[이슈리포트 2022-2호] 혁신성장 정책금융 동향.pdf', 'total_pages': 18, 'page': 5, 'page_label': '6'}
```

```
#OCR기능 위해 설치
# OCR은 Optical Character Recognition의 약자로,
# 이미지나 PDF 안에 있는 글자를 “텍스트”로 변환해주는 기술.

!pip install rapidocr-onnxruntime

---

#PyPDFLoader 불러오기
from langchain_community.document_loaders import PyPDFLoader

# PDF파일 불러올 객체 PyPDFLoader 선언(extract_images 매개변수로 OCR 수행)
loader = PyPDFLoader(r"/content/drive/MyDrive/[이슈리포트 2022-2호] 혁신성장 정책금융 동향.pdf", extract_images=True)

# PDF파일 로드 및 페이지별로 자르기
pages = loader.load_and_split()
print(pages[5].page_content)
```

### Word

```
#docx2txt 설치
!pip install --upgrade --quiet  docx2txt

--

#Docx2txtLoader 불러오기
from langchain_community.document_loaders import Docx2txtLoader

#Docx2txtLoader로 워드 파일 불러오기(경로 설정)
loader = Docx2txtLoader(r"/content/drive/MyDrive/Langchain  RAG AI 챗봇 완전정복/활용문서/[삼성전자] 사업보고서(일반법인) (2021.03.09).docx")

#페이지로 분할하여 불러오기
data = loader.load_and_split()

# data[0]
#첫번째 페이지 출력하기
print(data[12].page_content[:500])
```

### CSV

```
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path=r"/content/drive/MyDrive/Langchain  RAG AI 챗봇 완전정복/활용문서/mlb_teams_2012.csv")

data = loader.load()

data[0]
```

### PPT

```
#pthon-pptx 패키지 설치
!pip install -q python-pptx unstructured

--

#UnstructuredPowerPointLoader 불러오기
from langchain_community.document_loaders import UnstructuredPowerPointLoader

#mode=elements를 통해 pptx의 요소별로 Document 객체로 가져오기
loader = UnstructuredPowerPointLoader(r"/content/drive/MyDrive/Langchain  RAG AI 챗봇 완전정복/활용문서/Copilot-scenarios-for-Marketing.pptx", mode="elements")

#pptx 파일을 분할 로드하기
data = loader.load_and_split()

data[1]
```

### 인터넷 정보 로드

```
from langchain_community.document_loaders import WebBaseLoader
#텍스트 추출할 URL 입력
loader = WebBaseLoader("https://www.espn.com/")
#ssl verification 에러 방지를 위한 코드
loader.requests_kwargs = {'verify':False}
data = loader.load()
data

-- 헤드라인만 가져오기 --
# 헤드라인만 가져오기

import bs4
from langchain_community.document_loaders import WebBaseLoader
#텍스트 추출할 URL 입력
loader = WebBaseLoader("https://www.espn.com/",
                        bs_kwargs=dict(
                            parse_only=bs4.SoupStrainer(
                                class_=("headlineStack top-headlines") # 가져오고 싶은 부분의 class name
                                                        )
                                        )
                      )
#ssl verification 에러 방지를 위한 코드
loader.requests_kwargs = {'verify':False}
data = loader.load()
data

-- 여러 웹정보 로드 --
# 여러 웹사이트 정보 로드

loader = WebBaseLoader(["https://www.espn.com/", "https://google.com"])
docs = loader.load()
docs
```
