# RAG 구성 요소 실습 - Models

## LLM API 활용하기

- Langchain을 활용하면 다양한 모델 API를 일관된 형식으로 불러올 수 있음.
- Langchain에서는 LLM에게 보내는 프롬프트의 형식을 크게 3가지로 구분

1. SystemMessage : LLM에게 역할을 부여하는 메시지
2. HumanMessage : LLM에게 전달하는 사용자의 메시지
3. AIMessage : LLM이 출력한 메시지

```
from langchain_core.prompts import ChatPromptTemplate


chat_template = ChatPromptTemplate.from_messages(
    [
        # SystemMessage: 유요한 쳇봇이라는 역할과 이름을 부여
        ("system", "You are a helpful AI bot. Your name is {name}"),

        # HumanMeesage와 AIMessage: 서로 안부를 묻고 답하는 대화 히스토리 주입
        ("human", "Hello. how are you doing?"),
        ("ai", "I'm doing well. thanks"),

        # HumanMessage로 사용자가 입력한 프롬프트를 전달
        ("human", "{user_input}")
    ]
)

messages = chat_template.format_messages(name="Bob", user_input="What is your name?")

print(messages)
```

## LLM의 Temperature 이해하기

- LLM의 매개변수 중 하나인 Temperature는 답변의 일관성을 조정.

```
import os
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"

# Temperature=0
gemini_temp0_1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature = 0)
gemini_temp0_2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature = 0)

# Temperature=1
gemini_temp1_1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature = 1)
gemini_temp1_2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature = 1)

model_list = [gemini_temp0_1, gemini_temp0_2, gemini_temp1_1, gemini_temp1_2]

for i in model_list:
  answer = i.invoke("왜 파이썬이 가장 인기있는 프로그래밍 언어인지 한 문장으로 설명해줘.", max_tokens = 128)
  print("-"*100)
  print(">>>", answer.content)
```

## 답변 스트리밍 하기

```
from langchain_google_genai import ChatGoogleGenerativeAI

chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature = 0)

for chunk in chat.stream("달에 관한 시를 써줘"):
  print(chunk.content, end="", flush=True)
```

## LLM 답변 캐싱하기

- LLM은 답변을 생성하는 데에 시간과 비용을 소모함. 이를 효과적으로 관리하기 위해 같은 답은 캐싱하여 사용해야함.

```
%%time
from langchain_core.caches import InMemoryCache
set_llm_cache(InMemoryCache()) # 캐시메모리 설정

chat.invoke("일반상대성 이론을 한마디로 설명해줘")
```

```
CPU times: user 12.4 ms, sys: 496 µs, total: 12.9 ms
Wall time: 5.19 s

AIMessage(content='**중력은 시공간의 휘어짐이다.**\n\n(조금 더 풀어서 설명하자면, 질량과 에너지가 시공간을 휘게 만들고, 그 휘어진 시공간이 바로 우리가 중력이라고 느끼는 현상입니다.)', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'model_provider': 'google_genai'}, id='lc_run--81370abb-7471-48a8-b358-204581af0608-0', usage_metadata={'input_tokens': 15, 'output_tokens': 990, 'total_tokens': 1005, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 933}})
```

같은 질문 전달

```
%%time
# 같은 질문 전달
chat.invoke("일반상대성 이론을 한마디로 설명해줘")
```

```
CPU times: user 656 µs, sys: 0 ns, total: 656 µs
Wall time: 670 µs

AIMessage(content='**중력은 시공간의 휘어짐이다.**\n\n(조금 더 풀어서 설명하자면, 질량과 에너지가 시공간을 휘게 만들고, 그 휘어진 시공간이 바로 우리가 중력이라고 느끼는 현상입니다.)', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'model_provider': 'google_genai'}, id='lc_run--81370abb-7471-48a8-b358-204581af0608-0', usage_metadata={'input_tokens': 15, 'output_tokens': 990, 'total_tokens': 1005, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 933}, 'total_cost': 0})
```
