# RAG 구성 요소 실습 - Prompt Templates

## Prompt Template의 개념

- **"프롬프트"** 는 모델에 대한 입력을 의미.

  - 실제 LLM 서비스들의 경우, 사용자가 전부 입력하도록 만들지 않고 Back 단에서 여러 구성요소를 통해 편리한 입력을 지원하도록 함.

- **"프롬프트 템플릿"** 은 이러한 편리한 입력 지원을 위한 모듈.
  - Langchain은 프롬프트를 쉽게 구성하고 작업할 수 있도록 여러 클래스와 함수 제공.

= 반복되는 프롬프트를 템플릿으로 구성.

## Prompt Template의 종류

- 기본적인 템플릿 설정을 위한 Prompt Template과 상세 설정이 가능한 ChatPromptTemplate 이 있음.

  - Prompt Template : 하나의 프롬프트 안에 매개변수 모두 집어 넣음(간단)
  - ChatPromptTemplate : 역할을 나눠서 프롬프트 세부적으로 설정

```
from langchain_core.prompts import PromptTemplate

prompt = (
PromptTemplate.from_template(
"""
너는 요리사야. 내가 가진 재료들을 갖고 만들 수 있는 요리를 {개수}추천하고,
그 요리의 레시피를 제시해줘. 내가 가진 재료는 아래와 같아.
<재료>
{재료}
"""
)
)

prompt

-- 결과 --
PromptTemplate(input_variables=['개수', '재료'], input_types={}, partial_variables={}, template='\n        너는 요리사야. 내가 가진 재료들을 갖고 만들 수 있는 요리를 {개수}추천하고, \n        그 요리의 레시피를 제시해줘. 내가 가진 재료는 아래와 같아. \n        <재료> \n        {재료}\n        ')
```

```
prompt.format(개수=3, 재료="사과, 양파, 계란")

-- 결과 --
너는 요리사야. 내가 가진 재료들을 갖고 만들 수 있는 요리를 3추천하고,
그 요리의 레시피를 제시해줘. 내가 가진 재료는 아래와 같아.
<재료>
사과, 양파, 계란
```

```
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

prompt = SystemMessage(content =
        """
        너는 항상 밝은 말투로 대화하는 챗봇이야. 답변의 끝에 이모티콘 붙여줘.
        """
        )

new_prompt = (
    prompt
    + HumanMessage(content =
                  """
                  오늘은 날씨가 어때?
                  """)
    + AIMessage(content =
                """
                오늘은 날씨가 아주 좋아요!
                """)
    + """{input}"""
)

new_prompt.format_messages(input = "오늘 너의 기분은 어때?")

-- 결과 --
[SystemMessage(content='\n        너는 항상 밝은 말투로 대화하는 챗봇이야. 답변의 끝에 이모티콘 붙여줘.\n        ', additional_kwargs={}, response_metadata={}),
 HumanMessage(content=' \n                  오늘은 날씨가 어때?\n                  ', additional_kwargs={}, response_metadata={}),
 AIMessage(content='\n                오늘은 날씨가 아주 좋아요!\n                ', additional_kwargs={}, response_metadata={}),
 HumanMessage(content='오늘 너의 기분은 어때?', additional_kwargs={}, response_metadata={})]
```

### 간단한 LLM Chain 구성해보기

```
import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# new_prompt를 직접 체인의 첫 번째 요소로 사용합니다.
chain = new_prompt | llm
chain.invoke("오늘 너의 기분은 어때?")

-- 결과 --
AIMessage(content='저는 언제나 여러분을 도와드릴 준비가 되어 있어서 늘 기분이 최고랍니다! 😊✨', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'model_provider': 'google_genai'}, id='lc_run--61e7ee55-decb-4584-9295-212f89fa5b7f-0', usage_metadata={'input_tokens': 66, 'output_tokens': 165, 'total_tokens': 231, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 146}})
```

## Few-shot Prompt Template

- 퓨샷 예제를 제공하면 해당 예제와 유사한 형태의 결과물을 출력. 프롬프트로 표현하기 어려운 경우 사용.

```
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

examples = [
    {
        "question": "아이유로 삼행시 만들어줘.",
        "answer":
        """
        아: 아이유는
        이: 이런 강의를 들을 이
        유: 유가 없다.
        """
    }
]

example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="""
    질문: {question}
    답변: {answer}
    """
)

print(example_prompt.format(**examples[0])) # ** -> key와 value 값을 나열해서 출력하겠다는 의미

-- 결과 --
  질문: 아이유로 삼행시 만들어줘.
    답변:
        아: 아이유는
        이: 이런 강의를 들을 이
        유: 유가 없다.
```

```
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="질문: {input}\n답변:",
    input_variables=["input"]
)

print(prompt.format(input="손흥민으로 삼행시 만들어줘"))

-- 결과 --

    질문: 아이유로 삼행시 만들어줘.
    답변:
        아: 아이유는
        이: 이런 강의를 들을 이
        유: 유가 없다.



질문: 손흥민으로 삼행시 만들어줘
답변:
```

```
result = model.invoke(prompt.format(input="손흥민으로 삼행시 만들어줘"))
print(result.content)

-- 결과 --
네, 손흥민으로 삼행시 지어드릴게요!

**손:** 손에 땀을 쥐게 하는
**흥:** 흥미진진한 경기로 우리를 즐겁게 하고,
**민:** 민족의 자부심을 드높이는 선수!
```

## Partial Prompt Template

- Prompt Template의 매개변수 중 몇 개만 미리 지정해두는 것.

```
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("나이 {age} \n직업: {job}")
partial_prompt = prompt.partial(age="20")
print(partial_prompt.format(job="개발자"))

-- 결과 --
나이 20
직업: 개발자
```

```
from datetime import datetime

def _get_datetime():
  now = datetime.now()
  return now.strftime("%m/%d/%Y, %H:%M:%S")

prompt = PromptTemplate(
    template="오늘의 날짜: {today}. 날씨는 {weather}",
    input_variables=["today", "weather"]
)

partial_prompt = prompt.partial(today=_get_datetime)
print(partial_prompt.format(weather="맑음"))

-- 결과 --
오늘의 날짜: 11/29/2025, 16:10:53. 날씨는 맑음
```
