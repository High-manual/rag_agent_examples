"""
Graph API:에이전트를 노드(node)와 엣지(edge)로 이루어진 그래프 구조로 정의하고 싶을 때 사용

“이 에이전트, 나중에 엄청 복잡해질 것 같다”
“사람들한테 구조를 설명해야 한다”
“분기·병렬·상태 흐름이 핵심이다”
"""
#기본 설정
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#----------------------------------------------
#1. 도구와 모델 정의하기
#----------------------------------------------
from langchain.tools import tool
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "gpt-4o-mini",
    temperature=0
)


# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.
    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.
    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.
    Args:
        a: First int
        b: Second int
    """
    return a / b


# LLM에 도구 추가
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools} #실제 툴 객체를 바로 찾을 수 있도록
model_with_tools = model.bind_tools(tools) #모델에 도구 연결


#----------------------------------------------
#2. 상태 정의하기
#----------------------------------------------
from langchain.messages import AnyMessage #어떤 메시지든지 가능
from typing_extensions import TypedDict, Annotated
import operator #지금은 파이썬 연산자 사용


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add] 
    # 메시지 누적 리스트, reducer (단순 add적용)
    llm_calls: int #llm 호출 횟수 확인용


#----------------------------------------------
#3. 모델 노드 정의하기
#----------------------------------------------
from langchain.messages import SystemMessage


def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="당신은 주어진 입력값들에 대해 산술 연산을 수행하는 유능한 비서입니다."
                    )
                ]
                + state["messages"] #유저 메시지 + 이전 대화 + tool message
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1 #llm 호출 횟수 증가
    }


#----------------------------------------------
#4. 툴 노드 정의하기
#----------------------------------------------
from langchain.messages import ToolMessage


def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


#----------------------------------------------
#5. 엔드 로직 정의하기
#----------------------------------------------
from typing import Literal #주어진 보기 안에서 선택하도록 강제
from langgraph.graph import StateGraph, START, END


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END


#----------------------------------------------
#6. 에이전트 빌드, 컴파일 (그래프 연결)
#----------------------------------------------
# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
) #조건 처리
agent_builder.add_edge("tool_node", "llm_call")

# 설계도(그래프)를 “실행 가능한 에이전트”로 바꿈
agent = agent_builder.compile() 

# Show the agent
print("Here is the mermaid graph syntax. You can paste it into https://mermaid.live/ :") #사이트 들어가서 코드 붙여넣기
print(agent.get_graph(xray=True).draw_mermaid())

# Invoke
from langchain.messages import HumanMessage
messages = [HumanMessage(content="Multiply 3 and 4.")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()