# 기본 설정
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

#-----------------------------------------
#1. 문서 전처리
#-----------------------------------------
# print(f"\n####1. 문서 전처리####")

from langchain_community.document_loaders import WebBaseLoader
#웹페이지를 읽어서 Document 객체 리스트로 바꿔주는 로더

urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

docs = [WebBaseLoader(url).load() for url in urls]
# print(f"가져온 문서 일부 확인하기: {docs[0][0].page_content.strip()[:10]}")


from langchain_text_splitters import RecursiveCharacterTextSplitter

docs_list = [item for sublist in docs for item in sublist]
#2중 리스트 형태의 docs를 flatten

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50, add_start_index=True
)
doc_splits = text_splitter.split_documents(docs_list)
# print(f"indexing을 위해 검색된 문서를 작은 chunk로 분할:{doc_splits[0].page_content.strip()[:10]}")


# -----------------------------------------
# 2. Retriever 도구 생성
# -----------------------------------------
# print(f"\n####2. Retriever 도구 생성####")

#in-memory vector store and openai embedding
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits, embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

#create a retriever tool
from langchain.tools import tool

@tool
def retrieve_blog_posts(query: str) -> str:
    """Lilian Weng 블로그 게시물에서 정보를 검색하여 반환합니다."""
    docs = retriever.invoke(query)
    
    # print(f"First page content: {docs[0].page_content[:200]}\n")
    # print(f"Metadata: {docs[0].metadata}")
    
    return "\n\n".join([doc.page_content for doc in docs])

retriever_tool = retrieve_blog_posts

# 검색 도구 테스트
# result = retriever_tool.invoke({"query": "types of reward hacking"})
# print(f"\n검색된 문서 조각: {result}")


#-----------------------------------------
#3. Query 생성
#-----------------------------------------
# print(f"\n####3. Query 생성####")

# 쿼리 생성 또는 응답 노드 정의
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model

response_model = init_chat_model("gpt-4o", temperature=0)


def generate_query_or_respond(state: MessagesState):
    """현재 상태를 기반으로 모델을 호출하여 응답을 생성합니다.
    질문에 따라 검색 도구를 사용할지, 아니면 바로 사용자에게 응답할지 결정합니다.
    """
    response = (
        response_model
        .bind_tools([retriever_tool]).invoke(state["messages"])
    )
    return {"messages": [response]}

# 일반 질문으로 테스트
input = {"messages": [{"role": "user", "content": "hello!"}]}
# print("\n일반 질문으로 테스트")
# generate_query_or_respond(input)["messages"][-1].pretty_print()

# 검색이 필요한 질문으로 테스트
input = {
    "messages": [
        {
            "role": "user",
            "content": "What does Lilian Weng say about types of reward hacking?",
        }
    ]
}
# print("\n검색이 필요한 질문으로 테스트")
# generate_query_or_respond(input)["messages"][-1].pretty_print()


#-----------------------------------------
#4. 찾아온 문서 품질 평가
#-----------------------------------------
# print(f"\n####4. 찾아온 문서 품질 평가####")

from pydantic import BaseModel, Field
from typing import Literal

GRADE_PROMPT = (
    "You are a grader deciding whether the retrieved text is useful "
    "for answering the user question.\n\n"
    "Retrieved text:\n{context}\n\n"
    "User question:\n{question}\n\n"
    "If the retrieved text contains factual information that could be used "
    "to answer the question (even partially), respond with 'yes'. "
    "Respond with 'no' only if it is completely unrelated."
)

class GradeDocuments(BaseModel):
    """문서의 관련성을 이진 점수로 평가합니다."""
    binary_score: str = Field(
        description="관련성 점수: 관련 있으면 'yes', 없으면 'no'"
    )

grader_model = init_chat_model("gpt-4o", temperature=0)

def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """검색된 문서가 질문과 관련이 있는지 판단합니다."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score

    if score == "yes":
        # print(f"[평가: {score}] 관련성 있는 문서 발견 → 답변 생성")
        return "generate_answer"
    else:
        # print(f"[평가: {score}] 관련성 부족 → 질문 재작성")
        return "rewrite_question"

# 관련 없는 문서로 테스트
from langchain_core.messages import convert_to_messages

input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            },
            {"role": "tool", "content": "meow", "tool_call_id": "1"},
        ]
    )
}
# print("\n관련 없는 문서로 테스트")
# grade_documents(input)

# 관련 있는 문서로 테스트
input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
                "tool_call_id": "1",
            },
        ]
    )
}
# print("\n관련 있는 문서로 테스트")
# grade_documents(input)


#-----------------------------------------
#5. 질문 재작성
#-----------------------------------------
# print(f"\n####5. 질문 재작성####")

from langchain.messages import HumanMessage

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)

def rewrite_question(state: MessagesState):
    """원래 사용자 질문을 더 나은 형태로 재작성합니다."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}

#관련 없는 질문 넣어보기
input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            },
            {"role": "tool", "content": "meow", "tool_call_id": "1"},
        ]
    )
}

# response = rewrite_question(input)
# print(f"재작성된 질문: {response['messages'][-1].content}")


#-----------------------------------------
#6. 답변 생성
#-----------------------------------------
# print(f"\n####6. 답변 생성####")

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


def generate_answer(state: MessagesState):
    """검색된 문서를 기반으로 최종 답변을 생성합니다."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
                "tool_call_id": "1",
            },
        ]
    )
}

# response = generate_answer(input)
# response["messages"][-1].pretty_print()


#-----------------------------------------
#7. 그래프 조립
#-----------------------------------------
print(f"\n####7. 그래프 조립####")

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

workflow = StateGraph(MessagesState)

# 노드 추가
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

# 검색 여부 결정
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

# 검색 후 문서 품질 평가
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# 그래프 컴파일
graph = workflow.compile()

# 그래프 시각화 (https://mermaid.live/ 에서 확인 가능)
print(graph.get_graph(xray=True).draw_mermaid())



#-----------------------------------------
#8. agentic RAG 실행
#-----------------------------------------
print(f"\n####8. agentic RAG 실행####")

for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            }
        ]
    }
):
    for node, update in chunk.items():
        print(f"\n#### [{node}] 노드 실행 결과 ####")
        update["messages"][-1].pretty_print()
        print("\n")