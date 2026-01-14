"""
Concepts
This guide focuses on retrieval of text data. We will cover the following concepts:
Documents and document loaders;
Text splitters;
Embeddings;
Vector stores and retrievers.
"""

#-----------------------------------
#basic setting
#-----------------------------------
# pip install langchain-community pypdf

from dotenv import load_dotenv
import os
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

#-----------------------------------
#1. Documents and Document Loaders
#-----------------------------------
"""
- page_content: 문서의 실제 내용을 담고 있는 문자열
- metadata: 임의의 메타데이터를 담는 딕셔너리
  (문서의 출처, 다른 문서와의 관계, 그 외 다양한 부가 정보를 담을 수 있음)
- id: (선택 사항) 문서를 식별하기 위한 문자열 ID
"""

#Loading documents
from langchain_community.document_loaders import PyPDFLoader

file_path = "nke-10k-2023.pdf"
loader = PyPDFLoader(file_path) 
#https://docs.langchain.com/oss/python/integrations/document_loaders/pypdfloader
## PyPDFLoader loads one Document object per PDF page.

docs = loader.load()

print(f"Number of pages: {len(docs)}")
print(f"First page content: {docs[0].page_content[:200]}\n")
print(f"Metadata: {docs[0].metadata}")

#-----------------------------------
#2. Text splitters
#-----------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
#https://docs.langchain.com/oss/javascript/integrations/splitters/recursive_text_splitter
# 문자 기반으로 분할하는 텍스트 분할기
# - chunkSize: 문자 수
# - chunkOverlap: 앞뒤 chunk가 겹치는 부분
# - add_start_index: chunk의 시작이 원문의 어느 위치에 있는지 표시

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(f"Number of splits: {len(all_splits)}")

#-----------------------------------
#3. Embeddings
#-----------------------------------
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])


#-----------------------------------
#4. Vector stores and retrievers
#-----------------------------------
# ChromaDB (다른 벡터 디비 사용해도 문제 X)
# pip install -qU langchain-chroma

from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="retrieval/chroma_langchain_db",  
    # Where to save data locally, remove if not necessary
)

ids = vector_store.add_documents(documents=all_splits)


"""
vectorstore 직접 사용해서 검색하기
"""
#---------------------------------------------------------
#1. 문자열 질문으로 유사 문서 찾기
results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)
print("\n1. 문자열 질문으로 유사 문서 찾기")
print(f"answer: {results[0]}")

#---------------------------------------------------------
#2. Async query 비동기 검색(이미 async 환경이라는 전제에서 사용해야함)
# results = await vector_store.asimilarity_search("When was Nike incorporated?")
# print("\n2. Async query 비동기 검색")
# print(f"answer: {results[0]}")

#---------------------------------------------------------
#3. 유사도 점수까지 같이 받기
## score는 provider마다 다르고, 여기선 distance(거리)라서 값이 낮을수록 더 유사함.
results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results[0]
print("\n3. 유사도 점수까지 같이 받기")
print(f"Score: {score}\n")
print(f"Doc: {doc}")

#---------------------------------------------------------
#4.“질문을 벡터로 만들어서” 검색하기
embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")
results = vector_store.similarity_search_by_vector(embedding)
print("\n4.‘질문을 벡터로 만들어서’ 검색하기")
print(f"answer: {results[0]}")

"""
Retrieval(검색기)를 사용해서 검색하기
- similarity: 가장 유사한 문서를 찾음
- mmr: 비슷하되, 서로 너무 겹치지 않게
- similarity_threshold: 유사도 점수가 기준 이하인 문서 제거
"""
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

results = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)
print("\nRetrieval(검색기)를 사용해서 검색하기")
print(f"answer: {results}")


#-----------------------------------
#5. Augmented Generation (증강 생성)
#-----------------------------------
"""
검색된 문서를 LLM에 전달하여 답변을 생성
"""

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("\n" + "="*80)
print("5. RAG - Augmented Generation (증강 생성)")
print("="*80)

# 1. 질문으로 관련 문서 검색
question = "How many distribution centers does Nike have in the US?"
docs = retriever.invoke(question)

# 2. 검색된 문서들을 하나의 컨텍스트로 합치기
context = "\n\n".join(doc.page_content for doc in docs)

# 3. 프롬프트 구성
prompt = f"""You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:"""

# 4. LLM으로 답변 생성
response = llm.invoke(prompt)
answer = response.content

print(f"\n질문: {question}")
print(f"답변: {answer}")