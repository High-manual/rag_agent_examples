# RAG Agent Examples

LangChain과 LangGraph를 활용한 RAG(Retrieval-Augmented Generation) 에이전트 예제 모음입니다.
기본 RAG 구현부터 복잡한 에이전트 워크플로우까지 단계별 예제를 제공합니다.

## 프로젝트 구조

```
rag_agent_ex/
├── examples/
│   ├── 01.rag_example.py          # 기본 RAG 파이프라인
│   ├── 02.graph_api.py            # LangGraph API 기초
│   ├── 03.custom_rag_agent.py     # 고급 RAG 에이전트
│   └── nke-10k-2023.pdf           # 예제 PDF 문서
├── env.sample                      # 환경 변수 템플릿
└── pyproject.toml                  # 의존성 정의
```

## 설치 방법

### 필수 요구사항
- Python 3.12 이상
- OpenAI API 키

### 의존성 설치

```bash
# uv 사용
uv sync

# 또는 pip 사용
pip install langchain[openai] langchain-chroma langchain-community langchain-openai langgraph pypdf python-dotenv langchain-text-splitters bs4
```

## 사용 방법

```bash
cd examples

# 기본 RAG 파이프라인
uv run 01.rag_example.py

# LangGraph API 기초
uv run 02.graph_api.py

# 고급 RAG 에이전트
uv run 03.custom_rag_agent.py
```

## 예제 설명

### 01.rag_example.py
Nike 연차보고서를 활용한 기본 RAG 파이프라인
- PDF 로딩 → 텍스트 분할 → 임베딩 생성 → 벡터 저장 → 검색 → 답변 생성

### 02.graph_api.py
산술 연산 에이전트로 배우는 LangGraph 기초
- 도구 정의, 상태 관리, 노드/엣지 구성, 조건부 분기

### 03.custom_rag_agent.py
블로그 검색 에이전트 (고급)
- 웹 크롤링, 문서 품질 평가, 질문 재작성, 조건부 워크플로우
