# ShopSmart 벡터 DB 기반 제품 추천 시스템

ShopSmart의 Vector DB를 활용한 시맨틱 검색 및 개인화된 제품 추천 시스템입니다. 
이 시스템은 벡터 DB(Chroma)와 자연어 처리를 통해 사용자의 쿼리를 이해하고 의미적으로 가장 관련성 높은 제품을 추천합니다.

## 📋 목차

1. [개요](#개요)
2. [기능](#기능)
3. [시스템 아키텍처](#시스템-아키텍처)
4. [설치 및 설정](#설치-및-설정)
5. [사용 방법](#사용-방법)
6. [구현 세부 사항](#구현-세부-사항)
7. [문제 해결](#문제-해결)
8. [GitHub 업로드 안내](#github-업로드-안내)
9. [추가 개발 가능성](#추가-개발-가능성)

## 📝 개요

ShopSmart는 Vector DB와 OpenAI API를 활용하여 고객의 자연어 쿼리를 이해하고 제품을 추천하는 AI 시스템입니다. 
기존 키워드 기반 검색 시스템과 달리, 의미 기반 검색을 통해 고객이 원하는 제품을 더 정확하게 찾아줍니다.

### 주요 특징

- **시맨틱 검색**: 고객의 의도를 이해하는 자연어 기반 제품 검색
- **OpenAI 임베딩**: 고품질 텍스트 임베딩을 위한 OpenAI API 활용
- **벡터 데이터베이스**: 효율적인 유사도 검색을 위한 Chroma DB 활용
- **멀티모달 인터페이스**: API, 웹 UI, 챗봇을 통한 다양한 접근 방식
- **고급 랭킹**: 유사도, 평점, 리뷰 수를 고려한 지능형 랭킹 알고리즘
- **통합 웹 검색**: 선택적으로 LangChain과 Tavily를 활용한 웹 검색을 제품 검색과 통합

## 🚀 기능

### 1. 시맨틱 검색

- 자연어 쿼리를 이해하고 관련 제품 추천
- 카테고리, 브랜드, 가격, 평점 등 다양한 필터링 지원
- 의미적 유사성을 기반으로 한 추천
- 유사도 점수 표시 및 제품 간 유사성 이유 분석

### 2. 제품 탐색

- 비슷한 제품 추천 기능
- 카테고리 및 브랜드별 탐색
- 가격, 평점, 리뷰 수에 따른 정렬
- 특징 태그 시각화

### 3. 웹 검색 기능

- LangChain 기반 웹 검색 기능
- Tavily 검색 API를 통한 실시간 정보 검색
- 웹 검색 결과를 활용한 쿼리 향상 및 제품 검색 정확도 개선
- 토글 방식으로 간편하게 활성화/비활성화 가능

### 4. 다양한 인터페이스

- **FastAPI 기반 REST API**: 다른 시스템과의 통합 용이
- **Streamlit 웹 UI**: 직관적인 사용자 인터페이스
- **제품 검색과 웹 검색 통합**: 단일 인터페이스에서 다양한 검색 옵션 제공

## 🏗️ 시스템 아키텍처

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│   Streamlit UI    │────▶│   FastAPI Server  │◀───▶│    Vector DB      │
│   (app.py)        │     │   (api_server.py) │     │    (Chroma)       │
│                   │     │                   │     │                   │
└───────────────────┘     └─────────┬─────────┘     └───────────────────┘
                                    │
                                    │
                          ┌─────────▼─────────┐     ┌───────────────────┐
                          │                   │     │                   │
                          │  Recommender Core │◀───▶│   OpenAI API      │
                          │  (core/)          │     │                   │
                          │                   │     │                   │
                          └─────────┬─────────┘     └───────────────────┘
                                    │
                                    │
                          ┌─────────▼─────────┐     ┌───────────────────┐
                          │                   │     │                   │
                          │  Web Search       │◀───▶│   Tavily API      │
                          │  Engine           │     │                   │
                          │  (core/web_search)│     │                   │
                          │                   │     │                   │
                          └───────────────────┘     └───────────────────┘
```

### 주요 컴포넌트

1. **추천 시스템 코어 (core/)**
   - `embedding.py`: OpenAI API를 통한 임베딩 생성
   - `recommender.py`: 제품 추천 로직 구현
   - `utils.py`: 유틸리티 함수 모음
   - `web_search.py`: LangChain 기반 웹 검색 엔진

2. **API 서버 (api_server.py)**
   - FastAPI 기반 RESTful API
   - 제품 검색 및 추천 API 엔드포인트 제공
   - 웹 검색 API 엔드포인트 제공

3. **웹 UI (app.py)**
   - Streamlit 기반 사용자 인터페이스
   - 제품 검색 및 웹 검색 통합 UI
   - 개선된 제품 카드 및 유사도 정보 시각화

## 🔧 설치 및 설정

### 필수 요구사항

- Python 3.9 이상
- OpenAI API 키 (https://platform.openai.com/api-keys)
- Tavily API 키 (https://tavily.com)

### 간편 설치 (윈도우)

1. **설치 스크립트 실행**

```
setup.bat
```

이 스크립트는 다음 작업을 수행합니다:
- 가상환경 생성 및 활성화
- 필요한 패키지 설치
- 환경 설정 파일(.env) 생성
- 필요한 디렉토리 생성

2. **API 키 설정**

`.env` 파일을 열고 다음 API 키를 설정합니다:

```
OPENAI_API_KEY=your-openai-api-key-here
TAVILY_API_KEY=your-tavily-api-key-here
API_URL=http://localhost:8000
```

### 수동 설치 (모든 OS)

1. **저장소 클론 또는 다운로드**

```bash
git clone <repository-url>  # 또는 ZIP 파일 다운로드 및 압축 해제
cd VectorDB_10k_실습
```

2. **가상환경 생성 및 활성화**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. **필요한 패키지 설치**

```bash
pip install -r requirements.txt
```

4. **환경 변수 설정**

`.env.template` 파일을 `.env`로 복사하고 필요한 API 키를 설정합니다:

```bash
# Windows
copy .env.template .env

# macOS/Linux
cp .env.template .env
```

그리고 `.env` 파일을 편집하여 API 키를 입력합니다:

```
OPENAI_API_KEY=your-openai-api-key-here
TAVILY_API_KEY=your-tavily-api-key-here
API_URL=http://localhost:8000
```

## 📱 사용 방법

### 벡터 DB 재생성 방법

벡터 DB를 다시 생성하여 모든 제품의 임베딩을 업데이트할 필요가 있는 경우:

1. **모든 Python 프로세스 종료**

```bash
# Windows
taskkill /F /IM python.exe

# macOS/Linux
pkill -f python
```

2. **chroma_db 디렉토리 삭제**

```bash
# Windows
rmdir /S /Q chroma_db

# macOS/Linux
rm -rf chroma_db
```

3. **API 서버 재시작**

```bash
python api_server.py
```

API 서버는 자동으로 새 벡터 DB를 생성하고 모든 제품에 대해 업데이트된 임베딩을 생성합니다.

### 시스템 실행 (중요: 실행 순서)

1. **API 서버 먼저 실행**

```bash
# Windows 명령 프롬프트
run_api.bat

# Windows Git Bash
sh run_api.bat
# 또는
python api_server.py
```

2. **그 다음 Streamlit 앱 실행 (새 터미널에서)**

```bash
# Windows 명령 프롬프트
run_app.bat

# Windows Git Bash
sh run_app.bat
# 또는
streamlit run app.py
```

3. **브라우저에서 Streamlit 앱 접속**
   - http://localhost:8501 접속

### 웹 인터페이스 사용법

1. 브라우저에서 http://localhost:8501 접속 (Streamlit 앱)

2. **제품 검색**
   - 검색창에 원하는 제품 설명 입력 (예: "고급스러운 남성복을 찾고 있어요")
   - 「웹 검색 활용」토글을 켜면 웹에서 최신 정보를 검색하여 더 정확한 제품 추천
   - 필요에 따라 카테고리, 브랜드, 가격, 평점 필터 사용
   - 검색 버튼 클릭하여 제품 찾기

### API 사용법

API 서버가 실행 중인 상태에서 다음 엔드포인트를 사용할 수 있습니다:

1. **제품 검색 API**

```
POST /search
```

요청 본문
```json
{
  "query": "고급스러운 남성복",
  "category": "패션",
  "brand": "나이키",
  "price_min": 10000,
  "price_max": 100000,
  "min_rating": 4.0,
  "top_k": 5
}
```

2. **유사 제품 API**

```
POST /similar
```

요청 본문
```json
{
  "product_id": "PROD-00001",
  "top_k": 5
}
```

3. **웹 검색 API**

```
POST /web-search
```

요청 본문
```json
{
  "query": "최신 스마트폰 트렌드는?"
}
```

4. **통합 검색 API** (신규)

```
POST /integrated-search
```

요청 본문
```json
{
  "query": "최신 트렌드에 맞는 바지",
  "use_web_search": true,
  "category": "패션",
  "brand": "리바이스",
  "price_min": 50000,
  "price_max": 150000,
  "min_rating": 4.0,
  "top_k": 5
}
```

## 🔍 구현 세부 사항

### 1. 벡터 임베딩

- 제품의 모든 관련 필드(이름, 설명, 카테고리, 서브카테고리, 브랜드, 특징 태그 등)를 포함한 통합 텍스트 생성
- OpenAI의 임베딩 모델(text-embedding-ada-002)을 사용하여 고품질 벡터 임베딩 생성
- 코사인 유사도를 사용한 효율적인 유사성 계산

### 2. 랭킹 알고리즘

- 유사도 점수(70%), 평점(20%), 리뷰 수(10%)를 가중치로 결합한 최종 점수 계산
- 가격 및 평점 기반 필터링으로 개인화된 결과 제공

### 3. 통합 웹 검색 기능

- 사용자 토글 방식으로 웹 검색 활성화/비활성화
- 활성화 시 Tavily API를 통해 최신 정보 검색
- GPT-4o 모델을 활용하여 웹 검색 결과와 원본 쿼리를 결합한 향상된 검색 쿼리 생성
- 향상된 쿼리를 통해 벡터 DB 검색 정확도 개선
- 웹 검색 결과를 UI에 함께 표시하여 참고 정보 제공

### 4. UI 개선사항

- 카드 형태의 제품 표시로 정보 가독성 향상
- 유사도 점수와 추천 점수를 시각적으로 명확하게 표시
- 특징 태그를 태그 형태로 직관적 표시
- 유사성 이유 표시로 추천 이유 파악 용이

## 🔧 문제 해결

### ChromaDB 관련 오류

1. **"Expected where to have exactly one operator" 오류**
   - 원인: ChromaDB의 필터 구문이 잘못된 형식일 때 발생
   - 해결: 필터링 쿼리에서 올바른 연산자 형식(`$gte`, `$lte` 등) 사용

2. **벡터 DB 관련 오류**
   - 원인: 기존 DB가 잠겨있거나 손상된 경우
   - 해결: 모든 Python 프로세스 종료 후 `chroma_db` 폴더 삭제 및 재생성

### 웹 검색 관련 오류

1. **"Tavily API key not found" 오류**
   - 원인: Tavily API 키가 설정되지 않았거나 잘못됨
   - 해결: `.env` 파일에 유효한 `TAVILY_API_KEY` 설정

2. **LangChain 관련 오류**
   - 원인: 필요한 LangChain 패키지가 설치되지 않음
   - 해결: `pip install langchain-community tavily-python` 실행

## 🌟 추가 개발 가능성

현재 시스템의 확장 가능성은 다음과 같습니다:

1. **멀티모달 검색**: 이미지와 텍스트를 함께 활용한 제품 검색
2. **개인화 추천**: 사용자의 검색 및 구매 이력을 기반으로 한 개인화된 추천
3. **대화형 쇼핑 어시스턴트**: 대화 흐름에 따른 지능형 제품 추천
4. **실시간 데이터 업데이트**: 제품 데이터 자동 스크래핑 및 임베딩 업데이트
5. **멀티 벡터 저장소 지원**: Pinecone, Weaviate 등 다양한 벡터 DB 지원

## 📤 GitHub 업로드 안내

이 프로젝트를 GitHub에 업로드할 때 다음 사항에 주의하세요

### .gitignore 설정

프로젝트에는 다음과 같은 항목이 .gitignore에 포함되어 있습니다

```
# 환경 변수 및 시크릿
.env
.env.*

# 가상환경
venv/

# Chroma DB 데이터
chroma_db/

# 로그 파일
logs/
*.log

# 데이터 파일 (선택적)
# product_data_10k.json
```

### 대용량 데이터 처리

- **product_data_10k.json**: 이 파일은 약 6.8MB로, GitHub에 기본적으로 포함됩니다. 파일 크기가 너무 큰 경우 .gitignore에서 주석을 해제하여 제외할 수 있습니다.
- **chroma_db/**: 벡터 DB 데이터는 크기가 크고 개인 환경에 맞게 재생성되므로 항상 제외됩니다.

### 업로드 전 확인사항

1. 민감한 API 키가 코드에 하드코딩되어 있지 않은지 확인하세요.
2. product_data_10k.json 파일이 저장소에 포함되어야 하는지 결정하세요.
   - 포함할 경우: 기본 설정 유지
   - 제외할 경우: .gitignore에서 해당 줄의 주석 해제
3. README.md에 데이터 파일 획득 방법에 대한 안내가 있는지 확인하세요.

---

## 📜 라이센스

이 프로젝트는 MIT 라이센스 하에 제공됩니다.

## 🙏 감사의 글

이 프로젝트는 다음 오픈소스 라이브러리의 도움을 받았습니다
- [Chroma](https://github.com/chroma-core/chroma)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [LangChain](https://github.com/langchain-ai/langchain)
- [OpenAI API](https://platform.openai.com/)
