"""
FastAPI를 활용한 제품 추천 API 서버
"""
import os
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# 로컬 모듈 임포트
from core.recommender import ProductRecommendationSystem
from core.utils import setup_logger

# 환경 변수 로드
load_dotenv()

# 로거 설정
logger = setup_logger("api_server", "logs/api_server.log")

# 웹 검색 모듈 임포트
try:
    from core.web_search import WebSearchEngine
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    logger.warning("웹 검색 모듈을 가져올 수 없습니다. 웹 검색 기능이 비활성화됩니다.")

# FastAPI 앱 생성
app = FastAPI(
    title="ShopSmart 제품 추천 API",
    description="벡터 DB를 활용한 시맨틱 검색 및 제품 추천 시스템 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 오리진 허용 (프로덕션에서는 특정 오리진으로 제한해야 함)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic 모델 - 요청
class SearchRequest(BaseModel):
    query: str = Field(..., description="검색 쿼리 (예: '고급스러운 남성복')")
    category: Optional[str] = Field(None, description="카테고리 필터 (예: '패션')")
    brand: Optional[str] = Field(None, description="브랜드 필터 (예: '나이키')")
    price_min: Optional[int] = Field(None, ge=0, description="최소 가격")
    price_max: Optional[int] = Field(None, ge=0, description="최대 가격")
    min_rating: Optional[float] = Field(None, ge=0, le=5, description="최소 평점 (0-5)")
    top_k: int = Field(5, ge=1, le=50, description="반환할 결과 수")

class SimilarProductRequest(BaseModel):
    product_id: str = Field(..., description="제품 ID (예: 'PROD-00001')")
    top_k: int = Field(5, ge=1, le=50, description="반환할 결과 수")

# Pydantic 모델 - 응답
class ProductSummary(BaseModel):
    product_id: str
    product_name: str
    description: str
    price: int
    rating: float
    category: str
    brand: str
    similarity_score: float
    final_score: Optional[float] = None

class SearchResponse(BaseModel):
    query: str
    processing_time: Optional[str] = None
    message: Optional[str] = None
    recommendations: List[ProductSummary]

class SimilarProductResponse(BaseModel):
    product_id: str
    product_name: str
    message: Optional[str] = None
    similar_products: List[ProductSummary]

# 글로벌 변수 - 추천 시스템
recommender: Optional[ProductRecommendationSystem] = None
web_search_engine: Optional[WebSearchEngine] = None

# 추천 시스템 생성 및 초기화
def get_recommender() -> ProductRecommendationSystem:
    global recommender
    
    if recommender is None:
        logger.info("추천 시스템 초기화 중...")
        
        # 제품 데이터 파일 경로 확인
        current_dir = Path(__file__).parent
        json_path = current_dir / "product_data_10k.json"
        
        if not json_path.exists():
            logger.error(f"제품 데이터 파일을 찾을 수 없습니다: {json_path}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="제품 데이터 파일을 찾을 수 없습니다."
            )
        
        # 추천 시스템 초기화
        recommender = ProductRecommendationSystem(str(json_path))
        
        # 데이터 로드
        if not recommender.load_data():
            logger.error("제품 데이터 로드 실패")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="제품 데이터 로드 실패"
            )
        
        # 데이터 전처리
        recommender.preprocess_data()
        
        # API 키 환경 변수 확인 - 실제 초기화는 각 API 요청 시에만 수행
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. 임베딩 기능이 동작하지 않을 수 있습니다.")
        
        logger.info("추천 시스템 기본 초기화 완료 (임베딩 모델은 첫 요청 시 초기화됩니다)")
    
    return recommender

# 웹 검색 요청 모델
class WebSearchRequest(BaseModel):
    query: str = Field(..., description="검색 쿼리 (예: '최신 스마트폰 트렌드는?')")

# 웹 검색 응답 모델
class WebSearchResponse(BaseModel):
    query: str
    response: str
    success: bool

# 웹 검색 엔진 초기화 함수
def get_web_search_engine() -> WebSearchEngine:
    global web_search_engine
    
    if not WEB_SEARCH_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="웹 검색 기능을 사용할 수 없습니다. 필요한 라이브러리가 설치되지 않았습니다."
        )
    
    if web_search_engine is None:
        logger.info("웹 검색 엔진 초기화 중...")
        
        # OpenAI API 키 확인
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OpenAI API 키가 설정되지 않았습니다. .env 파일이나 환경 변수를 확인하세요."
            )
        
        # Tavily API 키 확인
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            logger.error("TAVILY_API_KEY 환경 변수가 설정되지 않았습니다.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Tavily API 키가 설정되지 않았습니다. .env 파일이나 환경 변수를 확인하세요."
            )
        
        try:
            # 웹 검색 엔진 초기화
            web_search_engine = WebSearchEngine(api_key=openai_api_key)
            logger.info("웹 검색 엔진 초기화 완료")
            
        except Exception as e:
            logger.error(f"웹 검색 엔진 초기화 중 오류 발생: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"웹 검색 엔진 초기화 중 오류 발생: {str(e)}"
            )
    
    return web_search_engine

# 통합 검색 요청 모델
class IntegratedSearchRequest(BaseModel):
    query: str = Field(..., description="검색 쿼리 (예: '최신 트렌드에 맞는 바지')")
    use_web_search: bool = Field(False, description="웹 검색 사용 여부")
    category: Optional[str] = Field(None, description="카테고리 필터 (예: '패션')")
    brand: Optional[str] = Field(None, description="브랜드 필터 (예: '나이키')")
    price_min: Optional[int] = Field(None, ge=0, description="최소 가격")
    price_max: Optional[int] = Field(None, ge=0, description="최대 가격")
    min_rating: Optional[float] = Field(None, ge=0, le=5, description="최소 평점 (0-5)")
    top_k: int = Field(5, ge=1, le=50, description="반환할 결과 수")

# 통합 검색 응답 모델
class IntegratedSearchResponse(BaseModel):
    query: str
    processing_time: Optional[str] = None
    message: Optional[str] = None
    web_search_used: bool = False
    web_search_result: Optional[str] = None
    recommendations: List[ProductSummary]

# 애플리케이션 시작 시 초기화
@app.on_event("startup")
async def startup_event():
    try:
        # 추천 시스템 초기화 시도
        get_recommender()
    except Exception as e:
        logger.error(f"애플리케이션 시작 시 초기화 중 오류 발생: {str(e)}", exc_info=True)
        # 초기화 실패해도 서버는 시작되도록 함 (나중에 다시 시도 가능)

# API 엔드포인트
@app.get("/", tags=["기본"])
async def root():
    """API 서버 상태 확인"""
    return {
        "status": "online",
        "message": "ShopSmart 제품 추천 API가 실행 중입니다",
        "version": "1.0.0"
    }

@app.post("/search", response_model=SearchResponse, tags=["검색"])
async def search_products(request: SearchRequest, recommender: ProductRecommendationSystem = Depends(get_recommender)):
    """
    제품 검색 API
    
    사용자 쿼리와 필터를 기반으로 제품을 검색하고 추천합니다.
    """
    try:
        # OpenAI API 키 확인 및 임베딩 모델 설정 (처음 요청 시 한 번만 초기화)
        if recommender.embedding_model is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="OpenAI API 키가 설정되지 않았습니다. .env 파일이나 환경 변수를 확인하세요."
                )
            
            try:
                # 임베딩 모델 설정
                recommender.setup_embedding_model(api_key=openai_api_key)
                
                # 벡터 DB 설정
                recommender.setup_vector_db(force_recreate=False)
                
            except Exception as e:
                logger.error(f"임베딩 모델 초기화 중 오류 발생: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"임베딩 모델 초기화 중 오류 발생: {str(e)}"
                )
        
        # 쿼리 구성
        query = request.query
        
        # 필터 직접 구성 대신 별도 파라미터로 추천 시스템에 전달
        # price_range 파라미터를 조건부로 전달하여 None 값에 의한 문제 방지
        price_range = None
        if request.price_min is not None and request.price_max is not None:
            # 최소/최대 가격이 모두 0인 경우(기본값) 필터링 적용하지 않음
            if not(request.price_min == 0 and request.price_max == 0):
                price_range = (request.price_min, request.price_max)
                logger.info(f"가격 범위 필터 적용: {request.price_min:,}원-{request.price_max:,}원")
            else:
                logger.info("가격 범위 필터 미적용 (기본값 0-0)")
        
        # 최소 평점이 0인 경우(기본값) 필터링 적용하지 않음
        min_rating = None
        if request.min_rating is not None and request.min_rating > 0:
            min_rating = request.min_rating
            logger.info(f"최소 평점 필터 적용: {min_rating}")
        else:
            logger.info("최소 평점 필터 미적용 (기본값 0.0 또는 값 없음)")
        
        # 카테고리와 브랜드 필터 로깅
        if request.category:
            logger.info(f"카테고리 필터 적용: {request.category}")
        if request.brand:
            logger.info(f"브랜드 필터 적용: {request.brand}")
        
        # 로깅 추가
        logger.info(f"검색 요청: 쿼리='{query}', 카테고리={request.category}, 브랜드={request.brand}, "
                   f"가격범위={price_range}, 최소평점={min_rating}")
        
        result = recommender.search(
            query=query,
            top_k=request.top_k,
            category=request.category,
            brand=request.brand,
            price_range=price_range,
            min_rating=min_rating
        )
        
        return result
    
    except Exception as e:
        logger.error(f"검색 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"검색 중 오류 발생: {str(e)}"
        )

@app.post("/similar", response_model=SimilarProductResponse, tags=["추천"])
async def find_similar_products(request: SimilarProductRequest, recommender: ProductRecommendationSystem = Depends(get_recommender)):
    """
    유사 제품 추천 API
    
    특정 제품 ID를 기반으로 유사한 제품을 추천합니다.
    """
    try:
        # OpenAI API 키 확인 및 임베딩 모델 설정 (처음 요청 시 한 번만 초기화)
        if recommender.embedding_model is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="OpenAI API 키가 설정되지 않았습니다. .env 파일이나 환경 변수를 확인하세요."
                )
            
            try:
                # 임베딩 모델 설정
                recommender.setup_embedding_model(api_key=openai_api_key)
                
                # 벡터 DB 설정
                recommender.setup_vector_db(force_recreate=False)
                
            except Exception as e:
                logger.error(f"임베딩 모델 초기화 중 오류 발생: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"임베딩 모델 초기화 중 오류 발생: {str(e)}"
                )
        
        # 유사 제품 검색
        result = recommender.get_similar_products(request.product_id, top_k=request.top_k)
        
        # 오류 메시지가 있는 경우
        if "message" in result and not result.get("similar_products"):
            if "찾을 수 없습니다" in result["message"]:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=result["message"]
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=result["message"]
                )
        
        return result
    
    except HTTPException:
        # 이미 처리된 HTTP 예외는 그대로 전파
        raise
    
    except Exception as e:
        logger.error(f"유사 제품 검색 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"유사 제품 검색 중 오류 발생: {str(e)}"
        )

@app.post("/web-search", response_model=WebSearchResponse, tags=["웹 검색"])
async def web_search(request: WebSearchRequest, web_search_engine: WebSearchEngine = Depends(get_web_search_engine)):
    """
    웹 검색 API
    
    사용자 쿼리를 기반으로 웹 검색을 수행하고 결과를 요약합니다.
    """
    try:
        # 웹 검색 수행
        logger.info(f"웹 검색 요청: '{request.query}'")
        result = web_search_engine.search(request.query)
        return result
    
    except Exception as e:
        logger.error(f"웹 검색 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"웹 검색 중 오류 발생: {str(e)}"
        )

@app.post("/integrated-search", response_model=IntegratedSearchResponse, tags=["검색"])
async def integrated_search(
    request: IntegratedSearchRequest, 
    recommender: ProductRecommendationSystem = Depends(get_recommender)
):
    """
    통합 검색 API
    
    사용자 쿼리와 필터를 기반으로 제품을 검색하고, 
    웹 검색 옵션이 활성화된 경우 웹 검색 결과를 함께 제공합니다.
    """
    start_time = time.time()
    original_query = request.query
    enhanced_query = original_query
    web_search_result = None
    web_search_error = None
    
    # OpenAI API 키 확인 및 임베딩 모델 설정 (처음 요청 시 한 번만 초기화)
    if recommender.embedding_model is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OpenAI API 키가 설정되지 않았습니다. .env 파일이나 환경 변수를 확인하세요."
            )
        
        try:
            # 임베딩 모델 설정
            recommender.setup_embedding_model(api_key=openai_api_key)
            
            # 벡터 DB 설정
            recommender.setup_vector_db(force_recreate=False)
            
        except Exception as e:
            logger.error(f"임베딩 모델 초기화 중 오류 발생: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"임베딩 모델 초기화 중 오류 발생: {str(e)}"
            )
    
    # 웹 검색 옵션이 활성화된 경우
    if request.use_web_search and WEB_SEARCH_AVAILABLE:
        try:
            # 웹 검색 엔진 초기화 - 함수 내부에서 직접 호출
            web_search_engine = get_web_search_engine()
            
            logger.info(f"웹 검색 실행: '{original_query}'")
            
            # 웹 검색 수행 (타임아웃 5초로 단축)
            try:
                search_result = await asyncio.wait_for(
                    asyncio.to_thread(web_search_engine.search, original_query),
                    timeout=5.0
                )
                
                if search_result.get("success", False):
                    web_search_result = search_result.get("response")
                    
                    # GPT를 사용한 쿼리 향상 기능 분리 - 타임아웃 5초로 단축
                    if web_search_result:
                        try:
                            openai_api_key = os.getenv("OPENAI_API_KEY")
                            from openai import OpenAI
                            
                            client = OpenAI(api_key=openai_api_key)
                            
                            # 매우 간소화된 프롬프트
                            prompt = f"""원본 쿼리: "{original_query}"
                            제품 검색을 위한 웹 검색이 필요합니다. 다음 형식으로 검색에 최적화된 쿼리를 작성해주세요:
                            1. 제품 카테고리 키워드 (예: 노트북, 스마트폰)
                            2. 주요 특징 키워드 2-3개 (예: 가벼운, 고성능)
                            3. 최신 트렌드 관련 키워드 (가능한 경우)
                            쿼리만 반환하세요."""
                            
                            response = await asyncio.wait_for(
                                asyncio.to_thread(
                                    lambda: client.chat.completions.create(
                                        model="gpt-4o",
                                        messages=[{"role": "user", "content": prompt}],
                                        temperature=0.3,
                                        max_tokens=50
                                    )
                                ),
                                timeout=5.0
                            )
                            
                            enhanced_query = response.choices[0].message.content.strip()
                            logger.info(f"향상된 쿼리: '{enhanced_query}'")
                        
                        except asyncio.TimeoutError:
                            logger.warning("쿼리 향상 시간 초과. 원본 쿼리로 진행합니다.")
                        except Exception as e:
                            logger.error(f"쿼리 향상 오류: {str(e)}")
            
            except asyncio.TimeoutError:
                logger.warning("웹 검색 시간 초과. 제품 검색만 진행합니다.")
                web_search_error = "웹 검색 시간 초과"
            
        except Exception as e:
            logger.error(f"웹 검색 처리 오류: {str(e)}")
            web_search_error = f"웹 검색 오류: {str(e)}"
    
    # 제품 검색 실행
    try:
        # 필터 정보 구성
        price_range = None
        if request.price_min is not None and request.price_max is not None:
            if not(request.price_min == 0 and request.price_max == 0):
                price_range = (request.price_min, request.price_max)
                logger.info(f"가격 필터: {request.price_min}-{request.price_max}")
        
        min_rating = None
        if request.min_rating and request.min_rating > 0:
            min_rating = request.min_rating
            logger.info(f"평점 필터: {min_rating}")
        
        # 제품 검색 실행 (타임아웃 10초로 단축)
        logger.info(f"제품 검색 시작: '{enhanced_query}'")
        try:
            search_results = await asyncio.wait_for(
                asyncio.to_thread(
                    recommender.search,
                    query=enhanced_query,
                    category=request.category,
                    brand=request.brand,
                    price_range=price_range,
                    min_rating=min_rating,
                    top_k=request.top_k,
                    web_search_result=web_search_result  # 웹 검색 결과 전달
                ),
                timeout=10.0
            )
            logger.info(f"제품 검색 완료: {len(search_results.get('recommendations', []))}개 결과")
        except asyncio.TimeoutError:
            processing_time = f"{time.time() - start_time:.2f}초"
            logger.error(f"제품 검색 시간 초과 ({processing_time})")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="검색 시간이 초과되었습니다. 검색어를 단순화하거나 필터를 조정해보세요."
            )
        
        # 처리 시간 계산
        processing_time = f"{time.time() - start_time:.2f}초"
        
        # 웹 검색 오류가 있는 경우 메시지에 포함
        message = f"웹 검색{' 사용' if request.use_web_search else ' 미사용'}으로 검색 완료"
        if web_search_error:
            message += f" (참고: {web_search_error})"
        
        # 응답 구성
        return {
            "query": original_query,
            "processing_time": processing_time,
            "message": message,
            "web_search_used": request.use_web_search,
            "web_search_result": web_search_result,
            "recommendations": search_results.get("recommendations", [])
        }
        
    except HTTPException:
        # HTTP 예외는 그대로 전달
        raise
    except Exception as e:
        logger.error(f"통합 검색 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"검색 중 오류가 발생했습니다: {str(e)}"
        )

# 메인 함수 (직접 실행 시)
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8001, reload=True)
