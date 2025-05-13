"""
Streamlit 대시보드 애플리케이션
"""
import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# 로컬 모듈 임포트
try:
    from core.recommender import ProductRecommendationSystem
    from core.utils import setup_logger, format_price, format_date
    LOCAL_MODE = True
except ImportError:
    LOCAL_MODE = False

# 환경 변수 로드
load_dotenv()

# 로깅 설정
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("streamlit_app")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler("logs/streamlit_app.log", encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# API URL 설정
API_URL = os.getenv("API_URL", "http://localhost:8001")

# 페이지 설정
st.set_page_config(
    page_title="ShopSmart - 제품 추천 시스템",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #F8FAFC;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 1rem;
    }
    .card-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.75rem;
        color: #0F172A;
    }
    .product-price {
        font-size: 1.1rem;
        font-weight: bold;
        color: #0F766E;
    }
    .product-rating {
        color: #D97706;
        font-weight: bold;
    }
    .similarity-score {
        font-size: 1rem;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        background-color: #dbeafe;
        color: #1e40af;
        display: inline-block;
        margin-right: 0.5rem;
    }
    .final-score {
        font-size: 1rem;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        background-color: #dcfce7;
        color: #166534;
        display: inline-block;
    }
    .search-tabs {
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F1F5F9;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748B;
    }
    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E2E8F0;
        text-align: center;
        color: #94A3B8;
        font-size: 0.8rem;
    }
    .feature-tags {
        margin-top: 0.5rem;
    }
    .tag {
        display: inline-block;
        background-color: #f1f5f9;
        color: #334155;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        margin-right: 0.25rem;
        margin-bottom: 0.25rem;
        font-size: 0.8rem;
    }
    .similarity-reason {
        margin-top: 0.5rem;
        padding: 0.5rem;
        background-color: #f8fafc;
        border-left: 3px solid #94a3b8;
        font-style: italic;
    }
    /* 웹 검색 결과 스타일 */
    .web-result {
        padding: 1rem;
        background-color: #f0f9ff;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #0ea5e9;
    }
    /* 모바일 최적화 */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .sub-header {
            font-size: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "search"  # 'search' 또는 'detail' 모드

# 함수 정의
def search_products(query: str, category: Optional[str] = None, brand: Optional[str] = None, 
                   price_min: Optional[int] = None, price_max: Optional[int] = None,
                   min_rating: Optional[float] = None, top_k: int = 5) -> Dict[str, Any]:
    """
    API를 통해 제품 검색
    """
    try:
        payload = {
            "query": query,
            "category": category,
            "brand": brand,
            "price_min": price_min,
            "price_max": price_max,
            "min_rating": min_rating,
            "top_k": top_k
        }
        
        # API 요청
        response = requests.post(f"{API_URL}/search", json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API 오류: {response.status_code} - {response.text}")
            return {
                "query": query,
                "message": f"API 오류: {response.status_code}",
                "recommendations": []
            }
    
    except Exception as e:
        logger.error(f"API 요청 중 오류 발생: {str(e)}")
        st.error(f"API 연결 오류: {str(e)}")
        return {
            "query": query,
            "message": f"API 연결 오류: {str(e)}",
            "recommendations": []
        }

def web_search(query: str) -> Dict[str, Any]:
    """
    API를 통해 웹 검색
    """
    try:
        payload = {
            "query": query
        }
        
        # API 요청
        response = requests.post(f"{API_URL}/web-search", json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"웹 검색 API 오류: {response.status_code} - {response.text}")
            return {
                "query": query,
                "response": f"웹 검색 API 오류: {response.status_code}",
                "success": False
            }
    
    except Exception as e:
        logger.error(f"웹 검색 API 요청 중 오류 발생: {str(e)}")
        st.error(f"웹 검색 API 연결 오류: {str(e)}")
        return {
            "query": query,
            "response": f"웹 검색 API 연결 오류: {str(e)}",
            "success": False
        }

def find_similar_products(product_id: str, top_k: int = 5) -> Dict[str, Any]:
    """
    API를 통해 유사한 제품 검색
    """
    try:
        payload = {
            "product_id": product_id,
            "top_k": top_k
        }
        
        # API 요청
        response = requests.post(f"{API_URL}/similar", json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API 오류: {response.status_code} - {response.text}")
            return {
                "product_id": product_id,
                "message": f"API 오류: {response.status_code}",
                "similar_products": []
            }
    
    except Exception as e:
        logger.error(f"API 요청 중 오류 발생: {str(e)}")
        st.error(f"API 연결 오류: {str(e)}")
        return {
            "product_id": product_id,
            "message": f"API 연결 오류: {str(e)}",
            "similar_products": []
        }

def render_product_card(product: Dict[str, Any], index: int = 0, show_details: bool = False):
    """
    제품 카드 렌더링
    """
    with st.container():
        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # 이미지 (크기 축소)
            st.markdown("<div style='height: 30px; display: flex; align-items: center; justify-content: center;'>", unsafe_allow_html=True)
            # 이미지 URL의 크기도 더 작게 요청 (200x200 대신 150x150)
            st.image(f"https://picsum.photos/seed/{product['product_id']}/150/150", 
                    width=260)  # 표시 크기도 더 작게 제한
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # 제품명 및 기본 정보
            st.markdown(f"#### {product['product_name']}")
            
            # 가격, 평점
            price_col, rating_col = st.columns(2)
            with price_col:
                st.markdown(f"**가격:** <span class='product-price'>{format_price(product['price'])}원</span>", unsafe_allow_html=True)
            with rating_col:
                st.markdown(f"**평점:** <span class='product-rating'>{'⭐' * int(product['rating'])}</span> ({product['rating']})", unsafe_allow_html=True)
            
            # 유사도 점수와 최종 점수 표시
            if "similarity_score" in product or "final_score" in product:
                score_html = ""
                if "similarity_score" in product and product["similarity_score"] is not None:
                    similarity_percentage = product["similarity_score"] * 100
                    score_html += f"<span class='similarity-score'>유사도: {similarity_percentage:.1f}%</span>"
                
            if "final_score" in product and product["final_score"] is not None:
                final_percentage = product["final_score"] * 100
                score_html += f"<span class='final-score'>추천 점수: {final_percentage:.1f}%</span>"
                
                st.markdown(score_html, unsafe_allow_html=True)
            
            # 카테고리 정보
            cat_col, brand_col = st.columns(2)
            with cat_col:
                subcategory = product.get('subcategory', '')
                category_text = f"{product['category']}"
                if subcategory:
                    category_text += f" › {subcategory}"
                st.markdown(f"**카테고리:** {category_text}")
            
            with brand_col:
                st.markdown(f"**브랜드:** {product['brand']}")
            
            # 제품 설명
            st.markdown(f"**설명:** {product['description']}")
            
            # 특징 태그
            if "feature_tags" in product and product["feature_tags"]:
                st.markdown("<div class='feature-tags'>", unsafe_allow_html=True)
                tags_html = ""
                for tag in product["feature_tags"][:8]:  # 최대 8개 태그만 표시
                    tags_html += f"<span class='tag'>#{tag}</span>"
                st.markdown(f"{tags_html}</div>", unsafe_allow_html=True)
            
            # 유사성 이유
            if "similarity_reason" in product:
                st.markdown(f"<div class='similarity-reason'>{product['similarity_reason']}</div>", unsafe_allow_html=True)
            
            # 상세 정보 버튼
            col_detail, col_similar = st.columns(2)
            with col_detail:
                if not show_details:
                    if st.button("상세 정보", key=f"details_{product['product_id']}"):
                        st.session_state.selected_product = product
                        st.session_state.view_mode = "detail"
                        st.rerun()
            
        # 유사 제품 버튼
        with col_similar:
            if st.button("유사 제품", key=f"similar_{product['product_id']}"):
                with st.spinner("유사한 제품 검색 중..."):
                    similar_results = find_similar_products(product['product_id'])
                    
                    if similar_results.get("similar_products"):
                        st.subheader(f"{product['product_name']}와(과) 유사한 제품")
                        
                        # 이 부분을 수정: render_product_card를 다시 호출하지 않음
                        for i, similar in enumerate(similar_results["similar_products"], 1):
                            with st.container():
                                st.markdown(f"### {i}. {similar['product_name']}")
                                
                                # 기본 정보를 간단히 표시
                                st.markdown(f"**가격:** {format_price(similar['price'])}원 | **평점:** {'⭐' * int(similar['rating'])} ({similar['rating']})")
                                st.markdown(f"**카테고리:** {similar['category']} | **브랜드:** {similar['brand']}")
                                st.markdown(f"**설명:** {similar['description']}")
                                
                                # 유사도 점수 표시
                                if "similarity_score" in similar and similar["similarity_score"] is not None:
                                    similarity_percentage = similar["similarity_score"] * 100
                                    st.markdown(f"**유사도:** {similarity_percentage:.1f}%")
                                
                                # 특징 태그가 있으면 표시
                                if "feature_tags" in similar and similar["feature_tags"]:
                                    tags_str = ", ".join([f"#{tag}" for tag in similar["feature_tags"][:8]])
                                    st.markdown(f"**특징:** {tags_str}")
                                
                                # 구분선 추가
                                st.markdown("---")
                    else:
                        st.info(similar_results.get("message", "유사한 제품을 찾을 수 없습니다."))
        
        st.markdown("</div>", unsafe_allow_html=True)

def integrated_search(query: str, use_web_search: bool = False, category: Optional[str] = None, 
                      brand: Optional[str] = None, price_min: Optional[int] = None, 
                      price_max: Optional[int] = None, min_rating: Optional[float] = None, 
                      top_k: int = 5) -> Dict[str, Any]:
    """
    API를 통한 통합 검색 (제품 검색 + 웹 검색)
    """
    try:
        payload = {
            "query": query,
            "use_web_search": use_web_search,
            "category": category,
            "brand": brand,
            "price_min": price_min,
            "price_max": price_max,
            "min_rating": min_rating,
            "top_k": top_k
        }
        
        # API 요청 - 30초 타임아웃 추가
        response = requests.post(f"{API_URL}/integrated-search", json=payload, timeout=60)
        
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = f"API 오류: {response.status_code}"
            try:
                error_details = response.json().get("detail", "")
                if error_details:
                    error_msg += f" - {error_details}"
            except:
                pass
            
            logger.error(error_msg)
            st.error(error_msg)
            return {
                "query": query,
                "message": error_msg,
                "web_search_used": use_web_search,
                "web_search_result": None,
                "recommendations": []
            }
    
    except requests.exceptions.Timeout:
        error_msg = "API 요청 시간 초과. 서버가 응답하지 않습니다."
        logger.error(error_msg)
        st.error(error_msg)
        return {
            "query": query,
            "message": error_msg,
            "web_search_used": use_web_search,
            "web_search_result": None,
            "recommendations": []
        }
    except Exception as e:
        logger.error(f"통합 검색 API 요청 중 오류 발생: {str(e)}")
        st.error(f"API 연결 오류: {str(e)}")
        return {
            "query": query,
            "message": f"API 연결 오류: {str(e)}",
            "web_search_used": use_web_search,
            "web_search_result": None,
            "recommendations": []
        }

def display_integrated_search_results():
    """
    통합 검색 결과 표시
    """
    results = st.session_state.search_results
    
    # 결과 헤더
    st.header(f"'{results['query']}' 검색 결과")
    
    # 처리 시간 표시 (있는 경우)
    if 'processing_time' in results:
        st.markdown(f"*처리 시간: {results['processing_time']}*")
    
    # 웹 검색 결과 표시 (있는 경우)
    if results.get('web_search_used', False) and results.get('web_search_result'):
        with st.expander("웹 검색 결과", expanded=True):
            st.markdown("<div class='web-result'>", unsafe_allow_html=True)
            st.markdown(results.get('web_search_result', ''))
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("*이 정보는 웹 검색을 통해 얻은 실시간 데이터입니다.*")
    
    # 메시지 표시 (있는 경우)
    if 'message' in results and results['message']:
        st.info(results['message'])
    
    # 추천 결과 표시
    if results.get('recommendations'):
        for i, product in enumerate(results['recommendations'], 1):
            render_product_card(product, i)
    else:
        st.warning("검색 결과가 없습니다. 다른 검색어나 필터를 시도해보세요.")

def run_integrated_search(query, use_web_search=False, category=None, brand=None, price_min=None, price_max=None, min_rating=None, top_k=5):
    """
    통합 검색 실행 및 결과 저장
    """
    if not query:
        st.warning("검색어를 입력해주세요.")
        return
    
    try:
        with st.spinner("검색 중..." + (" (웹 검색 포함)" if use_web_search else "")):
            # 검색 실행
            results = integrated_search(
                query=query,
                use_web_search=use_web_search,
                category=category, 
                brand=brand,
                price_min=price_min,
                price_max=price_max,
                min_rating=min_rating,
                top_k=top_k
            )
            
            # 검색 결과 저장
            st.session_state.search_results = results
            
            # 검색 기록에 추가 (중복 제거)
            if query not in st.session_state.search_history:
                st.session_state.search_history.append(query)
            
            # 모드 변경
            st.session_state.view_mode = "search"
            
            # 결과가 없으면 사용자에게 알림
            if not results.get("recommendations"):
                message = results.get("message", "검색 결과가 없습니다. 다른 검색어를 시도해보세요.")
                st.warning(message)
    except Exception as e:
        st.error(f"검색 처리 중 오류가 발생했습니다: {str(e)}")
        logger.error(f"검색 처리 중 예외 발생: {str(e)}", exc_info=True)

def main():
    """
    메인 애플리케이션
    """
    # 헤더
    st.markdown("<h1 class='main-header'>ShopSmart 제품 추천 시스템</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>벡터 DB를 활용한 시맨틱 검색 및 개인화된 제품 추천</p>", unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.title("🛍️ ShopSmart")
        
        # 네비게이션
        nav_option = st.radio("탐색", ["제품 검색", "대시보드", "정보"])
        
        # 검색 기록
        if st.session_state.search_history:
            st.subheader("최근 검색")
            for query in st.session_state.search_history[-5:]:
                if st.button(f"🔍 {query}", key=f"history_{query}"):
                    # 검색 기록에서 쿼리 선택
                    st.session_state.view_mode = "search"
                    run_integrated_search(query)
                    st.rerun()
    
    # 상세 보기 모드
    if st.session_state.view_mode == "detail" and st.session_state.selected_product:
        product = st.session_state.selected_product
        
        # 뒤로 가기 버튼
        if st.button("← 검색 결과로 돌아가기"):
            st.session_state.view_mode = "search"
            st.session_state.selected_product = None
            st.rerun()
        
        # 제품 상세 정보
        st.header(product['product_name'])
        
        # 기본 정보
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**설명:** {product['description']}")
            st.markdown(f"**카테고리:** {product['category']}")
            st.markdown(f"**브랜드:** {product['brand']}")
            
            # 추가 정보 (있는 경우)
            if 'subcategory' in product:
                st.markdown(f"**서브카테고리:** {product['subcategory']}")
            
            # 특징 태그 (있는 경우)
            if 'feature_tags' in product:
                tags = product['feature_tags']
                if isinstance(tags, list) and tags:
                    st.markdown("**특징:**")
                    st.markdown(", ".join([f"#{tag}" for tag in tags]))
        
        with col2:
            # 가격 및 평점
            price = format_price(product['price'])
            st.markdown(f"<div class='product-price'>{price}</div>", unsafe_allow_html=True)
            
            # 할인 정보 (있는 경우)
            if 'discount_rate' in product and product['discount_rate'] > 0:
                st.markdown(f"**할인율:** {product['discount_rate']}%")
            
            # 평점
            rating = product['rating']
            stars = "⭐" * int(rating) + ("½" if rating % 1 >= 0.5 else "")
            st.markdown(f"<div class='product-rating'>{rating}/5.0</div>", unsafe_allow_html=True)
            st.markdown(f"{stars}")
            
            # 리뷰 수 (있는 경우)
            if 'review_count' in product:
                st.markdown(f"리뷰 {product['review_count']}개")
            
            # 재고 상태 (있는 경우)
            if 'stock_status' in product:
                status = product['stock_status']
                if status == "재고 있음":
                    st.markdown("🟢 **재고 있음**")
                elif status == "품절 임박":
                    st.markdown("🟠 **품절 임박**")
                elif status == "일시 품절":
                    st.markdown("🔴 **일시 품절**")
                else:
                    st.markdown(f"**재고 상태:** {status}")
        
        # 유사 제품 조회
        st.subheader("유사한 제품")
        with st.spinner("유사한 제품 검색 중..."):
            similar_results = find_similar_products(product['product_id'])
            
            if similar_results.get("similar_products"):
                for i, similar in enumerate(similar_results["similar_products"], 1):
                    render_product_card(similar, i, show_details=False)
            else:
                st.info(similar_results.get("message", "유사한 제품을 찾을 수 없습니다."))
    
    # 네비게이션에 따른 컨텐츠 표시
    elif nav_option == "제품 검색":
        # 검색 폼
        with st.form("search_form"):
            st.subheader("제품 검색")
            
            # 검색어 입력
            query = st.text_input("찾고 싶은 제품을 설명해주세요", 
                                placeholder="예: 고급스러운 남성복을 찾고 있어요")
            
            # 웹 검색 토글
            use_web_search = st.toggle("웹 검색 활용", value=False, 
                                      help="활성화 시 웹에서 최신 정보를 검색해 더 정확한 제품을 추천합니다")
            
            # 필터 섹션
            with st.expander("필터 옵션", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    category = st.text_input("카테고리", placeholder="예: 패션")
                    # 최소값을 1000으로 설정하여 0원 필터링 방지
                    price_range = st.slider("가격 범위 (원)", 0, 2000000, (1000, 1000000), step=10000)
                
                with col2:
                    brand = st.text_input("브랜드", placeholder="예: 나이키")
                    # 최소값을 0.5로 설정하여 0.0 필터링 방지
                    min_rating = st.slider("최소 평점", 0.0, 5.0, 0.5, step=0.5)
            
            # 결과 수
            top_k = st.slider("결과 수", min_value=1, max_value=20, value=5)
            
            # 검색 버튼
            submitted = st.form_submit_button("검색")
            
            if submitted and query:
                run_integrated_search(
                    query=query, 
                    use_web_search=use_web_search,
                    category=category, 
                    brand=brand, 
                    price_min=price_range[0], 
                    price_max=price_range[1], 
                    min_rating=min_rating, 
                    top_k=top_k
                )
        
        # 검색 결과 표시 (검색 모드이고 결과가 있는 경우)
        if st.session_state.view_mode == "search":
            if hasattr(st.session_state, 'search_results'):
                display_integrated_search_results()
    
    elif nav_option == "대시보드":
        st.header("제품 데이터 대시보드")
        
        # 대시보드 데이터 로드
        with st.spinner("데이터 분석 중..."):
            if LOCAL_MODE and 'product_data' not in st.session_state:
                try:
                    # 로컬 모드에서는 직접 JSON 파일을 읽음
                    with open("product_data_10k.json", 'r', encoding='utf-8') as f:
                        st.session_state.product_data = json.load(f)
                except Exception as e:
                    st.error(f"제품 데이터 로드 중 오류 발생: {str(e)}")
                    return
            
            if 'product_data' in st.session_state:
                products_df = pd.DataFrame(st.session_state.product_data)
                
                # 기본 통계
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{len(products_df):,}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>총 제품 수</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    categories = products_df['category'].nunique()
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{categories}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>카테고리 수</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    brands = products_df['brand'].nunique()
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{brands}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>브랜드 수</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col4:
                    avg_rating = products_df['rating'].mean()
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{avg_rating:.1f}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>평균 평점</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # 카테고리별 제품 분포
                st.subheader("카테고리별 제품 분포")
                category_counts = products_df['category'].value_counts().reset_index()
                category_counts.columns = ['category', 'count']
                
                fig = px.bar(category_counts, x='category', y='count',
                            labels={'count': '제품 수', 'category': '카테고리'},
                            color='count',
                            color_continuous_scale='blues')
                
                fig.update_layout(
                    xaxis_title="카테고리",
                    yaxis_title="제품 수",
                    height=400,
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 가격 분포
                st.subheader("가격 분포")
                
                # 이상치 제외 (상위 2%)
                price_threshold = np.percentile(products_df['price'], 98)
                filtered_prices = products_df[products_df['price'] < price_threshold]['price']
                
                fig = px.histogram(filtered_prices, 
                                nbins=50,
                                labels={'value': '가격 (원)'},
                                color_discrete_sequence=['#3B82F6'])
                
                fig.update_layout(
                    xaxis_title="가격 (원)",
                    yaxis_title="제품 수",
                    height=400,
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 평점별 제품 수
                st.subheader("평점별 제품 분포")
                rating_counts = products_df['rating'].value_counts().sort_index().reset_index()
                rating_counts.columns = ['rating', 'count']
                
                fig = px.bar(rating_counts, x='rating', y='count',
                            labels={'count': '제품 수', 'rating': '평점'},
                            color='rating',
                            color_continuous_scale='oranges')
                
                fig.update_layout(
                    xaxis_title="평점",
                    yaxis_title="제품 수",
                    height=400,
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("로컬 모드에서만 대시보드를 이용할 수 있습니다.")
    
    elif nav_option == "정보":
        st.header("ShopSmart 추천 시스템 정보")
        
        st.markdown("""
        ### 시스템 소개
        
        ShopSmart 추천 시스템은 벡터 데이터베이스를 활용한 시맨틱 검색 및 개인화된 제품 추천 시스템입니다.
        사용자의 자연어 쿼리를 분석하여 의미적으로 가장 유사한 제품을 찾아 추천합니다.
        
        ### 주요 기능
        
        - **시맨틱 검색**: 키워드 매칭이 아닌 의미 기반 검색을 통해 사용자 의도에 맞는 제품 추천
        - **필터링**: 카테고리, 브랜드, 가격, 평점 등 다양한 조건으로 결과 필터링
        - **유사 제품 추천**: 특정 제품과 유사한 다른 제품 추천
        - **직관적인 UI**: 사용자 친화적인 인터페이스로 쉽게 이용 가능
        
        ### 기술 스택
        
        - **벡터 DB**: Chroma DB를 활용한 효율적인 벡터 저장 및 검색
        - **임베딩 모델**: OpenAI API를 활용한 고품질 텍스트 임베딩
        - **백엔드**: FastAPI를 활용한 RESTful API 구현
        - **프론트엔드**: Streamlit을 활용한 직관적인 대시보드
        - **LangChain 통합**: 고급 쿼리 처리 및 대화형 인터페이스
        """)
        
        st.markdown("---")
        st.markdown("© 2025 ShopSmart. All rights reserved.")
    
    # 푸터
    st.markdown("<div class='footer'>© 2025 ShopSmart. All rights reserved.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
