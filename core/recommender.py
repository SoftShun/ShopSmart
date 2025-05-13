"""
Chroma DB를 활용한 시맨틱 검색 및 추천 시스템 구현
"""
import os
import json
import re
import math
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

from .embedding import OpenAIEmbedding

# 로거 설정
logger = logging.getLogger("recommender")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class ProductRecommendationSystem:
    """
    Chroma DB를 활용한 제품 추천 시스템
    """
    
    def __init__(self, json_path: str, persist_dir: str = "chroma_db"):
        """
        추천 시스템 초기화
        
        Args:
            json_path (str): 제품 데이터가 들어있는 JSON 파일 경로
            persist_dir (str, optional): Chroma DB 데이터 저장 디렉토리. 기본값 "chroma_db"
        """
        self.json_path = json_path
        self.persist_dir = persist_dir
        self.products = None
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        
        # 디렉토리 생성
        os.makedirs(persist_dir, exist_ok=True)
        
        logger.info(f"제품 추천 시스템 초기화 (데이터: {json_path}, DB: {persist_dir})")
    
    def load_data(self) -> bool:
        """
        JSON 파일에서 제품 데이터 로드
        
        Returns:
            bool: 성공 여부
        """
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.products = json.load(f)
            logger.info(f"총 {len(self.products)}개의 제품 데이터를 로드했습니다.")
            return True
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
            return False
    
    def preprocess_data(self) -> None:
        """각 제품에 대해 결합된 텍스트 생성"""
        logger.info("데이터 전처리 중...")
        
        for product in tqdm(self.products, desc="텍스트 전처리"):
            # feature_tags 리스트를 문자열로 변환
            feature_tags_str = " ".join(product.get("feature_tags", []))
            
            # 모든 관련 필드를 결합하여 더 풍부한 텍스트 생성
            combined_texts = [
                product['product_name'],
                product['description'],
                f"카테고리: {product['category']}",
                f"서브카테고리: {product['subcategory']}",
                f"브랜드: {product['brand']}",
                feature_tags_str
            ]
            
            # 추가 필드가 있는 경우 포함 (태그, 색상 등)
            if 'tags' in product and product['tags']:
                if isinstance(product['tags'], list):
                    combined_texts.append("태그: " + " ".join(product['tags']))
                else:
                    combined_texts.append(f"태그: {product['tags']}")
            
            if 'color' in product and product['color']:
                combined_texts.append(f"색상: {product['color']}")
                
            if 'material' in product and product['material']:
                combined_texts.append(f"소재: {product['material']}")
            
            # 결합된 텍스트 생성
            product["combined_text"] = " ".join(text for text in combined_texts if text)
            
            # 디버그용 로깅 (첫 몇 개 제품만)
            if product.get("product_id", "").endswith("0001"):
                logger.debug(f"전처리된 텍스트 예시 [{product['product_id']}]: {product['combined_text'][:200]}...")
        
        logger.info("데이터 전처리 완료!")
    
    def setup_embedding_model(self, api_key: Optional[str] = None) -> None:
        """
        OpenAI 임베딩 모델 설정
        
        Args:
            api_key (str, optional): OpenAI API 키. 없으면 환경 변수에서 가져옴
        """
        self.embedding_model = OpenAIEmbedding(api_key=api_key)
        logger.info("OpenAI 임베딩 모델 설정 완료")
    
    def setup_vector_db(self, force_recreate: bool = True) -> None:
        """Chroma 벡터 DB 설정"""
        logger.info("벡터 DB 설정 중...")
        
        # Chroma 클라이언트 생성
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        
        # 강제 재생성 옵션이 있을 경우 기존 컬렉션 삭제
        if force_recreate:
            try:
                self.chroma_client.delete_collection("product_collection")
                logger.info("기존 컬렉션 삭제됨 - 새로운 임베딩으로 다시 생성합니다.")
            except Exception as e:
                logger.info(f"컬렉션 삭제 시도 중 오류 (무시): {str(e)}")
        
        # 컬렉션이 이미 존재하는 경우
        try:
            self.collection = self.chroma_client.get_collection("product_collection")
            count = self.collection.count()
            logger.info(f"기존 컬렉션 로드됨 - {count}개의 항목 포함")
            
            # 강제 재생성이 아니고 이미 모든 데이터가 있으면 처리 건너뛰기
            if not force_recreate and count == len(self.products):
                logger.info("모든 데이터가 이미 벡터 DB에 있습니다.")
                return
        except Exception as e:
            # 컬렉션이 없으면 새로 생성
            logger.info(f"컬렉션 없음 또는 오류 발생: {str(e)}")
            try:
                self.collection = self.chroma_client.create_collection(
                    name="product_collection",
                    metadata={"hnsw:space": "cosine"}  # 코사인 유사도 사용
                )
                logger.info("새 컬렉션 생성됨")
            except Exception as e:
                logger.error(f"컬렉션 생성 오류: {str(e)}")
                raise
        
        # 제품 데이터를 벡터 DB에 저장
        batch_size = 50  # 한 번에 처리할 제품 수
        
        # 임베딩 모델 확인
        if self.embedding_model is None:
            logger.error("임베딩 모델이 설정되지 않았습니다. setup_embedding_model()을 먼저 호출하세요.")
            return
        
        total_batches = (len(self.products) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(self.products), batch_size), desc="벡터 DB 구축", total=total_batches):
            batch = self.products[i:i+batch_size]
            
            ids = [product["product_id"] for product in batch]
            
            # 이미 존재하는 ID 확인
            try:
                existing_ids = self.collection.get(ids=ids, include=[])["ids"]
                new_indices = [i for i, id in enumerate(ids) if id not in existing_ids]
                
                if not new_indices:
                    logger.debug(f"배치의 모든 항목이 이미 존재합니다. 건너뜁니다.")
                    continue
                
                # 새 항목만 추가
                new_ids = [ids[i] for i in new_indices]
                new_batch = [batch[i] for i in new_indices]
                
                # 텍스트 추출 및 임베딩 생성
                texts = [product["combined_text"] for product in new_batch]
                try:
                    embeddings = self.embedding_model.embed(texts)
                except Exception as e:
                    logger.error(f"임베딩 생성 오류: {str(e)}")
                    raise
                
                # 메타데이터 구성
                metadatas = []
                for product in new_batch:
                    metadata = {
                        "product_name": product["product_name"],
                        "category": product["category"],
                        "subcategory": product["subcategory"],
                        "brand": product["brand"],
                        "price": str(product["price"]),
                        "rating": str(product["rating"]),
                        "review_count": str(product["review_count"])
                    }
                    metadatas.append(metadata)
                
                # Chroma DB에 추가
                self.collection.add(
                    ids=new_ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=[product["combined_text"] for product in new_batch]
                )
                
                logger.debug(f"배치 {i//batch_size + 1}/{total_batches}: {len(new_ids)}개 항목 추가됨")
                
            except Exception as e:
                logger.error(f"배치 {i//batch_size + 1} 처리 중 오류: {str(e)}")
                continue
        
        count = self.collection.count()
        logger.info(f"벡터 DB 설정 완료! 총 {count}개 항목 저장됨")
    
    def parse_filter(self, query: str) -> tuple:
        """
        쿼리에서 필터 추출
        
        Args:
            query (str): 사용자 쿼리
            
        Returns:
            tuple: (필터가 제거된 쿼리, 필터 딕셔너리)
        """
        filters = {}
        clean_query = query
        
        # 카테고리 필터 추출 (예: "category: 패션")
        category_match = re.search(r'category:\s*([^,]+)', query)
        if category_match:
            filters["category"] = category_match.group(1).strip()
            clean_query = re.sub(r'category:\s*([^,]+)', '', clean_query).strip()
        
        # 브랜드 필터 추출 (예: "brand: 나이키")
        brand_match = re.search(r'brand:\s*([^,]+)', query)
        if brand_match:
            filters["brand"] = brand_match.group(1).strip()
            clean_query = re.sub(r'brand:\s*([^,]+)', '', clean_query).strip()
        
        # 가격 범위 필터 추출 (예: "price: 10000-50000")
        price_match = re.search(r'price:\s*(\d+)-(\d+)', query)
        if price_match:
            min_price = int(price_match.group(1))
            max_price = int(price_match.group(2))
            filters["price_range"] = (min_price, max_price)
            clean_query = re.sub(r'price:\s*\d+-\d+', '', clean_query).strip()
        
        # 최소 평점 필터 (예: "rating: 4.0")
        rating_match = re.search(r'rating:\s*([\d.]+)', query)
        if rating_match:
            min_rating = float(rating_match.group(1))
            filters["min_rating"] = min_rating
            clean_query = re.sub(r'rating:\s*[\d.]+', '', clean_query).strip()
        
        return clean_query, filters
    
    def search(self, query: str, top_k: int = 5, category: Optional[str] = None, 
            brand: Optional[str] = None, price_range: Optional[Tuple[int, int]] = None, 
            min_rating: Optional[float] = None, web_search_result: Optional[str] = None) -> Dict[str, Any]:
        """
        쿼리에 기반한 제품 검색
        
        Args:
            query (str): 검색 쿼리
            top_k (int, optional): 반환할 상위 결과 수. 기본값 5
            category (str, optional): 카테고리 필터
            brand (str, optional): 브랜드 필터
            price_range (Tuple[int, int], optional): 가격 범위 (최소, 최대)
            min_rating (float, optional): 최소 평점
        
        Returns:
            Dict[str, Any]: 검색 결과와 추천 제품 목록
        """
        # 쿼리 시작 시간
        start_time = time.time()
        
        # 필터 관련 추가 정보를 쿼리 문자열에서 제거
        clean_query, _ = self.parse_filter(query)
        
        # 파라미터로 전달된 필터 설정
        filters = {}
        if category:
            filters["category"] = category
        if brand:
            filters["brand"] = brand
        if price_range:
            filters["price_range"] = price_range
        if min_rating is not None:
            filters["min_rating"] = min_rating
        
        logger.info(f"쿼리: '{clean_query}', 필터: {filters}")
        logger.info(f"검색 요청 세부정보 - 최대 결과 수: {top_k}, 필터 개수: {len(filters)}")
        
        # 유효성 검사
        if not clean_query.strip():
            return {
                "query": query,
                "message": "검색어를 입력해주세요.",
                "recommendations": []
            }
        
        # 임베딩 모델 확인
        if self.embedding_model is None:
            logger.error("임베딩 모델이 설정되지 않았습니다. setup_embedding_model()을 먼저 호출하세요.")
            return {
                "query": clean_query,
                "message": "시스템 오류: 임베딩 모델이 설정되지 않았습니다.",
                "recommendations": []
            }
        
        # 벡터 DB 확인
        if self.collection is None:
            logger.error("벡터 DB가 설정되지 않았습니다. setup_vector_db()를 먼저 호출하세요.")
            return {
                "query": clean_query,
                "message": "시스템 오류: 벡터 DB가 설정되지 않았습니다.",
                "recommendations": []
            }
        
        try:
            # 쿼리 임베딩 생성 (타임아웃 10초 설정)
            try:
                query_embedding = self.embedding_model.embed([clean_query], timeout=10)[0]
            except TimeoutError:
                logger.error("쿼리 임베딩 생성 시간 초과")
                return {
                    "query": clean_query,
                    "message": "검색 시간이 초과되었습니다. 다시 시도해주세요.",
                    "processing_time": f"{time.time() - start_time:.2f}초",
                    "recommendations": []
                }
            
            # 단순화된 검색 로직
            product_results = []
            
            # 1. ChromaDB 필터 설정
            where_clause = {}
            if category:
                where_clause["category"] = {"$eq": category}
            if brand:
                where_clause["brand"] = {"$eq": brand}
            
            # 로그 기록
            logger.info(f"ChromaDB 쿼리 시작: {clean_query}")
            
            # 2. 타임아웃이 있는 쿼리 실행 - 더 적은 결과를 가져와 처리 부담 감소
            try:
                with_filter = bool(where_clause)
                fetch_count = min(top_k * 3, 50)  # 최대 50개로 제한
                
                if with_filter:
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=fetch_count,
                        where=where_clause
                    )
                else:
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=fetch_count
                    )
                
                logger.info(f"ChromaDB 쿼리 완료: {len(results['ids'][0])}개 결과 (필터 적용: {with_filter})")
            except Exception as e:
                logger.error(f"ChromaDB 쿼리 오류: {str(e)}")
                return {
                    "query": clean_query,
                    "message": f"검색 처리 중 오류가 발생했습니다: {str(e)}",
                    "processing_time": f"{time.time() - start_time:.2f}초",
                    "recommendations": []
                }
            
            # 결과가 없는 경우
            if len(results["ids"][0]) == 0:
                logger.warning("검색 결과 없음")
                return {
                    "query": clean_query,
                    "message": "검색 결과가 없습니다. 다른 검색어나 필터를 시도해보세요.",
                    "processing_time": f"{time.time() - start_time:.2f}초",
                    "recommendations": []
                }
            
            # 3. 결과 후처리
            for i in range(len(results["ids"][0])):
                product_id = results["ids"][0][i]
                metadata = results["metadatas"][0][i]
                
                # 원본 제품 데이터 가져오기
                original_product = next((p for p in self.products if p["product_id"] == product_id), None)
                
                if not original_product:
                    continue
                
                # 가격 필터 적용 (후처리)
                if price_range and price_range[0] <= price_range[1]:
                    price = int(original_product["price"])
                    if price < price_range[0] or price > price_range[1]:
                        continue
                
                # 평점 필터 적용 (후처리)
                if min_rating is not None:
                    rating = float(metadata["rating"])
                    if rating < min_rating:
                        continue
                
                # 유사도 점수 계산
                similarity_score = 1 - results["distances"][0][i]
                
                # 최종 점수 계산 (단순화)
                rating = float(metadata["rating"])
                review_count = int(metadata["review_count"])
                
                # 간소화된 최종 점수 - 유사도 80%, 평점 20%
                final_score = similarity_score * 0.8 + (rating / 5.0) * 0.2
                
                # 결과 추가
                product_results.append({
                    "product_id": product_id,
                    "product_name": metadata["product_name"],
                    "description": original_product["description"],
                    "price": int(original_product["price"]),
                    "rating": rating,
                    "review_count": review_count,
                    "category": metadata["category"],
                    "subcategory": metadata["subcategory"],
                    "brand": metadata["brand"],
                    "feature_tags": original_product.get("feature_tags", []),
                    "similarity_score": similarity_score,
                    "final_score": final_score
                })
            
            # 웹 검색 결과가 있으면 키워드 추출하여 가중치 적용
            if web_search_result:
                logger.info("웹 검색 결과 활용하여 점수 조정")
                keywords = self._extract_keywords_from_web_result(web_search_result)
                for i, product in enumerate(product_results):
                    keyword_match_score = self._calculate_keyword_match(product, keywords)
                    # 최종 점수에 키워드 매치 점수 반영 (10%)
                    product_results[i]["final_score"] = product_results[i]["final_score"] * 0.9 + keyword_match_score * 0.1
                    product_results[i]["web_enhanced"] = True  # 웹 검색으로 개선된 결과임을 표시
            
            # 결과가 없는 경우 처리
            if not product_results:
                logger.warning("필터링 후 검색 결과 없음")
                return {
                    "query": clean_query,
                    "message": "지정한 필터에 맞는 검색 결과가 없습니다. 필터를 완화하거나 다른 검색어를 시도해보세요.",
                    "processing_time": f"{time.time() - start_time:.2f}초",
                    "recommendations": []
                }
            
            # 최종 점수 기준으로 정렬
            product_results.sort(key=lambda x: x["final_score"], reverse=True)
            
            # 상위 N개만 반환
            top_results = product_results[:top_k]
            
            # 간소화: 유사성 이유 분석 건너뛰기
            
            # 처리 시간 계산
            processing_time = f"{time.time() - start_time:.2f}초"
            
            logger.info(f"검색 완료. 처리 시간: {processing_time}, 결과 수: {len(top_results)}")
            
            # 결과 반환
            return {
                "query": clean_query,
                "processing_time": processing_time,
                "recommendations": top_results
            }
            
        except Exception as e:
            logger.error(f"검색 중 오류 발생: {str(e)}", exc_info=True)
            return {
                "query": clean_query,
                "message": f"검색 중 오류가 발생했습니다: {str(e)}",
                "processing_time": f"{time.time() - start_time:.2f}초",
                "recommendations": []
            }
    
    def get_similar_products(self, product_id: str, top_k: int = 5) -> Dict[str, Any]:
        """
        특정 제품과 유사한 제품 추천
        
        Args:
            product_id (str): 제품 ID
            top_k (int, optional): 반환할 상위 결과 수. 기본값 5
        
        Returns:
            Dict[str, Any]: 유사한 제품 목록
        """
        # 제품 ID 유효성 검사
        product = next((p for p in self.products if p["product_id"] == product_id), None)
        if not product:
            return {
                "product_id": product_id,
                "message": f"제품 ID '{product_id}'를 찾을 수 없습니다.",
                "similar_products": []
            }
        
        # 벡터 DB 확인
        if self.collection is None:
            logger.error("벡터 DB가 설정되지 않았습니다. setup_vector_db()를 먼저 호출하세요.")
            return {
                "product_id": product_id,
                "message": "시스템 오류: 벡터 DB가 설정되지 않았습니다.",
                "similar_products": []
            }
        
        try:
            # 제품 임베딩 가져오기
            result = self.collection.get(ids=[product_id], include=["embeddings"])
            
            if not result["ids"]:
                return {
                    "product_id": product_id,
                    "message": f"제품 ID '{product_id}'의 임베딩을 찾을 수 없습니다.",
                    "similar_products": []
                }
            
            product_embedding = result["embeddings"][0]
            
            # 같은 제품 제외하고 유사한 제품 검색
            results = self.collection.query(
                query_embeddings=[product_embedding],
                n_results=top_k + 1,  # 자기 자신도 포함될 수 있으므로 +1
                where={"$id": {"$ne": product_id}}  # 자기 자신 제외
            )
            
            # 결과가 없는 경우
            if len(results["ids"][0]) == 0:
                return {
                    "product_id": product_id,
                    "product_name": product["product_name"],
                    "message": "유사한 제품을 찾을 수 없습니다.",
                    "similar_products": []
                }
            
            # 결과 처리
            similar_products = []
            for i in range(len(results["ids"][0])):
                similar_id = results["ids"][0][i]
                metadata = results["metadatas"][0][i]
                
                # 원본 제품 데이터 가져오기
                similar_product = next((p for p in self.products if p["product_id"] == similar_id), None)
                
                if similar_product:
                    # 유사도 점수 계산
                    similarity_score = 1 - results["distances"][0][i]
                    
                    similar_products.append({
                        "product_id": similar_id,
                        "product_name": metadata["product_name"],
                        "description": similar_product["description"],
                        "price": int(similar_product["price"]),
                        "rating": float(metadata["rating"]),
                        "category": metadata["category"],
                        "subcategory": metadata["subcategory"],
                        "brand": metadata["brand"],
                        "feature_tags": similar_product.get("feature_tags", []),
                        "similarity_score": similarity_score,
                        "similarity_reason": self._analyze_similarity(product, similar_product)
                    })
            
            return {
                "product_id": product_id,
                "product_name": product["product_name"],
                "similar_products": similar_products[:top_k]
            }
            
        except Exception as e:
            logger.error(f"유사 제품 검색 중 오류 발생: {str(e)}", exc_info=True)
            return {
                "product_id": product_id,
                "product_name": product["product_name"] if product else "",
                "message": f"유사 제품 검색 중 오류 발생: {str(e)}",
                "similar_products": []
            }

    def _extract_keywords_from_web_result(self, web_search_result: str) -> List[str]:
        """웹 검색 결과에서 키워드 추출"""
        try:
            # 간단한 방식의 키워드 추출 (실제로는 NLP 라이브러리를 사용하면 더 좋음)
            # 불용어 목록
            stopwords = {"이", "그", "저", "것", "및", "에", "를", "의", "가", "은", "는", "이다", "있다", "하다", "있는", "등", "또한"}
            
            # 텍스트 정제
            text = re.sub(r'[^\w\s가-힣]', ' ', web_search_result.lower())
            words = text.split()
            
            # 불용어 제거 및 단어 길이 필터링
            keywords = [word for word in words if word not in stopwords and len(word) > 1]
            
            # 빈도수 기반 상위 키워드 추출
            from collections import Counter
            counter = Counter(keywords)
            top_keywords = [word for word, _ in counter.most_common(15)]  # 상위 15개 키워드
            
            logger.info(f"웹 검색 결과에서 추출된 키워드: {top_keywords}")
            return top_keywords
        except Exception as e:
            logger.error(f"키워드 추출 중 오류: {str(e)}")
            return []

    def _calculate_keyword_match(self, product: Dict[str, Any], keywords: List[str]) -> float:
        """제품 정보와 키워드 매칭 점수 계산"""
        try:
            if not keywords:
                return 0.0
            
            # 제품 관련 텍스트 결합
            product_text = " ".join([
                product.get("product_name", ""),
                product.get("description", ""),
                " ".join(product.get("feature_tags", [])),
                product.get("category", ""),
                product.get("subcategory", ""),
                product.get("brand", "")
            ]).lower()
            
            # 일치하는 키워드 수 계산
            matches = sum(1 for keyword in keywords if keyword.lower() in product_text)
            
            # 정규화된 점수 (0.0-1.0 범위)
            score = min(matches / len(keywords), 1.0)
            
            return score
        except Exception as e:
            logger.error(f"키워드 매칭 점수 계산 중 오류: {str(e)}")
            return 0.0

    def _analyze_similarity(self, original_product: Dict[str, Any], similar_product: Dict[str, Any]) -> str:
        """
        두 제품 간의 유사성을 분석하여 설명 제공
        
        Args:
            original_product (Dict[str, Any]): 원본 제품 데이터
            similar_product (Dict[str, Any]): 유사한 제품 데이터
        
        Returns:
            str: 유사성에 대한 설명
        """
        reasons = []
        
        # 같은 브랜드 확인
        if original_product["brand"] == similar_product["brand"]:
            reasons.append(f"같은 브랜드({original_product['brand']})")
        
        # 같은 카테고리 확인
        if original_product["category"] == similar_product["category"]:
            reasons.append(f"같은 카테고리({original_product['category']})")
            
            # 같은 서브카테고리 확인
            if original_product["subcategory"] == similar_product["subcategory"]:
                reasons.append(f"같은 서브카테고리({original_product['subcategory']})")
        
        # 가격대 유사성 확인
        price_diff = abs(original_product["price"] - similar_product["price"])
        price_ratio = price_diff / max(original_product["price"], 1) * 100
        
        if price_ratio < 10:
            reasons.append("매우 유사한 가격대")
        elif price_ratio < 30:
            reasons.append("유사한 가격대")
        
        # 특징 태그 유사성 확인
        original_tags = set(original_product.get("feature_tags", []))
        similar_tags = set(similar_product.get("feature_tags", []))
        
        common_tags = original_tags.intersection(similar_tags)
        if common_tags:
            if len(common_tags) > 2:
                reasons.append(f"여러 공통 특징({', '.join(list(common_tags)[:2])} 외 {len(common_tags)-2}개)")
            else:
                reasons.append(f"공통 특징({', '.join(common_tags)})")
        
        # 결과 형식화
        if reasons:
            return ", ".join(reasons)
        else:
            return "텍스트 기반 유사성"
