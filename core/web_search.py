"""
LangChain을 활용한 웹 검색 모듈
"""
import os
import logging
import json
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 환경 변수 로드
load_dotenv()

# 로거 설정
logger = logging.getLogger("web_search")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class WebSearchEngine:
    """
    웹 검색 엔진 - Tavily API 및 OpenAI를 직접 호출
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        웹 검색 엔진 초기화
        
        Args:
            api_key (str, optional): OpenAI API 키. 없으면 환경 변수에서 가져옴
        """
        self.openai_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API 키가 필요합니다. api_key 파라미터를 입력하거나 OPENAI_API_KEY 환경 변수를 설정하세요.")
        
        # Tavily API 키 확인
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            raise ValueError("Tavily API 키가 필요합니다. TAVILY_API_KEY 환경 변수를 설정하세요.")
        
        # LLM 설정
        self.llm = ChatOpenAI(api_key=self.openai_api_key, model="gpt-4o", temperature=0)
        
        logger.info("웹 검색 엔진 초기화 완료")
    
    def tavily_search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Tavily API를 직접 호출하여 웹 검색 수행
        
        Args:
            query (str): 검색 쿼리
            max_results (int): 최대 결과 수
            
        Returns:
            List[Dict[str, Any]]: 검색 결과 목록
        """
        try:
            url = "https://api.tavily.com/search"
            headers = {
                "content-type": "application/json",
                "Authorization": f"Bearer {self.tavily_api_key}"
            }
            payload = {
                "query": query,
                "max_results": max_results,
                "include_domains": [],
                "exclude_domains": []
            }
            
            # 5초 타임아웃 추가
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response_json = response.json()
            
            if "results" in response_json:
                return response_json["results"]
            return []
            
        except requests.exceptions.Timeout:
            logger.error("Tavily API 요청 시간 초과")
            raise TimeoutError("Tavily API 요청 시간 초과")
        except Exception as e:
            logger.error(f"Tavily 검색 중 오류 발생: {str(e)}")
            return []
    
    def summarize_results(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        검색 결과 요약
        
        Args:
            query (str): 원본 쿼리
            search_results (List[Dict[str, Any]]): 검색 결과 목록
            
        Returns:
            str: 요약된 텍스트
        """
        try:
            # 검색 결과가 없으면 메시지 반환
            if not search_results:
                return "검색 결과가 없습니다."
            
            # 검색 결과 텍스트 구성 - 간소화
            results_text = ""
            for i, result in enumerate(search_results[:3], 1):  # 결과 수 제한
                results_text += f"{i}. {result.get('title', '제목 없음')}\n"
                results_text += f"   내용: {result.get('content', '내용 없음')[:300]}...\n\n"
            
            # OpenAI API로 요약 생성
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            prompt = f"""검색 결과:
            {results_text}
            
            질문: {query}
            
            위 검색 결과를 다음 형식으로 정리해주세요:
            1. 가장 관련성 높은 핵심 정보 요약 (2-3문장)
            2. 제품 관련 특징이나 장점 나열 (불릿 포인트)
            3. 가격 범위나 비교 정보 (있을 경우)
            4. 최신 트렌드나 인기 브랜드 언급 (있을 경우)
            
            사실에 기반한 정보만 제공하고, 검색 결과에 없는 내용은 추가하지 마세요.
            """
            
            # 5초 타임아웃 추가
            try:
                import time
                start = time.time()
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "간결한 요약만 제공하세요."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=400,
                    request_timeout=5  # 5초 타임아웃
                )
                
                logger.info(f"요약 생성 시간: {time.time() - start:.2f}초")
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.error(f"OpenAI API 호출 오류: {str(e)}")
                # 기본 요약 반환
                return f"검색 결과: {search_results[0].get('title', '제목 없음')} - {search_results[0].get('content', '내용 없음')[:200]}..."
        
        except Exception as e:
            logger.error(f"결과 요약 중 오류 발생: {str(e)}")
            return f"결과를 요약할 수 없습니다: {str(e)}"
    
    def search(self, query: str) -> Dict[str, Any]:
        """
        웹 검색 수행 및 결과 요약
        
        Args:
            query (str): 검색 쿼리
            
        Returns:
            Dict[str, Any]: 검색 결과와 요약
        """
        try:
            logger.info(f"웹 검색 시작: '{query}'")
            
            # Tavily 검색 수행
            try:
                search_results = self.tavily_search(query, max_results=3)  # 결과 수 제한
            except TimeoutError:
                return {
                    "query": query,
                    "response": "웹 검색 시간이 초과되었습니다.",
                    "success": False
                }
            
            # 검색 결과가 없는 경우
            if not search_results:
                return {
                    "query": query,
                    "response": "검색 결과가 없습니다. 다른 검색어를 사용해보세요.",
                    "success": True
                }
            
            # 검색 결과 요약
            summary = self.summarize_results(query, search_results)
            
            logger.info(f"웹 검색 완료: '{query}'")
            return {
                "query": query,
                "response": summary,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"웹 검색 오류: {str(e)}")
            return {
                "query": query,
                "response": "웹 검색 처리 중 오류가 발생했습니다.",
                "success": False
            } 