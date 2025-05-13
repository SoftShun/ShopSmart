"""
OpenAI API를 활용한 임베딩 생성 모듈
"""
import os
import time
import logging
from typing import List, Dict, Union, Any
import numpy as np
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam
from dotenv import load_dotenv
import concurrent.futures

# 환경 변수 로드
load_dotenv()

# 로거 설정
logger = logging.getLogger("embedding")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class OpenAIEmbedding:
    """OpenAI API를 활용한 임베딩 생성 클래스"""
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-ada-002"):
        """
        OpenAI 임베딩 클라이언트 초기화
        
        Args:
            api_key (str, optional): OpenAI API 키. 없으면 환경 변수에서 가져옴
            model (str, optional): 사용할 임베딩 모델. 기본값 "text-embedding-ada-002"
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API 키가 필요합니다. api_key 파라미터를 입력하거나 OPENAI_API_KEY 환경 변수를 설정하세요.")
        
        self.model = model
        
        # 최신 OpenAI 라이브러리에 맞게 클라이언트 초기화
        self.client = OpenAI(
            api_key=self.api_key
        )
        
        logger.info(f"OpenAI 임베딩 모델 '{model}' 초기화 완료")
    
    def embed(self, texts: List[str], retry_count: int = 3, retry_delay: float = 1.0, timeout: float = 15.0) -> List[List[float]]:
        """
        텍스트 목록을 임베딩 벡터로 변환
        
        Args:
            texts (List[str]): 임베딩할 텍스트 목록
            retry_count (int, optional): 재시도 횟수. 기본값 3
            retry_delay (float, optional): 재시도 간격(초). 기본값 1.0
            timeout (float, optional): API 요청 타임아웃(초). 기본값 15.0
            
        Returns:
            List[List[float]]: 임베딩 벡터 목록
        """
        if not texts:
            return []
        
        # 배치 크기를 8로 감소 (더 안정적인 처리를 위해)
        batch_size = 8
        
        # 텍스트 길이 체크 및 로깅
        total_tokens = sum(len(text.split()) for text in texts)
        total_chars = sum(len(text) for text in texts)
        mean_tokens = total_tokens / len(texts)
        max_tokens = max(len(text.split()) for text in texts)
        
        logger.info(f"임베딩 요청: {len(texts)}개 텍스트, 평균 {mean_tokens:.1f}토큰/텍스트, 최대 {max_tokens}토큰")
        
        # 결과 저장용 리스트
        all_embeddings = []
        
        # 배치 처리
        for i in range(0, len(texts), batch_size):
            batch = texts[i:min(i+batch_size, len(texts))]
            logger.debug(f"배치 처리 중 ({i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}): {len(batch)}개 텍스트")
            
            # 재시도 로직
            for attempt in range(retry_count):
                try:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            self.client.embeddings.create,
                            model=self.model,
                            input=batch,
                            encoding_format="float"
                        )
                        
                        try:
                            # 타임아웃 적용
                            response = future.result(timeout=timeout)
                            # 성공하면 임베딩 추출
                            batch_embeddings = [item.embedding for item in response.data]
                            all_embeddings.extend(batch_embeddings)
                            break  # 성공하면 재시도 루프 종료
                            
                        except concurrent.futures.TimeoutError:
                            logger.warning(f"임베딩 생성 타임아웃 (시도 {attempt+1}/{retry_count})")
                            if attempt == retry_count - 1:
                                raise TimeoutError(f"OpenAI API 호출 타임아웃 (배치 {i//batch_size + 1})")
                            time.sleep(retry_delay)  # 재시도 전 대기
                            
                except Exception as e:
                    logger.error(f"임베딩 생성 오류 (시도 {attempt+1}/{retry_count}): {str(e)}")
                    
                    if attempt == retry_count - 1:
                        # 마지막 시도에서도 실패하면 예외 발생
                        logger.error(f"임베딩 생성 최종 실패: {str(e)}")
                        raise
                    
                    # 대기 후 재시도
                    time.sleep(retry_delay)
        
        logger.info(f"임베딩 생성 완료: {len(all_embeddings)}개 벡터 생성됨")
        return all_embeddings
