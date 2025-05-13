"""
유틸리티 함수 모음
"""
import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from functools import wraps
from datetime import datetime

# 로깅 설정
def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    로거 설정 함수
    
    Args:
        name (str): 로거 이름
        log_file (str, optional): 로그 파일 경로. 없으면 콘솔만 사용
        level (int, optional): 로그 레벨. 기본값 logging.INFO
    
    Returns:
        logging.Logger: 설정된 로거
    """
    # 로그 디렉토리 생성
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    # 로거 설정
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 핸들러가 이미 있는지 확인
    if not logger.handlers:
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # 파일 핸들러 (선택사항)
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
    
    return logger

# 성능 측정 데코레이터
def timer(logger=None):
    """
    함수 실행 시간을 측정하는 데코레이터
    
    Args:
        logger (logging.Logger, optional): 로깅에 사용할 로거
    
    Returns:
        function: 데코레이터 함수
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            
            # 로그 출력
            log_message = f"{func.__name__} 실행 시간: {elapsed_time:.4f}초"
            if logger:
                logger.info(log_message)
            else:
                print(log_message)
                
            return result
        return wrapper
    return decorator

# JSON 유틸리티 함수
def load_json(file_path: str) -> Any:
    """
    JSON 파일 로드
    
    Args:
        file_path (str): JSON 파일 경로
    
    Returns:
        Any: 로드된 JSON 데이터
    
    Raises:
        FileNotFoundError: 파일을 찾을 수 없는 경우
        json.JSONDecodeError: JSON 파싱 오류
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """
    데이터를 JSON 파일로 저장
    
    Args:
        data (Any): 저장할 데이터
        file_path (str): 저장할 파일 경로
        indent (int, optional): JSON 들여쓰기. 기본값 2
    """
    # 디렉토리 생성
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

# 데이터 변환 유틸리티
def format_price(price: Union[int, float, str]) -> str:
    """
    가격 포맷팅
    
    Args:
        price (Union[int, float, str]): 가격
    
    Returns:
        str: 포맷팅된 가격 문자열 (예: ₩10,000)
    """
    if isinstance(price, str):
        try:
            price = int(float(price))
        except ValueError:
            return price
    
    return f"₩{int(price):,}"

def format_date(date_str: str, input_format: str = "%Y-%m-%d", output_format: str = "%Y년 %m월 %d일") -> str:
    """
    날짜 포맷팅
    
    Args:
        date_str (str): 날짜 문자열
        input_format (str, optional): 입력 날짜 형식. 기본값 "%Y-%m-%d"
        output_format (str, optional): 출력 날짜 형식. 기본값 "%Y년 %m월 %d일"
    
    Returns:
        str: 포맷팅된 날짜 문자열
    """
    try:
        date_obj = datetime.strptime(date_str, input_format)
        return date_obj.strftime(output_format)
    except ValueError:
        return date_str
