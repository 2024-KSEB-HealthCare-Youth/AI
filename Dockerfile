# Python 3.9 공식 이미지를 기반으로 설정
FROM python:3.10.4

# 작업 디렉토리 설정
WORKDIR /

# 모든 프로젝트 파일을 작업 디렉토리로 복사
COPY .. .

# 필요한 Python 라이브러리 설치
RUN pip install --no-cache-dir -r requirements.txt

# 컨테이너 실행 시 실행할 명령
CMD ["python", "/recommendation-service/app/app.py"]
