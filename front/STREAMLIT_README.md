# Streamlit을 이용한 GroundingDINO 웹 인터페이스

## 개요
이 프로젝트는 Streamlit을 사용해서 GroundingDINO 모델에 데이터를 보내고 결과를 받는 두 가지 방법을 제공합니다.

## 방법 1: 클라이언트-서버 아키텍처 (권장)

### 장점
- 모델과 웹 인터페이스 분리
- 확장성이 좋음
- 여러 클라이언트가 동시 접속 가능
- API로 다른 애플리케이션에서도 사용 가능

### 실행 방법

1. **의존성 설치**
   ```bash
   pip install -r front/streamlit_requirements.txt
   ```

2. **백엔드 서버 실행**
   ```bash
   uvicorn inference.app:app --host 0.0.0.0 --port 8000
   ```
   또는 Docker Compose를 사용할 경우:
   ```bash
   docker compose up inference
   ```

3. **Streamlit 앱 실행** (새 터미널에서)
   ```bash
   streamlit run front/streamlit_app.py
   ```
   웹 브라우저에서 http://localhost:8501 로 접속합니다.

### 사용 방법
1. 검색할 객체 이름 입력 (예: `person . car . dog`)
2. 필요시 사이드바에서 임계값 및 최대 결과 수 조정
3. "검색하기" 버튼 클릭
4. GroundingDINO가 `data/gallery/`에 있는 이미지에서 해당 객체를 찾고, 박스가 주석된 이미지를 UI에 표시합니다.

## 방법 2: 직접 처리 (올인원)

### 장점
- 단순한 구조
- 별도 서버 불필요
- 설정이 간단함

### 실행 방법

1. **의존성 설치**
   ```bash
   pip install streamlit pillow numpy opencv-python
   ```

2. **Streamlit 앱 실행**
   ```bash
   streamlit run front/streamlit_direct.py
   ```

## API 엔드포인트 (방법 1)

### GET /health
서버 상태 확인

### POST /search
객체 검색 수행

**Parameters:**
- `text`: 검색할 객체 텍스트 (예: `"bread . juice"`)
- `box_threshold`: 박스 검출 임계값 (기본값: 0.25)
- `text_threshold`: 텍스트 매칭 임계값 (기본값: 0.25)
- `limit`: 반환할 최대 이미지 수 (기본값: 6)

**Response:**
```json
{
  "results": [
    {
      "image": "data/gallery/shelf.jpg",
      "detections": [
        {
          "box": [120.4, 230.1, 420.7, 560.0],
          "label": "bread",
          "score": 0.78
        }
      ],
      "annotated_image": {
        "data": "<base64-encoded image>",
        "mime_type": "image/jpeg"
      }
    }
  ]
}
```

## 배포

### Docker 배포
`docker-compose.yml`을 사용하면 `inference`(FastAPI)와 `frontend`(Streamlit)를 한 번에 실행할 수 있습니다.

### 클라우드 배포
- 백엔드: AWS ECS, Google Cloud Run 등
- 프론트엔드: Streamlit Cloud, Heroku 등

## 트러블슈팅

1. **포트 충돌**: 8000, 8501 포트가 사용 중이면 다른 포트 사용
2. **모델 로딩 실패**: GroundingDINO 환경 설정 확인
3. **메모리 부족**: GPU 메모리 또는 RAM 확인

## 개발자 노트

- Streamlit의 `@st.cache_resource`를 사용해서 모델을 한 번만 로드
- 검색 대상 이미지는 기본적으로 `data/gallery/`에 위치하며, `GDINO_SEARCH_DIR` 환경변수로 변경할 수 있습니다.
- 대용량 이미지 처리시 메모리 최적화 필요
- 실시간 처리를 위해서는 WebSocket 연결 고려
