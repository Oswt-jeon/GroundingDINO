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
   python backend_server.py
   ```
   서버가 http://localhost:8000 에서 실행됩니다.

3. **Streamlit 앱 실행** (새 터미널에서)
   ```bash
   streamlit run front/streamlit_app.py
   ```
   웹 브라우저에서 http://localhost:8501 로 접속합니다.

### 사용 방법
1. 검색할 객체 이름 입력 (예: "person . car . dog")
2. 분석할 이미지 업로드
3. 필요시 사이드바에서 임계값 조정
4. "검색하기" 버튼 클릭

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
- `image`: 업로드할 이미지 파일
- `query`: 검색할 객체 텍스트
- `box_threshold`: 박스 검출 임계값 (기본값: 0.35)
- `text_threshold`: 텍스트 매칭 임계값 (기본값: 0.25)

**Response:**
```json
{
    "status": "success",
    "query": "person . car",
    "detections": [
        {
            "label": "person",
            "confidence": 0.85,
            "bbox": [100, 100, 200, 200]
        }
    ],
    "detection_count": 1
}
```

## 실제 GroundingDINO 통합

현재 코드는 더미 데이터를 반환합니다. 실제 GroundingDINO를 사용하려면:

1. `backend_server.py`의 TODO 부분을 실제 GroundingDINO 코드로 교체
2. `front/streamlit_direct.py`의 TODO 부분을 실제 모델 로딩/추론 코드로 교체

```python
# 실제 구현 예시
from groundingdino.util.inference import load_model, predict, annotate

# 모델 로드
model = load_model("config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")

# 추론
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=text_prompt,
    box_threshold=box_threshold,
    text_threshold=text_threshold
)

# 결과 시각화
annotated_image = annotate(image_source=image, boxes=boxes, logits=logits, phrases=phrases)
```

## 배포

### Docker 배포
1. 백엔드 서버를 Docker로 패키징
2. Streamlit 앱을 별도 컨테이너로 실행
3. docker-compose로 통합 관리

### 클라우드 배포
- 백엔드: AWS ECS, Google Cloud Run 등
- 프론트엔드: Streamlit Cloud, Heroku 등

## 트러블슈팅

1. **포트 충돌**: 8000, 8501 포트가 사용 중이면 다른 포트 사용
2. **모델 로딩 실패**: GroundingDINO 환경 설정 확인
3. **메모리 부족**: GPU 메모리 또는 RAM 확인

## 개발자 노트

- Streamlit의 `@st.cache_resource`를 사용해서 모델을 한 번만 로드
- 대용량 이미지 처리시 메모리 최적화 필요
- 실시간 처리를 위해서는 WebSocket 연결 고려
