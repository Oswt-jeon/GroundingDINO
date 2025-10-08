from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image
import io
import base64
import json

app = FastAPI(title="GroundingDINO API Server")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "GroundingDINO API Server is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "GroundingDINO API"}

@app.post("/search")
async def search_objects(
    image: UploadFile = File(...),
    query: str = Form(...),
    box_threshold: float = Form(0.35),
    text_threshold: float = Form(0.25)
):
    """
    객체 검색 API
    """
    try:
        # 이미지 읽기
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        # 이미지를 numpy 배열로 변환
        image_array = np.array(pil_image)
        
        # TODO: 여기에 실제 GroundingDINO 모델 추론 코드를 구현
        # 현재는 더미 데이터를 반환
        
        # 더미 검출 결과
        dummy_detections = [
            {
                "label": query.split('.')[0].strip() if '.' in query else query.strip(),
                "confidence": 0.85,
                "bbox": [100, 100, 200, 200]
            },
            {
                "label": query.split('.')[1].strip() if len(query.split('.')) > 1 else "object",
                "confidence": 0.72,
                "bbox": [300, 150, 400, 250]
            }
        ]
        
        # 결과 반환
        result = {
            "status": "success",
            "query": query,
            "parameters": {
                "box_threshold": box_threshold,
                "text_threshold": text_threshold
            },
            "image_info": {
                "filename": image.filename,
                "size": image_array.shape,
                "format": pil_image.format
            },
            "detections": dummy_detections,
            "detection_count": len(dummy_detections)
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류가 발생했습니다: {str(e)}")

# 실제 GroundingDINO 모델을 사용하는 함수 (예시)
def run_grounding_dino(image_array, text_prompt, box_threshold=0.35, text_threshold=0.25):
    """
    실제 GroundingDINO 모델 추론을 수행하는 함수
    """
    # TODO: 실제 모델 로딩 및 추론 코드 구현
    # from groundingdino.util.inference import load_model, predict
    # model = load_model(config_path, checkpoint_path)
    # boxes, logits, phrases = predict(model, image_array, text_prompt, box_threshold, text_threshold)
    
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
