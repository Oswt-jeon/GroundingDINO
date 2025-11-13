from __future__ import annotations

import os
import cv2
import time
import glob
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from api.dependencies import get_detection_manager_dependency, get_db_session_dependency
from api.schemas.detections import (
    AnnotatedImageResponse,
    DetectItem,
    ImageSearchResult,
    SearchRequest,
    SearchResponse,
)
from src.db.database import Database
from src.repositories.owlv2_examples import OwlV2ExampleRepository
from src.services.manager import DetectionServiceManager
from src.utils.file_io import encode_file_to_base64


def enhance_image_quality(image):
    """
    이미지 품질을 개선하는 함수
    """
    # 1. 노이즈 제거 (가우시안 필터)
    denoised = cv2.GaussianBlur(image, (3, 3), 0)
    
    # 2. 샤프닝 필터 적용
    kernel_sharpening = np.array([[-1,-1,-1],
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel_sharpening)
    
    # 3. 밝기와 대비 자동 조정 (CLAHE)
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    
    # LAB 채널 재결합
    lab = cv2.merge((l_channel, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 4. 색상 보정 (선택적 적용)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=10)  # alpha: 대비, beta: 밝기
    
    return enhanced


router = APIRouter()


def capture_webcam_images(save_directory: str = "data/gallery") -> List[str]:
    """
    연결된 모든 웹캠에서 이미지를 캡처하여 지정된 디렉토리에 저장합니다.
    실제 웹캠이 없는 경우 테스트 이미지를 생성합니다.
    
    Args:
        save_directory: 이미지를 저장할 디렉토리 경로
        
    Returns:
        저장된 이미지 파일 경로들의 리스트
    """
    # 저장 디렉토리를 절대 경로로 변환하고 생성
    if not os.path.isabs(save_directory):
        save_directory = os.path.join(os.getcwd(), save_directory)
    
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 웹캠 이미지 저장 경로: {save_path.absolute()}")
    
    # 기존 웹캠 이미지 파일들 삭제 (실시간 캡처를 위해)
    for old_file in glob.glob(str(save_path / "webcam_*.jpg")) + glob.glob(str(save_path / "virtual_webcam_*.jpg")):
        try:
            os.remove(old_file)
            print(f"🗑️  기존 파일 삭제: {old_file}")
        except:
            pass
    
    captured_files = []
    # 마이크로초 단위까지 포함한 정밀한 타임스탬프
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # 웹캠 인덱스 0-3까지 시도
    webcam_found = False
    for camera_idx in range(4):
        try:
            print(f"🎥 웹캠 {camera_idx} 연결 시도 중...")
            cap = cv2.VideoCapture(camera_idx)
            
            # 웹캠이 열렸는지 확인
            if not cap.isOpened():
                print(f"⚠️  웹캠 {camera_idx} 연결 실패")
                continue
            
            # 고화질 웹캠 설정 최적화
            # 해상도를 Full HD (1920x1080)로 설정
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            # FPS 설정 (고해상도에서는 30fps가 어려울 수 있으므로 15fps로 설정)
            cap.set(cv2.CAP_PROP_FPS, 15)
            
            # 추가 화질 개선 설정
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)    # 밝기 조정 (0.0-1.0)
            cap.set(cv2.CAP_PROP_CONTRAST, 0.5)      # 대비 조정 (0.0-1.0)
            cap.set(cv2.CAP_PROP_SATURATION, 0.6)    # 채도 조정 (0.0-1.0)
            cap.set(cv2.CAP_PROP_SHARPNESS, 0.5)     # 선명도 조정 (0.0-1.0)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # 자동 노출 활성화
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)       # 자동 초점 활성화
            
            # 실제 설정된 값 확인
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"📏 설정된 해상도: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
            
            # 몇 프레임 건너뛰기 (버퍼 클리어를 위해)
            for _ in range(5):
                cap.read()
                
            # 실제 프레임 읽기
            ret, frame = cap.read()
            
            if ret and frame is not None and frame.size > 0:
                # 이미지 품질 개선 처리
                frame = enhance_image_quality(frame)
                
                webcam_found = True
                # 파일명 생성 (마이크로초 포함)
                filename = f"webcam_{camera_idx}_{timestamp}.jpg"
                file_path = save_path / filename
                
                # 고화질 JPEG 저장 설정
                jpeg_quality = [cv2.IMWRITE_JPEG_QUALITY, 95]  # 95% 품질로 설정 (기본값: 95)
                png_compression = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # PNG 압축 레벨 1 (최소 압축)
                
                # 이미지 저장 (높은 품질로)
                success = cv2.imwrite(str(file_path), frame, jpeg_quality)
                
                if success:
                    captured_files.append(str(file_path))
                    print(f"✅ 웹캠 {camera_idx} 실시간 이미지 저장됨: {file_path}")
                    print(f"📐 이미지 크기: {frame.shape}")
                else:
                    print(f"❌ 웹캠 {camera_idx} 이미지 저장 실패")
                
            else:
                print(f"❌ 웹캠 {camera_idx} 프레임 읽기 실패")
            
            cap.release()
            
        except Exception as e:
            print(f"❌ 웹캠 {camera_idx} 캡처 오류: {str(e)}")
            continue
    
    # 실제 웹캠이 없는 경우
    if not webcam_found:
        print("⚠️ 연결된 웹캠이 없습니다.")
    
    print(f"📊 총 {len(captured_files)}개의 웹캠 이미지가 저장되었습니다.")
    if captured_files:
        print(f"🕒 캡처 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    return captured_files


@router.get("/healthz")
def health_check():
    return {"ok": True}


@router.post("/detect", response_model=List[DetectItem])
async def detect(
    file: UploadFile = File(...),
    text: str = Form(...),
    box_threshold: Optional[float] = Form(None),
    text_threshold: Optional[float] = Form(None),
    model: Optional[str] = Form(None),
    detection_manager: DetectionServiceManager = Depends(get_detection_manager_dependency),
) -> List[DetectItem]:
    detection_service = detection_manager.resolve(model)
    payload = await file.read()
    result = detection_service.detect_from_bytes(
        data=payload,
        filename=file.filename,
        caption=text,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    return [DetectItem.from_domain(item) for item in result.items]


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    detection_manager: DetectionServiceManager = Depends(get_detection_manager_dependency),
    session: Session = Depends(get_db_session_dependency),
) -> SearchResponse:
    # 🎥 웹캠 이미지 캡처를 먼저 수행
    print("🎥 웹캠 이미지 캡처를 시작합니다...")
    captured_files = capture_webcam_images("data/gallery")
    
    if captured_files:
        print(f"✅ {len(captured_files)}개의 웹캠 이미지가 data/gallery/에 저장되었습니다.")
        for f in captured_files:
            print(f"  📁 {f}")
    else:
        print("⚠️ 캡처된 웹캠 이미지가 없습니다. 기본 검색을 수행합니다.")
    
    detection_service = detection_manager.resolve(request.model)
    
    results_payload = None
    if getattr(detection_service, "model_name", None) == "owlv2":
        owns_session = False
        override_session = None
        database: Optional[Database] = None
        if request.database_url:
            database = Database.create(request.database_url)
            override_session = database.session()
            repository = OwlV2ExampleRepository(session=override_session)
            owns_session = True
        else:
            repository = OwlV2ExampleRepository(session=session)

        examples = repository.list_examples(query_text=request.text)

        if not examples:
            if owns_session and override_session is not None:
                override_session.close()
                if database is not None:
                    database.engine.dispose()
            return SearchResponse(results=[])

        query_images = [example.image_data for example in examples]
        query_labels = [
            example.filename or f"example_{example.id}" for example in examples
        ]
        try:
            results_payload = detection_service.detect_in_directory(
                caption=request.text,
                directory=None,
                patterns=request.patterns,
                box_threshold=request.box_threshold,
                text_threshold=request.text_threshold,
                limit=request.limit,
                only_with_detections=True,
                query_images=query_images,
                query_labels=query_labels,
            )
        finally:
            if owns_session and override_session is not None:
                override_session.close()
                if database is not None:
                    database.engine.dispose()
    else:
        results_payload = detection_service.detect_in_directory(
            caption=request.text,
            directory=None,
            patterns=request.patterns,
            box_threshold=request.box_threshold,
            text_threshold=request.text_threshold,
            limit=request.limit,
            only_with_detections=True,
        )

    response_items = []
    for payload in results_payload:
        annotated = None
        if payload.annotated_path and payload.annotated_path.exists():
            try:
                data, mime = encode_file_to_base64(payload.annotated_path)
                annotated = AnnotatedImageResponse(data=data, mime_type=mime)
            except FileNotFoundError:
                annotated = None
        response_items.append(
            ImageSearchResult(
                image=str(payload.source_path),
                detections=[DetectItem.from_domain(item) for item in payload.items],
                annotated_image=annotated,
            )
        )

    return SearchResponse(results=response_items)