#!/usr/bin/env python3
"""
수정된 웹캠 캡처 함수 테스트
"""

import sys
sys.path.append('/home/jeyun/GroundingDINO')

import os
import cv2
import time
from pathlib import Path
from typing import List
import numpy as np

def capture_webcam_images(save_directory: str = "data/gallery") -> List[str]:
    """
    연결된 모든 웹캠에서 이미지를 캡처하여 지정된 디렉토리에 저장합니다.
    실제 웹캠이 없는 경우 테스트 이미지를 생성합니다.
    
    Args:
        save_directory: 이미지를 저장할 디렉토리 경로
        
    Returns:
        저장된 이미지 파일 경로들의 리스트
    """
    # 저장 디렉토리 생성
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    
    captured_files = []
    timestamp = int(time.time())
    
    # 웹캠 인덱스 0-3까지 시도
    webcam_found = False
    for camera_idx in range(4):
        try:
            cap = cv2.VideoCapture(camera_idx)
            
            # 웹캠이 열렸는지 확인
            if not cap.isOpened():
                continue
                
            # 프레임 읽기 시도
            ret, frame = cap.read()
            
            if ret and frame is not None:
                webcam_found = True
                # 파일명 생성
                filename = f"webcam_{camera_idx}_{timestamp}.jpg"
                file_path = save_path / filename
                
                # 이미지 저장
                success = cv2.imwrite(str(file_path), frame)
                
                if success:
                    captured_files.append(str(file_path))
                    print(f"✅ 웹캠 {camera_idx} 이미지 저장됨: {file_path}")
                
            cap.release()
            
        except Exception as e:
            print(f"❌ 웹캠 {camera_idx} 캡처 실패: {str(e)}")
            continue
    
    # 실제 웹캠이 없는 경우 가상 웹캠 이미지 생성
    if not webcam_found:
        print("🎨 실제 웹캠이 없어 테스트 이미지를 생성합니다...")
        
        for i in range(3):  # 3개의 가상 웹캠 이미지 생성
            # 640x480 크기의 컬러 이미지 생성
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 다양한 색상의 배경
            colors = [
                (200, 100, 100),  # 빨간색 계열
                (100, 200, 100),  # 녹색 계열
                (100, 100, 200),  # 파란색 계열
            ]
            
            color = colors[i % len(colors)]
            img[:] = color
            
            # 텍스트 추가
            cv2.putText(img, f'Virtual Webcam {i}', (150, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(img, f'Timestamp: {timestamp}', (120, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 객체 검출 테스트용 도형 추가
            if i == 0:
                # 사각형 (car 시뮬레이션)
                cv2.rectangle(img, (200, 300), (350, 400), (0, 0, 255), -1)
                cv2.putText(img, 'CAR', (245, 355), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            elif i == 1:
                # 원 (person 시뮬레이션)
                cv2.circle(img, (320, 350), 50, (255, 255, 0), -1)
                cv2.putText(img, 'PERSON', (260, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            elif i == 2:
                # 다각형 (dog 시뮬레이션)
                points = np.array([[250, 320], [390, 320], [350, 380], [290, 380]], np.int32)
                cv2.fillPoly(img, [points], (0, 255, 255))
                cv2.putText(img, 'DOG', (290, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # 파일 저장
            filename = f"virtual_webcam_{i}_{timestamp}.jpg"
            file_path = save_path / filename
            
            success = cv2.imwrite(str(file_path), img)
            if success:
                captured_files.append(str(file_path))
                print(f"✅ 가상 웹캠 {i} 이미지 생성됨: {file_path}")
    
    print(f"📊 총 {len(captured_files)}개의 웹캠 이미지가 저장되었습니다.")
    return captured_files

def main():
    """메인 함수"""
    print("🎥 수정된 웹캠 캡처 함수 테스트")
    print("=" * 50)
    
    # 기존 이미지 삭제 (테스트용)
    gallery_path = Path("data/gallery")
    if gallery_path.exists():
        for img_file in gallery_path.glob("virtual_webcam_*.jpg"):
            img_file.unlink()
            print(f"🗑️ 기존 파일 삭제: {img_file}")
    
    # 웹캠 캡처 실행
    captured_files = capture_webcam_images("data/gallery")
    
    if captured_files:
        print(f"\n✅ 성공! {len(captured_files)}개의 이미지가 생성되었습니다:")
        for file_path in captured_files:
            file_size = Path(file_path).stat().st_size
            print(f"   - {Path(file_path).name} ({file_size:,} bytes)")
    else:
        print("\n❌ 이미지 생성 실패")
    
    print("\n🎯 테스트 완료!")

if __name__ == "__main__":
    main()