#!/usr/bin/env python3
"""
웹캠 캡처 기능 테스트 스크립트
"""

import os
import cv2
import time
from pathlib import Path
from typing import List

def capture_webcam_images(save_directory: str = "data/gallery") -> List[str]:
    """
    연결된 모든 웹캠에서 이미지를 캡처하여 지정된 디렉토리에 저장합니다.
    
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
    
    print(f"🎥 웹캠 이미지 캡처를 시작합니다... (저장 위치: {save_path})")
    
    # 웹캠 인덱스 0-3까지 시도
    for camera_idx in range(4):
        try:
            print(f"📹 웹캠 {camera_idx} 테스트 중...")
            cap = cv2.VideoCapture(camera_idx)
            
            # 웹캠이 열렸는지 확인
            if not cap.isOpened():
                print(f"❌ 웹캠 {camera_idx}: 열 수 없습니다")
                continue
                
            # 프레임 읽기 시도
            ret, frame = cap.read()
            
            if ret and frame is not None:
                # 파일명 생성
                filename = f"webcam_{camera_idx}_{timestamp}.jpg"
                file_path = save_path / filename
                
                # 이미지 저장
                success = cv2.imwrite(str(file_path), frame)
                
                if success:
                    captured_files.append(str(file_path))
                    print(f"✅ 웹캠 {camera_idx} 이미지 저장됨: {file_path}")
                else:
                    print(f"❌ 웹캠 {camera_idx} 이미지 저장 실패")
            else:
                print(f"❌ 웹캠 {camera_idx}: 프레임 읽기 실패")
                
            cap.release()
            
        except Exception as e:
            print(f"❌ 웹캠 {camera_idx} 캡처 실패: {str(e)}")
            continue
    
    print(f"📊 총 {len(captured_files)}개의 웹캠 이미지가 저장되었습니다.")
    return captured_files

def main():
    """메인 함수"""
    print("🎥 웹캠 캡처 테스트 시작")
    print("=" * 50)
    
    # OpenCV 버전 확인
    print(f"📦 OpenCV 버전: {cv2.__version__}")
    
    # 웹캠 이미지 캡처
    captured_files = capture_webcam_images("data/gallery")
    
    if captured_files:
        print("\n✅ 캡처 성공!")
        print("📁 저장된 파일들:")
        for file_path in captured_files:
            file_size = Path(file_path).stat().st_size
            print(f"   - {file_path} ({file_size:,} bytes)")
    else:
        print("\n❌ 캡처된 이미지가 없습니다.")
        print("   웹캠이 연결되어 있는지 확인해주세요.")
    
    print("\n🎯 테스트 완료!")

if __name__ == "__main__":
    main()