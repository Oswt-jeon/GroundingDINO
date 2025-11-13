#!/usr/bin/env python3
"""
가상 웹캠 시뮬레이션 및 테스트 이미지 생성
"""

import cv2
import numpy as np
import time
from pathlib import Path
import subprocess
import os

def create_test_images(save_directory: str = "data/gallery", count: int = 3):
    """
    테스트용 이미지들을 생성합니다 (가상 웹캠 시뮬레이션)
    """
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    created_files = []
    
    print(f"🎨 테스트 이미지 생성 중... (저장 위치: {save_path})")
    
    for i in range(count):
        # 640x480 크기의 컬러 이미지 생성
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 다양한 색상의 배경
        colors = [
            (255, 100, 100),  # 빨간색 계열
            (100, 255, 100),  # 녹색 계열
            (100, 100, 255),  # 파란색 계열
        ]
        
        color = colors[i % len(colors)]
        img[:] = color
        
        # 텍스트 추가
        cv2.putText(img, f'Virtual Webcam {i}', (150, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(img, f'Timestamp: {timestamp}', (120, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 간단한 도형 추가 (객체 검출 테스트용)
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
        filename = f"webcam_{i}_{timestamp}.jpg"
        file_path = save_path / filename
        
        success = cv2.imwrite(str(file_path), img)
        if success:
            created_files.append(str(file_path))
            print(f"✅ 가상 웹캠 {i} 이미지 생성됨: {file_path}")
        else:
            print(f"❌ 가상 웹캠 {i} 이미지 생성 실패")
        
        time.sleep(0.1)  # 약간의 지연
    
    return created_files

def check_video_devices():
    """
    시스템의 비디오 장치들을 확인합니다
    """
    print("🔍 비디오 장치 확인 중...")
    
    # /dev/video* 장치 확인
    video_devices = []
    for i in range(10):
        device_path = f"/dev/video{i}"
        if Path(device_path).exists():
            video_devices.append(device_path)
    
    if video_devices:
        print(f"✅ 발견된 비디오 장치들: {video_devices}")
        return video_devices
    else:
        print("❌ 비디오 장치를 찾을 수 없습니다")
        return []

def simulate_webcam_with_opencv():
    """
    OpenCV를 사용한 가상 웹캠 시뮬레이션
    """
    print("🎥 OpenCV 가상 웹캠 시뮬레이션 시작...")
    
    # VideoWriter를 사용해서 가상 비디오 생성
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    # 임시 비디오 파일 생성
    temp_video = "temp_webcam.avi"
    out = cv2.VideoWriter(temp_video, fourcc, 10.0, (640, 480))
    
    print("📹 임시 비디오 파일 생성 중...")
    
    for frame_num in range(50):  # 5초 분량 (10fps)
        # 프레임 생성
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 움직이는 패턴
        x = int(320 + 200 * np.sin(frame_num * 0.2))
        y = int(240 + 100 * np.cos(frame_num * 0.15))
        
        cv2.circle(img, (x, y), 30, (0, 255, 0), -1)
        cv2.putText(img, f'Frame {frame_num}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(img)
    
    out.release()
    print(f"✅ 임시 비디오 생성 완료: {temp_video}")
    
    # 생성된 비디오를 VideoCapture로 읽기 테스트
    cap = cv2.VideoCapture(temp_video)
    
    if cap.isOpened():
        print("✅ 비디오 캡처 테스트 성공!")
        
        # 몇 프레임 읽어서 이미지로 저장
        save_path = Path("data/gallery")
        save_path.mkdir(parents=True, exist_ok=True)
        
        for i in range(3):
            ret, frame = cap.read()
            if ret:
                filename = f"virtual_webcam_frame_{i}.jpg"
                file_path = save_path / filename
                cv2.imwrite(str(file_path), frame)
                print(f"✅ 프레임 {i} 저장됨: {file_path}")
        
        cap.release()
    else:
        print("❌ 비디오 캡처 테스트 실패")
    
    # 임시 파일 정리
    if Path(temp_video).exists():
        os.remove(temp_video)
        print(f"🗑️ 임시 파일 삭제: {temp_video}")

def main():
    """메인 함수"""
    print("🎥 가상 웹캠 시뮬레이션 도구")
    print("=" * 50)
    
    # 시스템 정보 확인
    print(f"📦 OpenCV 버전: {cv2.__version__}")
    print(f"📁 현재 디렉토리: {os.getcwd()}")
    
    # 비디오 장치 확인
    video_devices = check_video_devices()
    
    print("\n📋 옵션을 선택하세요:")
    print("1. 테스트 이미지 생성 (정적)")
    print("2. 가상 비디오 시뮬레이션 (동적)")
    print("3. 둘 다 실행")
    
    try:
        choice = input("\n선택 (1-3): ").strip()
        
        if choice == "1":
            files = create_test_images()
            print(f"\n✅ {len(files)}개의 테스트 이미지가 생성되었습니다!")
            
        elif choice == "2":
            simulate_webcam_with_opencv()
            print(f"\n✅ 가상 웹캠 시뮬레이션 완료!")
            
        elif choice == "3":
            files = create_test_images()
            print(f"\n✅ {len(files)}개의 테스트 이미지가 생성되었습니다!")
            
            simulate_webcam_with_opencv()
            print(f"\n✅ 가상 웹캠 시뮬레이션 완료!")
            
        else:
            print("❌ 잘못된 선택입니다.")
            return
        
        # 생성된 파일들 확인
        gallery_path = Path("data/gallery")
        if gallery_path.exists():
            image_files = list(gallery_path.glob("*.jpg"))
            if image_files:
                print(f"\n📁 data/gallery/ 에 {len(image_files)}개의 이미지 파일이 있습니다:")
                for img_file in sorted(image_files):
                    size = img_file.stat().st_size
                    print(f"   - {img_file.name} ({size:,} bytes)")
        
    except KeyboardInterrupt:
        print("\n👋 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()