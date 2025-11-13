import cv2
import numpy as np
import time
import sys
import os

def test_webcam():
    """
    웹캠 테스트 함수 - 여러 방법으로 웹캠 연결을 시도합니다
    """
    print("🎥 웹캠 테스트를 시작합니다...")
    
    # 다양한 웹캠 인덱스 시도
    webcam_indices = [0, 1, 2, 3]
    
    for idx in webcam_indices:
        print(f"\n📹 웹캠 인덱스 {idx} 테스트 중...")
        
        try:
            cap = cv2.VideoCapture(idx)
            
            # 웹캠이 열렸는지 확인
            if not cap.isOpened():
                print(f"❌ 웹캠 인덱스 {idx}: 열 수 없습니다")
                continue
            
            # 웹캠 속성 확인
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"✅ 웹캠 인덱스 {idx}: 연결 성공!")
            print(f"   해상도: {width}x{height}")
            print(f"   FPS: {fps}")
            
            # 몇 프레임 읽어보기
            frame_count = 0
            success_count = 0
            
            print(f"📊 프레임 테스트 중... (10프레임)")
            
            for i in range(10):
                ret, frame = cap.read()
                frame_count += 1
                
                if ret and frame is not None:
                    success_count += 1
                    if i == 0:  # 첫 번째 프레임 정보
                        print(f"   프레임 shape: {frame.shape}")
                        print(f"   프레임 dtype: {frame.dtype}")
                    
                    # 프레임을 이미지 파일로 저장 (GUI 대신)
                    if i < 3:  # 처음 3개 프레임만 저장
                        filename = f"webcam_{idx}_frame_{i}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"   프레임 {i} 저장됨: {filename}")
                
                time.sleep(0.1)  # 100ms 대기
            
            print(f"📈 결과: {success_count}/{frame_count} 프레임 성공")
            
            if success_count > 0:
                print(f"🎯 웹캠 인덱스 {idx} 테스트 완료!")
                print("📸 이미지 파일들이 저장되었습니다.")
                
                # 연속 프레임 캡처 (10초간)
                print("📹 10초간 연속 프레임 캡처 중...")
                start_time = time.time()
                captured_frames = 0
                
                while time.time() - start_time < 10:
                    ret, frame = cap.read()
                    if ret:
                        captured_frames += 1
                    time.sleep(0.033)  # ~30fps
                
                print(f"📊 10초간 {captured_frames}개 프레임 캡처됨 (평균 {captured_frames/10:.1f} fps)")
                
                cap.release()
                return True
            
            cap.release()
            
        except Exception as e:
            print(f"❌ 웹캠 인덱스 {idx} 에러: {str(e)}")
            continue
    
    print("\n❌ 사용 가능한 웹캠을 찾을 수 없습니다")
    return False

def test_ip_camera():
    """
    IP 카메라 연결 테스트
    """
    print("\n🌐 IP 카메라 테스트...")
    
    # 일반적인 IP 카메라 URL 패턴들
    ip_urls = [
        "http://192.168.1.100:8080/video",  # 일반적인 IP 카메라
        "http://192.168.0.100:8080/video",
        "http://localhost:8090",  # 로컬 스트리밍
        "http://127.0.0.1:8090",
    ]
    
    for url in ip_urls:
        print(f"🔗 {url} 테스트 중...")
        try:
            cap = cv2.VideoCapture(url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 설정
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ {url}: 연결 성공!")
                    print(f"   프레임 크기: {frame.shape}")
                    cv2.imwrite(f"ip_camera_frame.jpg", frame)
                    print(f"   프레임 저장됨: ip_camera_frame.jpg")
                    cap.release()
                    return url
            
            cap.release()
            print(f"❌ {url}: 연결 실패")
            
        except Exception as e:
            print(f"❌ {url} 에러: {str(e)}")
    
    return None

def create_test_video():
    """
    테스트용 가상 비디오 생성
    """
    print("\n🎬 테스트용 가상 비디오 생성 중...")
    
    try:
        # 가상 비디오 생성 (640x480, 30fps)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('test_video.avi', fourcc, 30.0, (640, 480))
        
        print("📹 300프레임 생성 중...")
        for i in range(300):  # 10초 분량 (30fps * 10)
            # 색상이 변하는 프레임 생성
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 색상 변화
            color = (
                int(127 * (1 + np.sin(i * 0.1))),
                int(127 * (1 + np.sin(i * 0.1 + 2))),
                int(127 * (1 + np.sin(i * 0.1 + 4)))
            )
            
            frame[:] = color
            
            # 텍스트 추가
            cv2.putText(frame, f'Test Frame {i}', (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
            
            if i % 50 == 0:
                print(f"   진행률: {i/300*100:.1f}%")
        
        out.release()
        print("✅ test_video.avi 파일이 생성되었습니다")
        
        # 생성된 비디오에서 몇 프레임 추출
        cap = cv2.VideoCapture('test_video.avi')
        
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            
            print(f"📊 비디오 정보:")
            print(f"   총 프레임: {total_frames}")
            print(f"   FPS: {fps}")
            print(f"   재생 시간: {duration:.1f}초")
            
            # 몇 개 프레임 추출해서 저장
            frame_indices = [0, 75, 150, 225, 299]
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(f"test_video_frame_{idx}.jpg", frame)
                    print(f"   프레임 {idx} 저장됨: test_video_frame_{idx}.jpg")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ 테스트 비디오 생성 실패: {str(e)}")
        return False

def analyze_saved_images():
    """
    저장된 이미지들 분석
    """
    print("\n🔍 저장된 이미지 분석...")
    
    image_files = [f for f in os.listdir('.') if f.endswith('.jpg')]
    
    if not image_files:
        print("❌ 저장된 이미지가 없습니다.")
        return
    
    print(f"📁 총 {len(image_files)}개의 이미지 파일 발견:")
    
    for img_file in sorted(image_files):
        try:
            img = cv2.imread(img_file)
            if img is not None:
                height, width, channels = img.shape
                file_size = os.path.getsize(img_file)
                print(f"   📷 {img_file}: {width}x{height}x{channels}, {file_size:,} bytes")
            else:
                print(f"   ❌ {img_file}: 읽기 실패")
        except Exception as e:
            print(f"   ❌ {img_file}: 에러 - {str(e)}")

def main():
    """
    메인 함수
    """
    print("🎥 웹캠 프레임 테스트 프로그램 (헤드리스 모드)")
    print("=" * 60)
    
    # OpenCV 버전 확인
    print(f"📦 OpenCV 버전: {cv2.__version__}")
    print(f"📁 현재 디렉토리: {os.getcwd()}")
    
    while True:
        print("\n📋 테스트 옵션을 선택하세요:")
        print("1. 웹캠 테스트 (자동 감지)")
        print("2. IP 카메라 테스트")
        print("3. 테스트 비디오 생성")
        print("4. 저장된 이미지 분석")
        print("5. 종료")
        
        try:
            choice = input("\n선택 (1-5): ").strip()
            
            if choice == '1':
                test_webcam()
            elif choice == '2':
                test_ip_camera()
            elif choice == '3':
                create_test_video()
            elif choice == '4':
                analyze_saved_images()
            elif choice == '5':
                print("👋 프로그램을 종료합니다.")
                break
            else:
                print("❌ 잘못된 선택입니다. 1-5 중에서 선택해주세요.")
                
        except KeyboardInterrupt:
            print("\n👋 프로그램이 중단되었습니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()