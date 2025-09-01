import cv2, time

def open_cam(dev="/dev/video0"):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)

    # 포맷/해상도 강제: 이 장치는 YUYV만 지원합니다.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    time.sleep(0.2)  # 드라이버 설정 반영 대기

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"{dev}에서 프레임을 못 받았습니다. (YUYV/640x480 확인)")

    return cap

if __name__ == "__main__":
    cap = open_cam("/dev/video0")  # 필요시 /dev/video1로 바꿔 테스트
    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임 읽기 실패")
            break
        cv2.imshow("cam", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
