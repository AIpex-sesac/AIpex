#!/usr/bin/env python3
import cv2
import time
from picamera2 import Picamera2

SCREEN_W = 800
SCREEN_H = 480

def main():
    print("[DEBUG] FRONT TX preview start...")
    cam = Picamera2(1)

    # ★ BoardTransmission 에서 쓰는 것과 동일한 설정
    config = cam.create_video_configuration(
        main={"size": (1640, 1232)},
        lores={"size": (640, 480), "format": "RGB888"},
    )
    cam.configure(config)
    cam.start()
    time.sleep(0.5)

    cv2.namedWindow("FRONT_TX_MAIN", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FRONT_TX_MAIN", SCREEN_W, SCREEN_H)

    try:
        while True:
            # ★ A 파이로 보내는 것과 같은 main 스트림
            frame = cam.capture_array("main")  # RGB

            # ★ 전송할 때와 동일하게 180도 회전
            frame = cv2.rotate(frame, cv2.ROTATE_180)

            # OpenCV 표시용 BGR로 변환
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 화면에 맞게 리사이즈
            h, w, _ = frame_bgr.shape
            scale = min(SCREEN_W / w, SCREEN_H / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            frame_resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            cv2.imshow("FRONT_TX_MAIN", frame_resized)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()
        print("[DEBUG] FRONT TX preview end")

if __name__ == "__main__":
    main()
