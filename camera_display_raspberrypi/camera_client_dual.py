#!/usr/bin/env python3
import socket
import struct
import json
import cv2
import numpy as np
import threading
import time

# ====== A 파이 주소/포트 설정 ======
A_HOST = "192.168.0.10"    #### 헤일로 장착 라즈베리파이 IP로 바꾸기
FRONT_PORT = 50000
REAR_PORT = 50001

# ====== 카메라 인덱스 설정 ======
CAM_INDEX_FRONT = 0
CAM_INDEX_REAR = 1

# ====== 공용 유틸 ======
def recvall(sock, n):
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

# ====== 디텍션 결과 공유 변수 ======
front_result = {"width": 640, "height": 480, "detections": []}
rear_result = {"width": 640, "height": 480, "detections": []}
front_lock = threading.Lock()
rear_lock = threading.Lock()

# ====== 카메라 + 소켓 스레드 ======
def camera_thread(cam_name, cam_index, port, result_ref, lock):
    """
    cam_name: "FRONT"/"REAR"
    cam_index: /dev/videoN
    port: A 파이에서 listen 중인 포트
    result_ref: 전역 결과 dict (front_result 또는 rear_result)
    lock: 해당 결과용 Lock
    """
    print(f"[B][{cam_name}] Opening camera index {cam_index} ...")
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[B][{cam_name}] Failed to open camera.")
        return

    print(f"[B][{cam_name}] Connecting to A {A_HOST}:{port} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((A_HOST, port))
    except Exception as e:
        print(f"[B][{cam_name}] Failed to connect:", e)
        cap.release()
        return

    print(f"[B][{cam_name}] Connected.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[B][{cam_name}] Failed to read frame.")
                break

            # 필요하면 해상도 줄이기 (예: 640x480)
            # frame = cv2.resize(frame, (640, 480))

            # JPEG 인코딩
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            ok, encoded = cv2.imencode(".jpg", frame, encode_param)
            if not ok:
                print(f"[B][{cam_name}] Failed to encode frame.")
                continue
            frame_bytes = encoded.tobytes()

            # 길이 + 데이터 전송
            sock.sendall(struct.pack(">I", len(frame_bytes)))
            sock.sendall(frame_bytes)

            # JSON 길이 수신
            length_buf = recvall(sock, 4)
            if not length_buf:
                print(f"[B][{cam_name}] Server closed connection.")
                break
            (json_len,) = struct.unpack(">I", length_buf)

            # JSON 데이터 수신
            json_data = recvall(sock, json_len)
            if json_data is None:
                print(f"[B][{cam_name}] Failed to receive JSON.")
                break

            result = json.loads(json_data.decode("utf-8"))

            # 결과 공유 변수 갱신
            with lock:
                result_ref.clear()
                result_ref.update(result)

            # 너무 빨리 돌면 부하 크면 살짝 sleep
            # time.sleep(0.01)

    except Exception as e:
        print(f"[B][{cam_name}] Exception:", e)
    finally:
        print(f"[B][{cam_name}] Thread exit.")
        cap.release()
        sock.close()

# ====== 바운딩박스 렌더링 도우미 ======
def render_black_canvas_from_result(result):
    w = result.get("width", 640)
    h = result.get("height", 480)
    detections = result.get("detections", [])

    black = np.zeros((h, w, 3), dtype=np.uint8)

    for det in detections:
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])
        cls_name = det.get("cls_name", "obj")
        conf = float(det.get("conf", 0.0))

        # 박스
        cv2.rectangle(black, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 라벨 (중앙 정렬)
        label = f"{cls_name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        box_w = x2 - x1
        text_x = x1 + (box_w - text_w) // 2
        text_y = y2 + text_h + 5

        # 화면 범위 보정
        text_x = max(0, min(text_x, w - text_w))
        text_y = max(text_h, min(text_y, h - 1))

        cv2.putText(black, label, (text_x, text_y),
                    font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    return black

# ====== 메인 (HUD 렌더 루프) ======
def main():
    global front_result, rear_result

    # 카메라 스레드 두 개 시작
    t_front = threading.Thread(
        target=camera_thread,
        args=("FRONT", CAM_INDEX_FRONT, FRONT_PORT, front_result, front_lock),
        daemon=True,
    )
    t_rear = threading.Thread(
        target=camera_thread,
        args=("REAR", CAM_INDEX_REAR, REAR_PORT, rear_result, rear_lock),
        daemon=True,
    )

    t_front.start()
    t_rear.start()

    # OpenCV 윈도우 준비
    cv2.namedWindow("Front HUD", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Rear HUD", cv2.WINDOW_NORMAL)

    # Waveshare 3.5"에 전체화면으로 띄우고 싶으면:
    # cv2.setWindowProperty("Front HUD", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.setWindowProperty("Rear HUD", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("[B] HUD rendering started. Press ESC to quit.")

    while True:
        # Front 캔버스
        with front_lock:
            fr = dict(front_result)
        front_canvas = render_black_canvas_from_result(fr)

        # Rear 캔버스
        with rear_lock:
            rr = dict(rear_result)
        rear_canvas = render_black_canvas_from_result(rr)

        cv2.imshow("Front HUD", front_canvas)
        cv2.imshow("Rear HUD", rear_canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()
    print("[B] Exit.")

if __name__ == "__main__":
    main()
