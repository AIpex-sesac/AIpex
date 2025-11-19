#!/usr/bin/env python3
# camera_client_dual_picam2.py
# -------------------------------------------
# B 라즈베리파이:
# - Picamera2(0) : FRONT 카메라
# - Picamera2(1) : REAR  카메라
# - 각 프레임을 A 파이로 전송 → A 파이가 디텍팅 → JSON 결과 회신
# - FRONT HUD : 항상 전체화면 (480x320로 리사이즈)
#   + 우측 상단에 X1200 배터리 퍼센트 표시 (퍼센트 + 4칸 배터리 아이콘)
#   + 좌측 상단에 REAR 카메라 실영상 + 디텍팅 박스(PIP)
#   + 하단에 방위각 눈금자(스케일) 표시 (5° 눈금, 10° 숫자)
#   + 최종 HUD 화면 전체 좌우 반전 (버드베스/HUD 반사용)
# -------------------------------------------

import socket
import struct
import json
import cv2
import numpy as np
import threading
import time

from picamera2 import Picamera2
import smbus2  # X1200 I2C 배터리 게이지용

# ====== A 파이 주소/포트 설정 ======
A_HOST = "192.168.0.10"    # A 라즈베리파이 IP로 바꿔줘
FRONT_PORT = 50000
REAR_PORT = 50001

# ====== HUD / PIP 설정 ======
SCREEN_W = 480
SCREEN_H = 320

# rear 풀스크린 기준 (4:3 비율 유지, HUD(480x320) 안에 최대)
REAR_FULL_H = SCREEN_H
REAR_FULL_W = int(REAR_FULL_H * 4 / 3)
if REAR_FULL_W > SCREEN_W:
    REAR_FULL_W = SCREEN_W
    REAR_FULL_H = int(REAR_FULL_W * 3 / 4)

# rpicam 풀스크린을 기준으로 축소 비율
PIP_SCALE = 0.33   # 0.5면 1/2 크기, 0.6이면 60% 크기

PIP_W = int(REAR_FULL_W * PIP_SCALE)
PIP_H = int(REAR_FULL_H * PIP_SCALE)
PIP_X = 10   # 좌측 여백
PIP_Y = 10   # 상단 여백

# ====== 공용 유틸 ======
def recvall(sock, n: int):
    """정확히 n바이트를 받을 때까지 반복 수신."""
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

# ====== 디텍션 결과 + REAR 프레임 공유 변수 ======
front_result = {"width": 640, "height": 480, "detections": []}
rear_result  = {"width": 640, "height": 480, "detections": []}
front_lock = threading.Lock()
rear_lock  = threading.Lock()

rear_frame: np.ndarray | None = None
rear_frame_lock = threading.Lock()

# ====== 방위각 (폰에서 받아올 값; 지금은 테스트용 애니메이션) ======
heading_deg: float = 0.0
heading_lock = threading.Lock()
HEADING_SPEED_DEG_PER_SEC = 30.0  # 테스트용: 초당 30도 회전 (원하면 조절)

# ====== X1200 배터리 퍼센트 읽기 ======
I2C_BUS_ID = 1       # /dev/i2c-1
FG_ADDR    = 0x36    # X120x fuel gauge I2C 주소 (일반적으로 0x36)

def _read_word_swapped(bus, reg):
    raw = bus.read_word_data(FG_ADDR, reg)
    return ((raw & 0xFF) << 8) | (raw >> 8)

def get_battery_percentage() -> int | None:
    try:
        bus = smbus2.SMBus(I2C_BUS_ID)
        raw_soc = _read_word_swapped(bus, 0x04)
        bus.close()

        percent = raw_soc / 256.0
        percent = max(0.0, min(100.0, percent))
        return int(round(percent))
    except Exception as e:
        print("[BAT] Failed to read battery:", e)
        return None

# ====== 배터리 오버레이 (퍼센트 + 4칸 아이콘) ======
def draw_battery_overlay(frame: np.ndarray, level: int | None) -> np.ndarray:
    h, w, _ = frame.shape

    if level is None:
        percent_text = "--%"
        level_val = 0
    else:
        percent_text = f"{int(level)}%"
        level_val = max(0, min(100, int(level)))

    bw, bh = 70, 18
    margin = 8
    x2 = w - margin
    x1 = x2 - bw
    y1 = margin
    y2 = y1 + bh

    green = (0, 255, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), green, 2)

    head_w = 5
    cv2.rectangle(
        frame,
        (x2, y1 + bh // 4),
        (x2 + head_w, y2 - bh // 4),
        green,
        -1,
    )

    inner_margin = 3
    ix1 = x1 + inner_margin
    ix2 = x2 - inner_margin
    iy1 = y1 + inner_margin
    iy2 = y2 - inner_margin

    cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (255, 255, 255), -1)

    inner_width = ix2 - ix1
    fill_w = int(inner_width * (level_val / 100.0))
    if fill_w > 0:
        cv2.rectangle(
            frame,
            (ix1, iy1),
            (ix1 + fill_w, iy2),
            green,
            -1,
        )

    cells = 4
    for i in range(1, cells):
        x = ix1 + int(inner_width * i / cells)
        cv2.line(
            frame,
            (x, iy1),
            (x, iy2),
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (pt_w, pt_h), _ = cv2.getTextSize(percent_text, font, font_scale, thickness)

    text_x = x1 - pt_w - 8
    text_y = y1 + pt_h + 2

    cv2.putText(
        frame,
        percent_text,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        percent_text,
        (text_x, text_y),
        font,
        font_scale,
        green,
        thickness,
        cv2.LINE_AA,
    )

    return frame

# ====== 방위각 눈금자 오버레이 ======
def draw_heading_scale(frame: np.ndarray, heading_deg: float) -> np.ndarray:
    """
    화면 하단에 방위각 눈금자 / 삼각형 / 현재 각도 숫자를 그린다.
    heading_deg : 0~360 (0=북쪽 기준이라고 가정)
    """
    h, w, _ = frame.shape
    green = (0, 255, 0)

    band_h = 16          # 눈금 세로 높이
    px_per_deg = 4       # 1도당 몇 픽셀 이동할지(가로 스케일)

    # ✅ 좌우 마진 (전체 폭의 30%씩 비우기)
    side_margin = int(w * 0.30)
    left_bound  = side_margin
    right_bound = w - side_margin

    center_x = w // 2
    margin_bottom = 20   # 화면 맨 아래로부터 여유
    y0 = h - band_h - margin_bottom   # 눈금의 최상단 y

    font = cv2.FONT_HERSHEY_SIMPLEX

    # 화면에 그릴 수 있는 최대 degree offset (= 사용 가능한 폭 기준)
    usable_w = right_bound - left_bound
    max_offset = int((usable_w / 2) / px_per_deg) + 2

    # ==== 눈금 그리기 (현재 heading 기준 좌우로) ====
    for offset in range(-max_offset, max_offset + 1):
        deg = (heading_deg + offset) % 360.0
        x = int(center_x + offset * px_per_deg)

        # ✅ 좌우 마진 안쪽에서만 그리기
        if x < left_bound or x >= right_bound:
            continue

        d_int = int(round(deg))

        # 기본: 아무 것도 안 그림
        draw_tick = False
        length = band_h - 16  # 거의 0

        # 5° 단위 눈금
        if d_int % 5 == 0:
            length = band_h - 14  # 짧은 눈금
            draw_tick = True

        # 10° 단위 눈금 (조금 더 길게)
        if d_int % 10 == 0:
            length = band_h - 10
            draw_tick = True

        if draw_tick:
            y1 = y0
            y2 = y0 + length
            cv2.line(frame, (x, y1), (x, y2), green, 1, cv2.LINE_AA)

            # 10도 단위 숫자 표시 (눈금 아래)
            if d_int % 10 == 0:
                label = str(d_int % 360)
                font_scale = 0.45
                t_thick = 1
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, t_thick)
                tx = x - tw // 2
                ty = y2 + th + 2
                cv2.putText(
                    frame,
                    label,
                    (tx, ty),
                    font,
                    font_scale,
                    green,
                    t_thick,
                    cv2.LINE_AA,
                )

    # ==== 중앙 삼각형 마커 (▼ 아래로 향하도록, 눈금 위에) ====
    tri_height = 10
    offset_above_scale = 8     # 눈금과 삼각형 사이 거리
    base_y = max(0, y0 - offset_above_scale - tri_height)  # 삼각형 윗변 y
    tip_y  = base_y + tri_height                           # ▼ 아래 꼭짓점 y

    pts = np.array(
        [
            [center_x - 6, base_y],   # 왼쪽 위
            [center_x + 6, base_y],   # 오른쪽 위
            [center_x,     tip_y],    # 아래 꼭짓점
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, pts, green)

    # ==== 현재 heading 숫자를 "삼각형 위"에 배치 ====
    cur_label = f"{int(round(heading_deg)) % 360}"
    font_scale = 0.6
    t_thick = 2
    (tw, th), _ = cv2.getTextSize(cur_label, font, font_scale, t_thick)

    tx = center_x - tw // 2
    ty = max(th + 2, base_y - 6)

    cv2.putText(
        frame,
        cur_label,
        (tx, ty),
        font,
        font_scale,
        (0, 0, 0),
        t_thick + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        cur_label,
        (tx, ty),
        font,
        font_scale,
        green,
        t_thick,
        cv2.LINE_AA,
    )

    return frame

# ====== 카메라 + 소켓 스레드 (Picamera2) ======
def camera_thread_picam(cam_name: str,
                        cam_index: int,
                        port: int,
                        result_ref: dict,
                        lock: threading.Lock):
    global rear_frame, heading_deg

    print(f"[B][{cam_name}] Starting Picamera2 index {cam_index} ...")

    picam = Picamera2(cam_index)
    # rpicam-hello와 비슷한 FOV: 1640 x 1232
    config = picam.create_video_configuration(
        main={"size": (1640, 1232), "format": "RGB888"}
    )
    picam.configure(config)
    picam.start()
    time.sleep(0.5)

    # FRONT 카메라일 때 방위각 업데이트용 시간 기준
    last_t = time.time()

    sock = None
    try:
        print(f"[B][{cam_name}] Connecting to A {A_HOST}:{port} ...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.connect((A_HOST, port))
        sock.settimeout(None)
        print(f"[B][{cam_name}] Connected to A.")
    except Exception as e:
        print(f"[B][{cam_name}] Failed to connect to A: {e}")
        print(f"[B][{cam_name}] → 디텍팅 서버 없이 카메라만 실행합니다.")
        sock = None

    try:
        while True:
            frame_rgb = picam.capture_array("main")

            # 여기서는 드라이버가 주는 포맷 그대로 사용
            frame_bgr = frame_rgb

            # REAR 카메라는 PIP용 프레임 저장
            if cam_name.upper() == "REAR":
                with rear_frame_lock:
                    rear_frame = frame_bgr.copy()

            # FRONT 카메라일 때 방위각을 시간 기반으로 업데이트
            if cam_name.upper() == "FRONT":
                now = time.time()
                dt = now - last_t
                last_t = now
                with heading_lock:
                    heading_deg = (heading_deg +
                                   HEADING_SPEED_DEG_PER_SEC * dt) % 360.0

            if sock is not None:
                try:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                    ok, encoded = cv2.imencode(".jpg", frame_bgr, encode_param)
                    if not ok:
                        print(f"[B][{cam_name}] Failed to encode frame.")
                        continue

                    frame_bytes = encoded.tobytes()
                    sock.sendall(struct.pack(">I", len(frame_bytes)))
                    sock.sendall(frame_bytes)

                    length_buf = recvall(sock, 4)
                    if not length_buf:
                        print(f"[B][{cam_name}] Server closed connection.")
                        sock.close()
                        sock = None
                        continue

                    (json_len,) = struct.unpack(">I", length_buf)
                    json_data = recvall(sock, json_len)
                    if json_data is None:
                        print(f"[B][{cam_name}] Failed to receive JSON.")
                        sock.close()
                        sock = None
                        continue

                    result = json.loads(json_data.decode("utf-8"))
                    with lock:
                        result_ref.clear()
                        result_ref.update(result)

                except Exception as e_net:
                    print(f"[B][{cam_name}] Network error: {e_net}")
                    try:
                        sock.close()
                    except Exception:
                        pass
                    sock = None

    except Exception as e:
        print(f"[B][{cam_name}] Exception:", e)
    finally:
        print(f"[B][{cam_name}] Thread exit.")
        if sock is not None:
            sock.close()
        picam.close()

# ====== 바운딩박스 렌더링 (검은 배경용, FRONT / REAR 공용) ======
def render_black_canvas_from_result(result: dict) -> np.ndarray:
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

        cv2.rectangle(black, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{cls_name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        box_w = x2 - x1
        text_x = x1 + (box_w - text_w) // 2
        text_y = y2 + text_h + 5

        text_x = max(0, min(text_x, w - text_w))
        text_y = max(text_h, min(text_y, h - 1))

        cv2.putText(
            black,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA,
        )

    return black

# ====== 메인 (HUD 렌더 루프) ======
def main():
    global front_result, rear_result, rear_frame, heading_deg

    t_front = threading.Thread(
        target=camera_thread_picam,
        args=("FRONT", 0, FRONT_PORT, front_result, front_lock),
        daemon=True,
    )
    t_rear = threading.Thread(
        target=camera_thread_picam,
        args=("REAR", 1, REAR_PORT, rear_result, rear_lock),
        daemon=True,
    )

    t_front.start()
    t_rear.start()

    cv2.namedWindow("Front HUD", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "Front HUD",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    last_batt_read_time = 0.0
    batt_percent_cached: int | None = None

    print("[B] HUD started. ESC 로 종료.")

    while True:
        now = time.time()

        # --- 배터리 ---
        if now - last_batt_read_time > 1.0:
            batt_percent_cached = get_battery_percentage()
            last_batt_read_time = now

        # --- heading: 카메라 스레드에서 업데이트된 값 읽기 ---
        with heading_lock:
            cur_heading = heading_deg

        # --- FRONT ---
        with front_lock:
            fr = dict(front_result)

        front_canvas = render_black_canvas_from_result(fr)
        front_canvas_fs = cv2.resize(
            front_canvas,
            (SCREEN_W, SCREEN_H),
            interpolation=cv2.INTER_LINEAR
        )
        front_canvas_fs = draw_battery_overlay(front_canvas_fs, batt_percent_cached)

        # 방위각 스케일 오버레이
        front_canvas_fs = draw_heading_scale(front_canvas_fs, cur_heading)

        # ---------- 좌측 상단 REAR 카메라 PIP ----------
        with rear_frame_lock:
            if rear_frame is None:
                # rear_frame 없을 때도 HUD 전체 좌우 반전
                flipped = cv2.flip(front_canvas_fs, 1)
                cv2.imshow("Front HUD", flipped)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue
            rf = rear_frame.copy()

        # 방향 보정: 180도 회전 + 좌우반전
        rf = cv2.rotate(rf, cv2.ROTATE_180)
        rf = cv2.flip(rf, 1)

        with rear_lock:
            rr = dict(rear_result)

        rear_resized = cv2.resize(
            rf,
            (PIP_W, PIP_H),
            interpolation=cv2.INTER_LINEAR
        )

        # Rear PIP 테두리 (빨간색)
        cv2.rectangle(rear_resized, (0, 0), (PIP_W - 1, PIP_H - 1), (0, 0, 255), 2)

        rw = rr.get("width", 640)
        rh = rr.get("height", 480)
        detections = rr.get("detections", [])

        sx = PIP_W / float(rw) if rw > 0 else 1.0
        sy = PIP_H / float(rh) if rh > 0 else 1.0

        for det in detections:
            x1 = int(det["x1"] * sx)
            y1 = int(det["y1"] * sy)
            x2 = int(det["x2"] * sx)
            y2 = int(det["y2"] * sy)
            cls_name = det.get("cls_name", "obj")
            conf = float(det.get("conf", 0.0))

            cv2.rectangle(rear_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{cls_name} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            tx = x1
            ty = max(th, y1 - 2)
            cv2.putText(
                rear_resized,
                label,
                (tx, ty),
                font,
                font_scale,
                (0, 255, 0),
                thickness,
                cv2.LINE_AA,
            )

        y_end = min(PIP_Y + PIP_H, SCREEN_H)
        x_end = min(PIP_X + PIP_W, SCREEN_W)
        pip_h_eff = y_end - PIP_Y
        pip_w_eff = x_end - PIP_X

        if pip_h_eff > 0 and pip_w_eff > 0:
            front_canvas_fs[PIP_Y:y_end, PIP_X:x_end] = rear_resized[:pip_h_eff, :pip_w_eff]

        # 최종 HUD 전체 좌우 반전 (버드베스 / HUD 반사용)
        hud_flipped = cv2.flip(front_canvas_fs, 1)

        cv2.imshow("Front HUD", hud_flipped)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cv2.destroyWindow("Front HUD")
    cv2.destroyAllWindows()
    print("[B] Exit.")

if __name__ == "__main__":
    main()