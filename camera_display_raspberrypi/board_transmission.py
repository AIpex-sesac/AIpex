from concurrent import futures
from picamera2 import Picamera2
from data_types_pb2 import CameraFrame, Command, ServerMessage
from ComputeService_pb2_grpc import ComputeServiceServicer, ComputeServiceStub
from app_comm_pb2 import JSONRequest, JSONResponse
import app_comm_pb2 as app_comm_pb2
import app_comm_pb2_grpc as app_comm_pb2_grpc
import grpc
import time
import json
import cv2
import numpy as np
import threading
import socket
from typing import Optional, Dict, List
from queue import Queue, Full
import subprocess

class BoardTransmission:
    def __init__(self, camera: Picamera2, server_address: str, buffer_size: int = 3, camera_id: int = 0, app_comm_address: str | None = None):
        """
        Args:
            camera: Picamera2 인스턴스
            server_address: gRPC 서버 주소
            buffer_size: 프레임 큐 크기
            camera_id: front=0, rear=1 등
            app_comm_address: app_comm 서비스 주소 (선택)
        """
        self.camera = camera
        self.camera_id = int(camera_id)
        self.server_address = self._resolve_address(server_address)
        self.running = False

        # 결과 보호용 락 및 초기값
        self.result_lock = threading.Lock()
        self.detection_result: Dict = {
            "width": 640,
            "height": 640,
            "detections": [],
            "count": 0
        }

        # app_comm에서 받은 임의 JSON 보관(선택) 및 락
        self.app_lock = threading.Lock()
        self.last_app_json = None

        # app_comm (optional)
        self.app_comm_address = app_comm_address if app_comm_address else "localhost:50053"
        self.app_comm_channel: Optional[grpc.Channel] = None
        self.app_comm_stub: Optional[AppCommServiceStub] = None
        self.app_comm_thread: Optional[threading.Thread] = None
        self.app_comm_running = False
        
        # 프레임 큐 (카메라 → gRPC)
        self.frame_queue: Queue = Queue(maxsize=buffer_size)
        
        # FPS 통계
        self.total_frames_sent = 0
        self.total_frames_received = 0
        self.total_frames_dropped = 0
        self.start_time: Optional[float] = None
        self.stats_lock = threading.Lock()

        # app_comm 서버용 변수 (로컬에서 SendJSON을 수신)
        self.app_server: Optional[grpc.Server] = None
        self.app_server_thread: Optional[threading.Thread] = None
        # 외부에서 접근 가능하도록 0.0.0.0으로 바인드 (또는 필요시 실제 LAN IP로 설정)
        self.app_server_bind: str = "0.0.0.0:50053"  # 로컬 바인드 주소 (요구대로 localhost:50051)

    def _resolve_address(self, address: str) -> str:
        """mDNS 주소를 IP로 해석"""
        try:
            if ':' in address:
                host, port = address.rsplit(':', 1)
            else:
                host = address
                port = "50053"
            
            ip = socket.gethostbyname(host)
            resolved = f"{ip}:{port}"
            print(f"[BoardTransmission] Resolved {address} -> {resolved}")
            return resolved
            
        except socket.gaierror:
            print(f"[BoardTransmission] Using address as-is: {address}")
            return address

    def connect(self):
        """gRPC 서버에 연결"""
        print(f"[BoardTransmission] Connecting to {self.server_address}...")
        
        # 연결 옵션 최적화
        options = [
            ('grpc.max_send_message_length', 10 * 1024 * 1024),
            ('grpc.max_receive_message_length', 10 * 1024 * 1024),
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 10000),
            ('grpc.keepalive_permit_without_calls', 1),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 10000),
        ]
        
        self.channel = grpc.insecure_channel(self.server_address, options=options)
        self.stub = ComputeServiceStub(self.channel)
        print("[BoardTransmission] Connected to gRPC server")

    def disconnect(self):
        """gRPC 연결 종료"""
        if self.channel:
            self.channel.close()
            print("[BoardTransmission] Disconnected from gRPC server")

    def _create_camera_frame(self, frame_data: bytes, frame_id: int) -> CameraFrame:
        """CameraFrame 메시지 생성"""
        from google.protobuf.timestamp_pb2 import Timestamp
        
        camera_frame = CameraFrame()
        camera_frame.image_data = frame_data
        
        timestamp = Timestamp()
        timestamp.GetCurrentTime()
        camera_frame.timestamp.CopyFrom(timestamp)
        
        camera_frame.width = 640
        camera_frame.height = 640
        camera_frame.format = "JPEG"
        camera_frame.camera_id = self.camera_id   # <-- 추가
        
        return camera_frame

    def _create_command(self, camera_frame: CameraFrame) -> Command:
        """Command 메시지 생성"""
        command = Command()
        command.camera_frame.CopyFrom(camera_frame)
        return command

    def _parse_detection_result(self, server_message: ServerMessage) -> Dict:
        """ServerMessage에서 detection 결과 파싱 및 디버그 로그 출력"""
        try:
            # 수신된 ServerMessage 필드 확인(디버그)
            try:
                fields = [f.name for f, _ in server_message.ListFields()]
                # print(f"[BoardTransmission] ServerMessage fields: {fields}")
            except Exception:
                pass

            if server_message.HasField('detection_result'):
                result_json = server_message.detection_result.json

                # 원문 로그
                # print(f"[BoardTransmission] Raw detection_result JSON: {result_json}")

                # 포맷된(가독성 있는) 로그
                try:
                    parsed = json.loads(result_json)
                    pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
                    # print(f"[BoardTransmission] Parsed detection_result:\n{pretty}")
                except Exception as e:
                    print(f"[BoardTransmission] Failed to pretty-print JSON: {e}")

                # 기존 파싱 로직 유지 (서버 형식: {"detections":[...], "count":N})
                try:
                    result = json.loads(result_json)
                    if isinstance(result, dict) and "detections" in result:
                        return {
                            "width": 640,
                            "height": 640,
                            "detections": result.get("detections", []),
                            "count": result.get("count", len(result.get("detections", [])))
                        }
                    elif isinstance(result, list):
                        return {
                            "width": 640,
                            "height": 640,
                            "detections": result,
                            "count": len(result)
                        }
                except Exception:
                    pass

            else:
                print("[BoardTransmission] No detection_result field in ServerMessage")

            return {"width": 640, "height": 640, "detections": [], "count": 0}

        except Exception as e:
            print(f"[BoardTransmission] Exception in _parse_detection_result: {e}")
            import traceback
            traceback.print_exc()
            return {"width": 640, "height": 640, "detections": [], "count": 0}

    def _capture_loop(self):
        """카메라 프레임 캡처 전용 스레드"""
        frame_id = 0
        print("[BoardTransmission] Capture loop started")
        
        while self.running:
            try:
                # 카메라에서 프레임 캡처
                frame_rgb = self.camera.capture_array("main")
                
                # 센터 크롭 및 리사이즈
                h, w = frame_rgb.shape[:2]
                if h > w:
                    start = (h - w) // 2
                    frame_square = frame_rgb[start:start+w, :]
                else:
                    start = (w - h) // 2
                    frame_square = frame_rgb[:, start:start+h]
                
                frame_640 = cv2.resize(frame_square, (640, 640), interpolation=cv2.INTER_LINEAR)
                
                # RGB → BGR → JPEG
                frame_bgr = cv2.cvtColor(frame_640, cv2.COLOR_RGB2BGR)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                ok, encoded = cv2.imencode(".jpg", frame_bgr, encode_param)
                
                if ok:
                    frame_data = encoded.tobytes()
                    
                    # 큐에 추가 (꽉 차면 오래된 프레임 드롭)
                    try:
                        self.frame_queue.put_nowait((frame_data, frame_id))
                        frame_id += 1
                    except Full:
                        # 큐가 꽉 참 - 오래된 프레임 버리고 새 프레임 추가
                        try:
                            self.frame_queue.get_nowait()
                            with self.stats_lock:
                                self.total_frames_dropped += 1
                        except:
                            pass
                        self.frame_queue.put_nowait((frame_data, frame_id))
                        frame_id += 1
                
            except Exception as e:
                print(f"[BoardTransmission] Capture error: {e}")
                time.sleep(0.01)
        
        print("[BoardTransmission] Capture loop stopped")

    def _request_generator(self):
        """gRPC 전송용 제너레이터 (큐에서 읽기)"""
        print("[BoardTransmission] Request generator started")
        
        while self.running:
            try:
                # 큐에서 프레임 가져오기 (타임아웃 0.1초)
                frame_data, frame_id = self.frame_queue.get(timeout=0.1)
                
                # CameraFrame 생성
                camera_frame = self._create_camera_frame(frame_data, frame_id)
                command = self._create_command(camera_frame)
                
                yield command
                
                # 전송 카운트 증가
                with self.stats_lock:
                    self.total_frames_sent += 1
                
            except:
                # 큐가 비었으면 계속 대기
                if not self.running:
                    break
                continue
        
        print("[BoardTransmission] Request generator stopped")

    def _stream_loop(self):
        """gRPC 양방향 스트림 루프"""
        try:
            print("[BoardTransmission] Starting gRPC stream...")
            
            with self.stats_lock:
                self.start_time = time.time()
            
            response_iterator = self.stub.Datastream(self._request_generator())
            
            for server_message in response_iterator:
                if not self.running:
                    break
                
                with self.stats_lock:
                    self.total_frames_received += 1
                
                # 서버 응답 파싱
                result = self._parse_detection_result(server_message)
                
                # 결과 업데이트
                with self.result_lock:
                    self.detection_result.clear()
                    self.detection_result.update(result)
                
        except grpc.RpcError as e:
            print(f"[BoardTransmission] gRPC error: {e.code()} - {e.details()}")
        except Exception as e:
            print(f"[BoardTransmission] Stream error: {e}")
        finally:
            print("[BoardTransmission] Stream ended")

    class _LocalAppCommServicer(app_comm_pb2_grpc.AppCommServiceServicer):
        """AppComm SendJSON 콜백을 받아 self.last_app_json에 저장하도록 위임"""
        def __init__(self, parent: "BoardTransmission"):
            self._parent = parent

        def SendJSON(self, request, context):
            payload = getattr(request, "json_payload", None)
            if payload is None:
                return app_comm_pb2.JSONResponse(success=False)
            try:
                parsed = json.loads(payload)
            except Exception:
                parsed = payload  # JSON 파싱 실패하면 원문 문자열로 보관

            with self._parent.app_lock:
                self._parent.last_app_json = parsed
                # 메타 정보 저장(원하면 사용)
                self._parent.last_app_json_ts = time.time()

            print(f"[BoardTransmission][app_comm_server] Received SendJSON, stored payload type: {type(parsed).__name__}")
            return app_comm_pb2.JSONResponse(success=True)

        def ReceiveJSON(self, request, context):
            # 간단히 동일한 동작을 지원(필요시 확장)
            payload = getattr(request, "json_payload", None)
            if payload is None:
                return app_comm_pb2.JSONResponse(success=False)
            try:
                parsed = json.loads(payload)
            except Exception:
                parsed = payload
            with self._parent.app_lock:
                self._parent.last_app_json = parsed
                self._parent.last_app_json_ts = time.time()
            print(f"[BoardTransmission][app_comm_server] Received ReceiveJSON, stored payload type: {type(parsed).__name__}")
            return app_comm_pb2.JSONResponse(success=True)

    def _start_app_comm_server(self):
        """로컬에서 AppComm gRPC 서버 시작 (SendJSON 수신)"""
        if self.app_server is not None:
            print("[BoardTransmission] app_comm server already running")
            return

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        servicer = BoardTransmission._LocalAppCommServicer(self)
        app_comm_pb2_grpc.add_AppCommServiceServicer_to_server(servicer, server)

        bound = server.add_insecure_port(self.app_server_bind)
        if bound == 0:
            print(f"[BoardTransmission] Failed to bind app_comm server to {self.app_server_bind}")
            return

        def _serve():
            print(f"[BoardTransmission] Starting app_comm server on {self.app_server_bind} (bound={bound})")
            server.start()
            # 서버는 stop이 호출될 때까지 대기
            server.wait_for_termination()
            print("[BoardTransmission] app_comm server stopped")

        t = threading.Thread(target=_serve, daemon=True)
        t.start()

        self.app_server = server
        self.app_server_thread = t
        print(f"[BoardTransmission] app_comm server thread started (bound={bound})")

    def _stop_app_comm_server(self):
        if self.app_server:
            try:
                print("[BoardTransmission] Stopping app_comm server...")
                self.app_server.stop(0)
            except Exception as e:
                print(f"[BoardTransmission] Error stopping app_comm server: {e}")
            self.app_server = None
            self.app_server_thread = None

    def _app_comm_loop(self):
        """주기적으로 app_comm ReceiveJSON 호출해서 포워딩된 JSON을 수신하여 detection으로 반영"""
        print("[BoardTransmission] app_comm loop started")
        if not self.app_comm_stub:
            print("[BoardTransmission] app_comm stub not configured, exiting app_comm loop")
            return

        while self.app_comm_running:
            try:
                req = JSONRequest(json_payload="")  # 빈 페이로드로 최신 포워딩 데이터 요청 (서버 구현에 따라 변경 가능)
                resp = self.app_comm_stub.ReceiveJSON(req, timeout=1.0)
                if getattr(resp, "success", False):
                    payload = getattr(resp, "message", "")
                    if payload:
                        try:
                            parsed = json.loads(payload)
                        except Exception:
                            # message가 이미 JSON object string이 아닐 수 있으므로 무시
                            parsed = None

                        if isinstance(parsed, dict):
                            # 서버가 {"detections": [...], "count": N, ...} 형태로 포워딩할 경우 처리
                            if "detections" in parsed:
                                with self.result_lock:
                                    self.detection_result.clear()
                                    self.detection_result.update({
                                        "width": parsed.get("width", 640),
                                        "height": parsed.get("height", 640),
                                        "detections": parsed.get("detections", []),
                                        "count": parsed.get("count", len(parsed.get("detections", [])))
                                    })
                                # print(f"[BoardTransmission][app_comm] Applied forwarded detections: {len(parsed.get('detections', []))}")
                            else:
                                # forwarded JSON이 detection 이외의 정보도 포함하면 로그만 남김
                                pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
                                # print(f"[BoardTransmission][app_comm] Received non-detection JSON:\n{pretty}")
                        elif isinstance(parsed, list):
                            with self.result_lock:
                                self.detection_result.clear()
                                self.detection_result.update({
                                    "width": 640,
                                    "height": 640,
                                    "detections": parsed,
                                    "count": len(parsed)
                                })
                            # print(f"[BoardTransmission][app_comm] Applied forwarded detection list: {len(parsed)}")
                # 주기 (실시간성이 필요하면 더 짧게 설정)
                time.sleep(0.2)
            except grpc.RpcError as e:
                # 연결 문제 등은 로그로 표시하고 잠시 대기
                print(f"[BoardTransmission][app_comm] gRPC error: {e.code()} - {e.details()}")
                time.sleep(1.0)
            except Exception as e:
                print(f"[BoardTransmission][app_comm] Exception: {e}")
                time.sleep(0.5)

        print("[BoardTransmission] app_comm loop stopped")

    def get_statistics(self) -> Dict:
        """통계 정보 반환"""
        with self.stats_lock:
            if self.start_time is None:
                return {
                    "total_frames_sent": 0,
                    "total_frames_received": 0,
                    "total_frames_dropped": 0,
                    "duration_sec": 0,
                    "avg_fps_sent": 0,
                    "avg_fps_received": 0
                }
            
            duration = time.time() - self.start_time
            
            return {
                "total_frames_sent": self.total_frames_sent,
                "total_frames_received": self.total_frames_received,
                "total_frames_dropped": self.total_frames_dropped,
                "duration_sec": duration,
                "avg_fps_sent": self.total_frames_sent / duration if duration > 0 else 0,
                "avg_fps_received": self.total_frames_received / duration if duration > 0 else 0
            }

    def start_streaming(self, start_app_server: bool = True):
        """스트리밍 시작"""
        if self.running:
            print("[BoardTransmission] Already streaming")
            return

        self.running = True

        # 캡처 스레드 시작
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        # gRPC 스트림 스레드 시작
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()

        # 로컬 app_comm 서버는 옵션으로만 시작 (프로세스당 1회 권장)
        if start_app_server:
            try:
                self._start_app_comm_server()
            except Exception as e:
                print(f"[BoardTransmission] Failed to start local app_comm server: {e}")

        print("[BoardTransmission] Streaming started")

    def stop_streaming(self):
        """스트리밍 중지"""
        self.running = False

        # 로컬 app_comm 서버 중지
        try:
            self._stop_app_comm_server()
        except Exception as e:
            print(f"[BoardTransmission] Error stopping app_comm server: {e}")

        # app_comm 스레드 종료 (이전 방식 호환성 유지를 위해)
        if self.app_comm_running:
            self.app_comm_running = False
            if self.app_comm_thread:
                self.app_comm_thread.join(timeout=2.0)
            if self.app_comm_channel:
                try:
                    self.app_comm_channel.close()
                except Exception:
                    pass

        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
        print("[BoardTransmission] Streaming stopped")

    def get_detection_result(self) -> Dict:
        """현재 detection 결과 반환"""
        with self.result_lock:
            return dict(self.detection_result)

    def get_last_app_json(self):
        """AppComm SendJSON으로 들어온 최신 JSON 반환"""
        with self.app_lock:
            if self.last_app_json is None:
                return None
            if isinstance(self.last_app_json, dict):
                # 원본 보호용 얕은 복사
                return dict(self.last_app_json)
            # dict가 아니면 있는 그대로 돌려줌 (문자열 등)
            return self.last_app_json

# 사용 예시
def main():

    global rear_frame, heading_deg

    # === 센서 모드 확인용 코드 추가 ===
    print("[DEBUG] Listing FRONT camera sensor modes...")
    cam_test = Picamera2(0)
    print(cam_test.sensor_modes)
    cam_test.close()
    # ==================================
    
    # Picamera2 설정
    print("[Main] Initializing camera...")
    picam = Picamera2(0)
    config = picam.create_video_configuration(
        main={"size": (1640, 1232), "format": "RGB888"}
    )
    picam.configure(config)
    picam.start()
    print("[Main] Camera started, waiting for stabilization...")
    time.sleep(0.5)
    
    # 서버 주소 설정
    server_address = "AipexFW.local:50051"
    
    # BoardTransmission 생성 및 연결
    transmission = BoardTransmission(
        camera=picam,
        server_address=server_address
    )
    
    try:
        transmission.connect()
        time.sleep(1.0)
        transmission.start_streaming()
        
        print("[Main] Starting result monitoring loop...")
        
        # FPS 계산용 변수
        last_time = time.time()
        frame_count = 0
        
        # 결과 출력 루프 - 빠르게 폴링
        while True:
            result = transmission.get_detection_result()
            detections = result.get('detections', [])
            
            frame_count += 1
            current_time = time.time()
            
            # 1초마다 FPS 출력
            if current_time - last_time >= 1.0:
                fps = frame_count / (current_time - last_time)
                print(f"[Main] FPS: {fps:.1f}, Detections: {len(detections)}")
                frame_count = 0
                last_time = current_time
            
            # 짧은 sleep으로 CPU 사용률 조절 (선택사항)
            time.sleep(0.001)  # 1ms - 거의 실시간
            
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user")
    finally:
        transmission.stop_streaming()
        
        # 통계 출력
        stats = transmission.get_statistics()
        print("\n" + "="*60)
        print("Session Statistics:")
        print("="*60)
        print(f"Duration:              {stats['duration_sec']:.2f} seconds")
        print(f"Total Frames Sent:     {stats['total_frames_sent']}")
        print(f"Total Frames Received: {stats['total_frames_received']}")
        print(f"Average FPS (Sent):    {stats['avg_fps_sent']:.2f}")
        print(f"Average FPS (Received):{stats['avg_fps_received']:.2f}")
        print("="*60 + "\n")
        
        transmission.disconnect()
        picam.close()
        print("[Main] Shutdown complete")


if __name__ == "__main__":
    main()
