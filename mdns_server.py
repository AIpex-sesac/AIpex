import socket
import time
from zeroconf import ServiceInfo, Zeroconf

def register_service():
    # 1. 서비스 정보 정의
    # _<service>._<protocol>.local.
    service_type = "_compute._grpc._udp.local."
    service_name = "Compute Board gRPC Service"
    
    def get_local_ip():
        """
        외부로 나가는 경로의 로컬 IP를 반환.
        네트워크가 없거나 실패하면 '127.0.0.1'을 반환합니다.
        """
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # 실제로 연결을 수행하지 않음 — 라우팅 테이블을 통해 인터페이스 결정
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return '127.0.0.1'
    
    
    # 자신의 IP 주소와 포트 번호
    addr = socket.inet_aton(get_local_ip()) # 실제 IP로 변경하거나 자동 탐색 로직 사용
    port = 5005
    
    info = ServiceInfo(
        service_type,
        f"{service_name}.{service_type}",
        addresses=[addr],
        port=port,
        properties={'version': '1.0', 'os': 'raspberry-pi'},
        server=f"{socket.gethostname()}.local." # 현재 호스트 이름 사용
    )

    # 2. 서비스 등록
    zeroconf = Zeroconf()
    print(f"Registering service: {service_name} at port {port}")
    zeroconf.register_service(info)
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        # 서비스 해제 및 자원 정리
        print("Unregistering service...")
        zeroconf.unregister_service(info)
        zeroconf.close()

if __name__ == '__main__':
    register_service()