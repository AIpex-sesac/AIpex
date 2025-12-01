import time
import socket
from zeroconf import ServiceBrowser, Zeroconf, ServiceStateChange

def on_service_state_change(zeroconf, service_type, name, state_change):
    if state_change is ServiceStateChange.Added:
        info = zeroconf.get_service_info(service_type, name)
        if info:
            # IP 주소와 포트 정보 추출
            address = socket.inet_ntoa(info.addresses[0])
            port = info.port
            print(f"Service ADDED: {name}")
            print(f"  -> Host: {info.server}, IP: {address}, Port: {port}")
            
            # 여기서 gRPC 클라이언트를 초기화하고 연결 시작 (address, port 사용)
            # init_grpc_client(address, port)

    elif state_change is ServiceStateChange.Removed:
        print(f"Service REMOVED: {name}")
    # elif state_change is ServiceStateChange.Update:
    #     print(f"Service UPDATED: {name}")

def browse_service():
    zeroconf = Zeroconf()
    service_type = "_compute._grpc._udp.local."
    
    # ServiceBrowser를 사용하여 서비스 검색 시작
    print(f"Browsing for service type {service_type}...")
    browser = ServiceBrowser(zeroconf, service_type, handlers=[on_service_state_change])
    
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        zeroconf.close()

if __name__ == '__main__':
    browse_service()