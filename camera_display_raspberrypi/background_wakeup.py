import argparse
import grpc
import wakeup_pb2 as pb2
import wakeup_pb2_grpc as pb2_grpc
import sys
import socket
import subprocess
import shutil
import glob
import os
from google.protobuf import empty_pb2

# ... existing constants and helper functions ...

def resolve_hostname(name: str) -> str:
    # try native getaddrinfo
    try:
        infos = socket.getaddrinfo(name, None)
        for info in infos:
            addr = info[4][0]
            # prefer IPv4 addresses for readability; return first
            if '.' in addr:
                return addr
        # if no IPv4, return first addr (likely IPv6)
        if infos:
            return infos[0][4][0]
    except Exception:
        pass

    # fallback: avahi-resolve -n <name>
    if shutil.which("avahi-resolve"):
        try:
            p = subprocess.run(["avahi-resolve", "-n", name], capture_output=True, text=True, timeout=2)
            out = p.stdout.strip()
            # expected: "name\tip"
            if out:
                parts = out.split()
                if len(parts) >= 2:
                    return parts[1].strip()
        except Exception:
            pass
    return name  # return original if cannot resolve

def send_wakeup(target: str, script_name: str, args: str, timeout: float = 5.0) -> int:
    channel = grpc.insecure_channel(target)
    stub = pb2_grpc.WakeUpServiceStub(channel)
    req = pb2.WakeUpRequest(script_name=script_name, args=args)
    try:
        resp = stub.TriggerScript(req, timeout=timeout)
        print(f"[send_wakeup] success={resp.success} pid={resp.process_id} msg='{resp.message}'")
        return 0 if resp.success else 2
    except grpc.RpcError as e:
        print(f"[send_wakeup] RPC failed: code={e.code()} msg={e.details()}", file=sys.stderr)
        return 1
    
def is_display_on() -> (bool, str):
    # pid 체크하고 값 리턴
    try:
        pids = _find_pids(scrpit_full)
        if pids:
            return True, "screen is on"
        else:
            return False, "screen is off"
    except:
        return False, "cannot check status"
# --- existing start/stop helpers (_find_pids, _terminate_pids, _start_script) remain unchanged ---

class WakeUpServiceServicer(pb2_grpc.WakeUpServiceServicer):
    def TriggerScript(self, request, context):
        # 기존 구현 (토글 start/stop) 유지
        script_name = (request.script_name or DEFAULT_SCRIPT).strip()
        script_full = str((ROOT / f"{script_name}.py").resolve())
        args = request.args or ""

        try:
            pids = _find_pids(script_full)
            if pids:
                _terminate_pids(pids)
                return pb2.WakeUpResponse(
                    success=True,
                    message=f"Terminated {script_name}. PIDs: {pids}",
                    process_id=0
                )
            else:
                proc = _start_script(script_name, args)
                return pb2.WakeUpResponse(
                    success=True,
                    message=f"Started {script_name}. PID: {proc.pid}",
                    process_id=proc.pid
                )
        except Exception as e:
            return pb2.WakeUpResponse(
                success=False,
                message=str(e),
                process_id=0
            )

    def IsDisplayOn(self, request, context):
        try:
            on, info = is_display_on()
            return pb2.DisplayState(on=on, info=info)
        except Exception as e:
            return pb2.DisplayState(on=False, info=f"error: {e}")
        
    def TriggerShutdown(self, request, context):
        import os
        
        os.system("sudo shutdown -h now")

# server start unchanged
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_WakeUpServiceServicer_to_server(WakeUpServiceServicer(), server)
    server.add_insecure_port('0.0.0.0:50050')
    server.start()
    print("🚀 WakeUp gRPC Server started on port 50050...")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)
        print("Server stopped.")

if __name__ == '__main__':
    serve()