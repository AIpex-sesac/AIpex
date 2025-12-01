cd /home/xaqxaq/glaux_camera_display
source .env/bin/activate
/usr/bin/python3 /home/xaqxaq/glaux_camera_display/camera_client_grpc_hud.py \
  >> /home/xaqxaq/glaux_camera_display/hud.log 2>&1
