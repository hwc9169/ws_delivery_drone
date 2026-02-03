import cv2
import numpy as np
import asyncio
import os
import time

# Protobuf 설정
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

# --- [제어 파라미터] ---
TARGET_SPEED = 1.2          
PIXEL_TO_M_FACTOR = 0.0017  
TRANS_ALT = 5.0             # 저고도 모드 진입 고도
DECEL_START_ALT = 7.0       # 부드러운 감속 시작 고도 (7m부터 감속)
LAND_READY_ALT = 1.0        

# 저고도 정밀 제어 설정
GAIN_LOW_ALT = 0.004        
MAX_SPEED_LOW = 0.2         
DEADZONE_LOW = 10           
DESCENT_LOW = 0.3           

# 전역 변수
current_alt = 0.0
latest_frame = None
first_detection = True  
diff_x, diff_y = 0, 0
marker_found = False

def camera_callback(msg):
    global diff_x, diff_y, marker_found, latest_frame
    try:
        frame = np.frombuffer(msg.data, dtype=np.uint8).copy().reshape((msg.height, msg.width, 3))
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, _ = detector.detectMarkers(processed_frame)
        if ids is not None:
            marker_found = True
            c = corners[0][0]
            diff_x = int((c[0][0] + c[2][0]) / 2) - (msg.width // 2)
            diff_y = int((c[0][1] + c[2][1]) / 2) - (msg.height // 2)
            cv2.aruco.drawDetectedMarkers(processed_frame, corners, ids)
        else:
            marker_found = False
        latest_frame = processed_frame
    except: pass

async def display_loop():
    global latest_frame
    while True:
        if latest_frame is not None:
            cv2.imshow("Smooth Transition Landing", latest_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        await asyncio.sleep(0.03)

async def run_control():
    global current_alt, marker_found, diff_x, diff_y, first_detection
    drone = System()
    await drone.connect(system_address="udp://:14540")
    
    async for state in drone.core.connection_state():
        if state.is_connected: break

    async def observe_altitude():
        global current_alt
        async for pos in drone.telemetry.position():
            current_alt = pos.relative_altitude_m
    asyncio.ensure_future(observe_altitude())

    while current_alt == 0: await asyncio.sleep(0.1)
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    try: await drone.offboard.start()
    except: return

    # --- [Step 1] 대각선 이동 및 부드러운 감속 (5m까지) ---
    print(">>> [Step 1] 마커 탐지 시 대각선 하강 및 부드러운 감속 시작")
    while first_detection:
        if marker_found:
            snap_alt = current_alt
            dist_x = diff_x * snap_alt * PIXEL_TO_M_FACTOR
            dist_y = -diff_y * snap_alt * PIXEL_TO_M_FACTOR
            total_dist = np.sqrt(dist_x**2 + dist_y**2)
            travel_time = total_dist / TARGET_SPEED

            if travel_time > 0.1:
                vz_orig = (snap_alt - TRANS_ALT) / travel_time
                vx_orig = (dist_y / total_dist) * TARGET_SPEED
                vy_orig = (dist_x / total_dist) * TARGET_SPEED

                start_time = time.time()
                while time.time() - start_time < travel_time:
                    # [핵심] 7m부터 5m까지 선형 감속 적용
                    if current_alt < DECEL_START_ALT:
                        # 7m에서 5m 사이의 비율 계산 (1.0 -> 0.0)
                        ratio = max(0.0, (current_alt - TRANS_ALT) / (DECEL_START_ALT - TRANS_ALT))
                        # 속도를 현재 비율에 맞춰 줄임 (최소 속도는 저고도 모드 속도와 맞춤)
                        vx = vx_orig * ratio
                        vy = vy_orig * ratio
                        vz = vz_orig * max(0.5, ratio) # 하강 속도는 너무 줄이지 않음
                    else:
                        vx, vy, vz = vx_orig, vy_orig, vz_orig

                    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(vx, vy, vz, 0.0))
                    await asyncio.sleep(0.05)
            
            print(">>> [전환] 감속 완료, 정밀 트래킹 모드 진입")
            first_detection = False
        await asyncio.sleep(0.1)

    # --- [Step 2] 연속 정밀 하강 (1m까지) ---
    while current_alt > LAND_READY_ALT:
        vx, vy, vz = 0.0, 0.0, DESCENT_LOW
        if marker_found:
            target_vx = -diff_y * GAIN_LOW_ALT if abs(diff_y) > DEADZONE_LOW else 0.0
            target_vy = diff_x * GAIN_LOW_ALT if abs(diff_x) > DEADZONE_LOW else 0.0
            vx = max(min(target_vx, MAX_SPEED_LOW), -MAX_SPEED_LOW)
            vy = max(min(target_vy, MAX_SPEED_LOW), -MAX_SPEED_LOW)
        else:
            vz = 0.0 
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(vx, vy, vz, 0.0))
        await asyncio.sleep(0.05)

    # --- [Step 3] 1m 지점 최종 정렬 및 대기 ---
    print(">>> [Step 3] 1m 도달. 최종 정렬을 위해 2.5초간 대기")
    stop_start_time = time.time()
    while time.time() - stop_start_time < 2.5:
        vx, vy = 0.0, 0.0
        if marker_found:
            vx = max(min(-diff_y * GAIN_LOW_ALT, MAX_SPEED_LOW), -MAX_SPEED_LOW)
            vy = max(min(diff_x * GAIN_LOW_ALT, MAX_SPEED_LOW), -MAX_SPEED_LOW)
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(vx, vy, 0.0, 0.0))
        await asyncio.sleep(0.05)

    print(">>> [Step 4] 착륙")
    await drone.action.land()
    os._exit(0)

async def main():
    node = Node()
    topic = "/world/aruco/model/x500_mono_cam_down_0/link/camera_link/sensor/camera/image"
    node.subscribe(Image, topic, camera_callback)
    await asyncio.gather(run_control(), display_loop())

if __name__ == "__main__":
    asyncio.run(main())