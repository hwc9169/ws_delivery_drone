import cv2
import numpy as np
import asyncio
import os
import time

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

# --- 제어 설정값 ---
P_GAIN = 0.007             
LAND_ALT_10M = 10.0        
CONST_LAND_SPEED = 0.4     
ALIGN_THRESHOLD = 15       
FINAL_LAND_ALT = 0.4       

# 전역 변수
diff_x, diff_y = 0, 0
marker_found = False
current_alt = 0.0
latest_frame = None 

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
            cv2.imshow("Non-stop Descent Landing", latest_frame)
            cv2.waitKey(1)
        await asyncio.sleep(0.03)

async def run_control():
    global current_alt, marker_found, diff_x, diff_y
    drone = System()
    await drone.connect(system_address="udp://:14540")
    
    async def observe_altitude():
        global current_alt
        async for pos in drone.telemetry.position():
            current_alt = pos.relative_altitude_m
    asyncio.ensure_future(observe_altitude())

    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    try: await drone.offboard.start()
    except: return

    # --- [1단계] 마커 최초 포착 및 타겟 고정 정렬 ---
    print(">>> 1단계: 마커 최초 포착 대기...")
    while not marker_found:
        await asyncio.sleep(0.1)

    fixed_target_x = diff_x
    fixed_target_y = diff_y
    print(f">>> [Target Lock] 고정 좌표로 정렬 시작: X={fixed_target_x}, Y={fixed_target_y}")

    start_time = time.time()
    while True:
        vx, vy = -fixed_target_y * P_GAIN, fixed_target_x * P_GAIN
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(vx, vy, 0.0, 0.0))
        await asyncio.sleep(0.1)
        # 실시간 인식이 되어 오차가 줄어들거나 5초가 지나면 다음 단계로
        if marker_found and abs(diff_x) < ALIGN_THRESHOLD and abs(diff_y) < ALIGN_THRESHOLD:
            break
        if time.time() - start_time > 5.0:
            break

    # --- [2단계] 10m 고도까지 "무조건" 하강 ---
    # 인식 여부와 상관없이 수평 이동은 0으로 고정하고 하강만 수행
    print(f">>> 2단계: {LAND_ALT_10M}m까지 하강 시작")
    while current_alt > LAND_ALT_10M + 0.3:
        # 수평 속도 0, 하강 속도 0.8m/s (빠르게 진입)
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.8, 0.0))
        await asyncio.sleep(0.1)
    print(f">>> {LAND_ALT_10M}m 도달 완료.")

    # --- [3단계] 천천히 하강하며 1초마다 주기적 재정렬 ---
    print(">>> 3단계: 정밀 하강 및 주기적 재정렬 시작")
    while True:
        if current_alt < FINAL_LAND_ALT and marker_found and abs(diff_x) < ALIGN_THRESHOLD:
            print(">>> [최종 착륙] 명령")
            await drone.action.land()
            break

        # [A: 재정렬 하강] 1.2초 동안 실시간 마커 위치로 정렬
        start_time = time.time()
        while time.time() - start_time < 1.2:
            if current_alt < FINAL_LAND_ALT: break
            # 마커가 보일 때만 정렬 이동, 안 보이면 하강만 유지
            vx = -diff_y * P_GAIN if marker_found else 0.0
            vy = diff_x * P_GAIN if marker_found else 0.0
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(vx, vy, CONST_LAND_SPEED, 0.0))
            await asyncio.sleep(0.05)

        # [B: 안정화 하강] 1.0초 동안 수평 이동 중지하고 하강만 수행
        start_time = time.time()
        while time.time() - start_time < 1.0:
            if current_alt < FINAL_LAND_ALT: break
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, CONST_LAND_SPEED, 0.0))
            await asyncio.sleep(0.05)

async def main():
    node = Node()
    topic = "/world/aruco/model/x500_mono_cam_down_0/link/camera_link/sensor/camera/image"
    node.subscribe(Image, topic, camera_callback)
    await asyncio.gather(run_control(), display_loop())

if __name__ == "__main__":
    asyncio.run(main())