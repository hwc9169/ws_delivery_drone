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
P_GAIN_NORMAL = 0.007      # 일반 하강 시 정렬 감도
P_GAIN_SOFT = 0.003        # 최종 착륙 전 초정밀 정렬 감도 (매우 천천히)
DESCENT_SPEED_FAST = 0.7   # 5m까지 급강하 속도
DESCENT_SPEED_SLOW = 0.3   # 정밀 하강 속도
ALIGN_THRESHOLD = 12       
FINAL_LAND_ALT = 0.5       
STABILIZE_TIME = 2.0       

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
            cv2.imshow("Ultra Precision Landing", latest_frame)
            if cv2.waitKey(10) & 0xFF == ord('q'): break
        await asyncio.sleep(0.02)

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

    # [1, 2, 3단계] 마커 인식 -> 정렬 -> 자세 안정화 (2초)
    print(">>> 1-3단계: 마커 포착 및 초기 안정화 시작")
    while not marker_found: await asyncio.sleep(0.1)
    
    while True:
        vx, vy = -diff_y * P_GAIN_NORMAL, diff_x * P_GAIN_NORMAL
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(vx, vy, 0.0, 0.0))
        await asyncio.sleep(0.1)
        if marker_found and abs(diff_x) < ALIGN_THRESHOLD and abs(diff_y) < ALIGN_THRESHOLD:
            print(">>> 초기 정렬 완료. 2초간 대기하며 자세 안정화...")
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            await asyncio.sleep(2.0)
            break

    # [4단계] 5m 고도 이상이라면 5m까지 하강
    if current_alt > 5.0:
        print(f">>> 4단계: 고도 {current_alt:.1f}m -> 5m까지 하강")
        while current_alt > 5.2:
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, DESCENT_SPEED_FAST, 0.0))
            await asyncio.sleep(0.1)

    # [5단계] 주기적 마커 정렬하며 1m 고도까지 하강
    print(">>> 5단계: 1m 고도까지 주기적 정렬 하강")
    while current_alt > 1.0:
        # 1초 정렬 하강 / 0.8초 수직 하강 반복
        vx = -diff_y * P_GAIN_NORMAL if marker_found else 0.0
        vy = diff_x * P_GAIN_NORMAL if marker_found else 0.0
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(vx, vy, DESCENT_SPEED_SLOW, 0.0))
        await asyncio.sleep(1.0)
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, DESCENT_SPEED_SLOW, 0.0))
        await asyncio.sleep(0.8)

    # [6, 7단계] 마지막 마커 정렬 (초정밀/저속) 및 2초 안정화
    print(">>> 6-7단계: 최종 고도 도달. 초정밀 정렬 및 최종 안정화 시작")
    # 0.5m 고도까지 아주 천천히 내려가며 정렬
    while current_alt > FINAL_LAND_ALT:
        vx = -diff_y * P_GAIN_SOFT if marker_found else 0.0
        vy = diff_x * P_GAIN_SOFT if marker_found else 0.0
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(vx, vy, 0.15, 0.0))
        await asyncio.sleep(0.1)

    print(f">>> 마지막 자세 안정화 대기 ({STABILIZE_TIME}초)")
    start_time = time.time()
    while time.time() - start_time < STABILIZE_TIME:
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        await asyncio.sleep(0.05)

    # [8단계] 그대로 착륙
    print(">>> 8단계: 최종 착륙 실행")
    await drone.action.land()

async def main():
    node = Node()
    topic = "/world/aruco/model/x500_mono_cam_down_0/link/camera_link/sensor/camera/image"
    node.subscribe(Image, topic, camera_callback)
    await asyncio.gather(run_control(), display_loop())

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: cv2.destroyAllWindows()