import cv2
import numpy as np
import asyncio
import os
import time
import math
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw
from gz.transport13 import Node
from gz.msgs10.laserscan_pb2 import LaserScan

# --- [1. 제어 및 우회 파라미터] ---
TARGET_SPEED = 1.8         # 기본 비행 속도
OBSTACLE_THRESHOLD = 6.0   # 장애물 감지 거리
REPULSIVE_GAIN = 5.0       # 밀어내는 힘 (척력)
TANGENT_GAIN = 4.0         # 옆으로 흐르는 힘 (접선력 - 우회의 핵심)
ACCEPTANCE_RADIUS = 2.0

# 전역 변수
lidar_ranges = []
lidar_avoid_vector = np.array([0.0, 0.0])
is_blocked_front = False   # 정면 장애물 여부

# --- [2. Lidar 콜백: 접선력(우회) 계산] ---
def lidar_callback(msg):
    global lidar_avoid_vector, lidar_ranges, is_blocked_front
    lidar_ranges = np.array(msg.ranges)
    repulse_n, repulse_e = 0.0, 0.0
    
    num_beams = len(lidar_ranges)
    # 정면 30도 범위 내 밀도 체크
    front_idx = range(int(num_beams*5.5/12), int(num_beams*6.5/12))
    if np.min(lidar_ranges[front_idx]) < 2.5:
        is_blocked_front = True
    else:
        is_blocked_front = False

    for i, dist in enumerate(lidar_ranges):
        if 0.2 < dist < OBSTACLE_THRESHOLD:
            angle = msg.angle_min + i * msg.angle_step
            # 뒤쪽 120도는 무시 (전방 시야 집중)
            if abs(angle) > math.pi * (120/180): continue
            
            # 척력 강도 (가까울수록 폭발적으로 증가)
            force = (3.0 / (dist ** 3))
            
            # 기본 척력 (장애물 반대 방향)
            fn = -force * math.cos(angle)
            fe = -force * math.sin(angle)
            
            # [핵심] 접선력 추가 (90도 회전시켜 옆으로 흐르게 함)
            # 수직 벡터를 생성하여 장애물을 끼고 돌게 만듭니다.
            tn = -fe * TANGENT_GAIN
            te = fn * TANGENT_GAIN
            
            repulse_n += (fn + tn)
            repulse_e += (fe + te)

    lidar_avoid_vector = np.array([repulse_n, repulse_e]) * REPULSIVE_GAIN

# --- [3. 시각화 및 레이더 창] ---
async def display_loop():
    while True:
        radar = np.zeros((600, 600, 3), dtype=np.uint8)
        center = (300, 300)
        # 드론 표시 (정면이 막히면 노란색으로 경고)
        color = (0, 255, 255) if is_blocked_front else (0, 255, 0)
        cv2.circle(radar, center, 8, color, -1)
        
        if len(lidar_ranges) > 0:
            step = (2 * math.pi) / len(lidar_ranges)
            for i, dist in enumerate(lidar_ranges):
                if 0.1 < dist < 20.0:
                    angle = i * step
                    x = int(center[0] + (dist * 15) * math.sin(angle))
                    y = int(center[1] - (dist * 15) * math.cos(angle))
                    cv2.circle(radar, (x, y), 2, (0, 0, 255), -1)

        cv2.imshow("Tangent Avoidance Radar", radar)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        await asyncio.sleep(0.05)

# --- [4. 비행 미션 로직] ---
async def run_mission():
    global lidar_avoid_vector, is_blocked_front
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print(">>> 우회 주행 시스템 가동 중...")
    async for state in drone.core.connection_state():
        if state.is_connected: break

    goal_n, goal_e = 100.0, 100.0

    await drone.action.arm()
    await drone.action.takeoff()
    await asyncio.sleep(5)
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
    await drone.offboard.start()

    while True:
        async for nav in drone.telemetry.position_velocity_ned():
            curr_n, curr_e = nav.position.north_m, nav.position.east_m
            break
        
        dist_to_goal = math.sqrt((goal_n-curr_n)**2 + (goal_e-curr_e)**2)
        if dist_to_goal < ACCEPTANCE_RADIUS: break

        # 1. 인력(목적지) 계산
        dir_n, dir_e = goal_n-curr_n, goal_e-curr_e
        norm = math.sqrt(dir_n**2 + dir_e**2)
        attract_v = np.array([dir_n/norm, dir_e/norm]) * TARGET_SPEED
        
        # 2. 정면 장애물 발생 시 전진 속도 감소 (우회할 시간 확보)
        if is_blocked_front:
            attract_v *= 0.4 

        # 3. 최종 벡터 합성 (인력 + 척력 + 접선력)
        final_v = attract_v + lidar_avoid_vector
        
        # 속도 제한
        v_mag = np.linalg.norm(final_v)
        if v_mag > TARGET_SPEED:
            final_v = (final_v / v_mag) * TARGET_SPEED

        await drone.offboard.set_velocity_ned(VelocityNedYaw(final_v[0], final_v[1], 0.0, 0.0))
        await asyncio.sleep(0.1)

    print(">>> 장거리 우회 주행 완료!")
    await drone.action.land()
    os._exit(0)

async def main():
    node = Node()
    lidar_topic = "/world/baylands/model/x500_lidar_2d_0/link/link/sensor/lidar_2d_v2/scan"
    node.subscribe(LaserScan, lidar_topic, lidar_callback)
    await asyncio.gather(run_mission(), display_loop())

if __name__ == "__main__":
    asyncio.run(main())