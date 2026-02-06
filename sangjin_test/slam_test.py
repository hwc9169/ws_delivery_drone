import cv2
import numpy as np
import asyncio
import math
import sys
from mavsdk import System
from gz.transport13 import Node
from gz.msgs10.laserscan_pb2 import LaserScan

# --- [1. 설정] ---
MAP_SIZE = 800
MAP_RES = 0.2
CENTER = MAP_SIZE // 2

# 전역 변수
grid_map = np.full((MAP_SIZE, MAP_SIZE), 127, dtype=np.uint8) # 127: 회색(알 수 없음)
drone_state = {'n': 0.0, 'e': 0.0, 'd': 0.0, 'yaw': 0.0, 'roll': 0.0, 'pitch': 0.0}

# --- [2. Lidar 콜백: 장애물(점) vs 통로(선) 분리] ---
def lidar_callback(msg):
    global grid_map
    ranges = np.array(msg.ranges)
    
    dn, de = drone_state['n'], drone_state['e']
    roll, pitch, yaw = drone_state['roll'], drone_state['pitch'], drone_state['yaw']
    
    # 지도 위 드론 현재 인덱스
    cn, ce = CENTER - int(dn/MAP_RES), CENTER + int(de/MAP_RES)

    angle_min = msg.angle_min
    angle_step = msg.angle_step

    for i, dist in enumerate(ranges):
        if 0.5 < dist < 30.0:
            beam_phi = angle_min + (i * angle_step)
            
            # 기울기 보정
            cos_tilt = math.cos(pitch) * math.cos(roll)
            corrected_dist = dist * cos_tilt
            
            # 기체가 너무 많이 기울어지면 노이즈 방지를 위해 스킵
            if abs(pitch) > math.radians(25) or abs(roll) > math.radians(25):
                continue

            total_angle = yaw + beam_phi
            obj_n = dn + corrected_dist * math.cos(total_angle)
            obj_e = de + corrected_dist * math.sin(total_angle)
            
            idx_n = CENTER - int(obj_n / MAP_RES)
            idx_e = CENTER + int(obj_e / MAP_RES)
            
            if 0 <= idx_n < MAP_SIZE and 0 <= idx_e < MAP_SIZE:
                # [핵심 변경] 
                # 1. 드론 위치에서 장애물 지점까지 '흰색(255)' 선을 그어 길을 닦음
                cv2.line(grid_map, (ce, cn), (idx_e, idx_n), 255, 1)
                
                # 2. 레이저가 부딪힌 '그 지점'만 검은색(0) 점으로 찍음
                grid_map[idx_n, idx_e] = 0

# --- [3. 텔레메트리 수신] ---
async def get_telemetry(drone):
    async def pos_loop():
        async for pos in drone.telemetry.position_velocity_ned():
            drone_state['n'], drone_state['e'] = pos.position.north_m, pos.position.east_m
    async def att_loop():
        async for att in drone.telemetry.attitude_euler():
            drone_state['roll'] = math.radians(att.roll_deg)
            drone_state['pitch'] = math.radians(att.pitch_deg)
            drone_state['yaw'] = math.radians(att.yaw_deg)
    await asyncio.gather(pos_loop(), att_loop())

# --- [4. 시각화] ---
async def display_map():
    while True:
        display_img = cv2.cvtColor(grid_map, cv2.COLOR_GRAY2BGR)
        
        cn, ce = CENTER - int(drone_state['n'] / MAP_RES), CENTER + int(drone_state['e'] / MAP_RES)
        if 0 <= cn < MAP_SIZE and 0 <= ce < MAP_SIZE:
            cv2.circle(display_img, (ce, cn), 5, (0, 0, 255), -1) # 드론 빨간 점
            
        cv2.imshow("Clean SLAM (Black=Obstacle, White=Path)", display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        await asyncio.sleep(0.05)

async def main():
    drone = System()
    await drone.connect(system_address="udp://:14540")
    
    node = Node()
    lidar_topic = "/world/baylands/model/x500_lidar_2d_0/link/link/sensor/lidar_2d_v2/scan"
    node.subscribe(LaserScan, lidar_topic, lidar_callback)
    
    await asyncio.gather(get_telemetry(drone), display_map())

if __name__ == "__main__":
    asyncio.run(main())