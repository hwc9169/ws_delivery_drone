import asyncio
import random
import os
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw
from mavsdk.action import ActionError

async def setup_random_position():
    drone = System()
    # udp 주소 규격에 맞게 연결
    await drone.connect(system_address="udp://:14540")

    print(">>> 드론 연결 대기 중...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(">>> 드론 연결 성공!")
            break

    # --- 랜덤 위치 설정 및 이동 ---
    random_x = random.uniform(-6.0, 6.0)
    random_y = random.uniform(-6.0, 6.0)
    target_alt = -15.0 
    
    print(f">>> 랜덤 위치로 이동: X={random_x:.1f}, Y={random_y:.1f}, Alt=15m")

    # Offboard 진입을 위한 최소한의 데이터 전송
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))
    
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f">>> Offboard 시작 실패: {e._result.result}")
        return

    # 목표 위치로 이동
    await drone.offboard.set_position_ned(PositionNedYaw(random_x, random_y, target_alt, 0.0))
    await asyncio.sleep(5)
    
    print("-" * 40)
    print(f">>> 배치 완료! 이제 랜딩 코드를 실행하세요.")
    print("-" * 40)

if __name__ == "__main__":
    try:
        asyncio.run(setup_random_position())
    except KeyboardInterrupt:
        pass