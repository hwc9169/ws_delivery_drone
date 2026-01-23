import asyncio
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw, OffboardError

async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    # 1. 스테이션 좌표 설정 (Gazebo 월드 내 목표 위치)
    # 시뮬레이션 기본 위치 근처의 임의 좌표입니다.
    TARGET_LAT = 47.397750
    TARGET_LON = 8.545607
    TARGET_ALT = 0.0  # 지면

    print("드론 연결 중...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("연결 성공!")
            break

    # 2. 이륙 및 초기 고도 확보
    await drone.action.arm()
    await drone.action.takeoff()
    await asyncio.sleep(8)

    # 3. Offboard 모드 시작 (정밀 제어)
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Offboard 시작 실패: {error}")
        return

    print("정밀 착륙 스테이션으로 접근 시작...")

    # 4. 루프 내에서 위치 보정 및 하강
    async for pos in drone.telemetry.position():
        current_lat = pos.latitude_deg
        current_lon = pos.longitude_deg
        current_alt = pos.relative_altitude_m

        # 위도/경도 오차 계산 (단순 차이값 이용)
        lat_error = TARGET_LAT - current_lat
        lon_error = TARGET_LON - current_lon

        # P-제어기: 오차에 비례하여 이동 속도 결정 (K_p 값은 튜닝 필요)
        K_p = 100000.0  # GPS 좌표 차이는 매우 작으므로 큰 값을 곱함
        vel_north = lat_error * K_p
        vel_east = lon_error * K_p

        # 고도에 따른 하강 속도 결정
        if current_alt > 2.0:
            vel_down = 0.5  # 접근 단계
        else:
            vel_down = 0.2  # 최종 접지 단계

        # 속도 제한 (너무 빠르지 않게)
        vel_north = max(min(vel_north, 1.0), -1.0)
        vel_east = max(min(vel_east, 1.0), -1.0)

        print(f"고도: {current_alt:.2f}m | 오차보정(N, E): {vel_north:.2f}, {vel_east:.2f}")

        # 드론에게 속도 명령 전송
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(vel_north, vel_east, vel_down, 0)
        )

        # 지면 도착 판단
        if current_alt < 0.2:
            print("스테이션 착륙 완료!")
            break

    # 5. 시동 끄고 종료
    await drone.offboard.stop()
    await drone.action.land()

if __name__ == "__main__":
    asyncio.run(run())