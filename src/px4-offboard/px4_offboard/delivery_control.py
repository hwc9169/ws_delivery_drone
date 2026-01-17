#!/usr/bin/env python3

import asyncio
import math
import threading
import time
from typing import Any, Dict, Optional
from enum import Enum

import yaml
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger

from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw
from mavsdk.telemetry import LandedState
from ament_index_python.packages import get_package_share_directory


class DeliveryState(Enum):
    READY = "READY"
    GO_TO_PROVIDER = "GO_TO_PROVIDER"
    WAITING_PICKUP = "WAITING_PICKUP"
    GO_TO_CONSUMER = "GO_TO_CONSUMER"
    WAITING_DELIVER = "WAITING_DELIVER"
    DONE = "DONE"
    ABORT = "ABORT"


class DeliveryNode(Node):
    def __init__(self) -> None:
        super().__init__("delivery_control")

        default_yaml = self._default_station_yaml()
        self.declare_parameter("station_yaml", default_yaml)
        self.declare_parameter("system_address", "udpin://:14540")
        self.declare_parameter("auto_start", True)
        self.declare_parameter("auto_start_delay_s", 2.0)
        self.declare_parameter("loop_delivery", False)
        self.declare_parameter("setpoint_rate_hz", 20.0)
        self.declare_parameter("yaw_deg", 0.0)

        self._state = DeliveryState.READY
        self._start_requested = False
        self._pickup_confirmed = False
        self._deliver_confirmed = False
        self._auto_start_deadline = None

        self._config: Dict[str, Any] = self._load_config(
            self.get_parameter("station_yaml").get_parameter_value().string_value
        )
        self._system_address = (
            self.get_parameter("system_address").get_parameter_value().string_value
        )
        self._setpoint_rate_hz = (
            self.get_parameter("setpoint_rate_hz").get_parameter_value().double_value
        )
        self._yaw_deg = self.get_parameter("yaw_deg").get_parameter_value().double_value

        self._drone: Optional[System] = None
        self._stream_task: Optional[asyncio.Task] = None
        self._streaming = False
        self._target_setpoint: Optional[PositionNedYaw] = None

        self._last_position = None
        self._last_landed_state: Optional[LandedState] = None
        self._last_armed: Optional[bool] = None

        self._start_srv = self.create_service(
            Trigger, "/start_delivery", self._handle_start_delivery
        )
        self._pickup_srv = self.create_service(
            Trigger, "/confirm_pickup", self._handle_confirm_pickup
        )
        self._deliver_srv = self.create_service(
            Trigger, "/confirm_deliver", self._handle_confirm_deliver
        )

        self._auto_start_timer = self.create_timer(0.5, self._auto_start_tick)

    def _default_station_yaml(self) -> str:
        try:
            share_dir = get_package_share_directory("px4_offboard")
            return f"{share_dir}/station_locations.yaml"
        except Exception:
            return "station_locations.yaml"

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)

        mode = config.get("mode", "local")

        def _required_global(name: str) -> dict:
            station = config.get(name)
            if not station or "lat" not in station or "lon" not in station:
                raise ValueError(f"Missing required global station: {name}")
            return {
                "lat": float(station.get("lat", 0.0)),
                "lon": float(station.get("lon", 0.0)),
                "alt": float(station.get("alt", 0.0)),
            }

        return {
            "mode": mode,
            "provider_global": _required_global("provider_global"),
            "consumer_global": _required_global("consumer_global"),
            "safe_alt_m": float(config.get("safe_alt_m", 10.0)),
            "approach_alt_m": float(config.get("approach_alt_m", 3.0)),
            "xy_accept_m": float(config.get("xy_accept_m", 0.5)),
            "z_accept_m": float(config.get("z_accept_m", 0.3)),
            "max_global_range_m": float(config.get("max_global_range_m", 20000.0)),
            "timeouts_s": {
                "takeoff": float(config.get("timeouts_s", {}).get("takeoff", 20)),
                "cruise": float(config.get("timeouts_s", {}).get("cruise", 120)),
                "approach": float(config.get("timeouts_s", {}).get("approach", 30)),
                "land": float(config.get("timeouts_s", {}).get("land", 30)),
            },
            "wait_s": {
                "pickup": float(config.get("wait_s", {}).get("pickup", 10)),
                "deliver": float(config.get("wait_s", {}).get("deliver", 10)),
            },
        }

    def _handle_start_delivery(self, request, response):
        self._start_requested = True
        response.success = True
        response.message = "Delivery start requested."
        return response

    def _handle_confirm_pickup(self, request, response):
        self._pickup_confirmed = True
        response.success = True
        response.message = "Pickup confirmed."
        return response

    def _handle_confirm_deliver(self, request, response):
        self._deliver_confirmed = True
        response.success = True
        response.message = "Delivery confirmed."
        return response

    def _auto_start_tick(self) -> None:
        if not self.get_parameter("auto_start").get_parameter_value().bool_value:
            return
        if self._start_requested:
            return
        if self._auto_start_deadline is None:
            delay = self.get_parameter("auto_start_delay_s").get_parameter_value().double_value
            self._auto_start_deadline = time.monotonic() + delay
        if time.monotonic() >= self._auto_start_deadline:
            self.get_logger().info("Auto-starting delivery.")
            self._start_requested = True

    async def run(self) -> None:
        await self._connect_mavsdk()
        await self._start_telemetry()
        await self._prepare_targets()

        self.get_logger().info("Delivery service ready.")
        while rclpy.ok():
            if self._state == DeliveryState.READY:
                await self._handle_ready()
            elif self._state == DeliveryState.GO_TO_PROVIDER:
                await self._handle_go_to_provider()
            elif self._state == DeliveryState.WAITING_PICKUP:
                await self._handle_waiting_pickup()
            elif self._state == DeliveryState.GO_TO_CONSUMER:
                await self._handle_go_to_consumer()
            elif self._state == DeliveryState.WAITING_DELIVER:
                await self._handle_waiting_deliver()
            elif self._state == DeliveryState.DONE:
                await self._handle_done()
            elif self._state == DeliveryState.ABORT:
                await self._handle_abort()
            await asyncio.sleep(0.1)

    async def _connect_mavsdk(self) -> None:
        self.get_logger().info(f"Connecting MAVSDK: {self._system_address}")
        self._drone = System()
        await self._drone.connect(system_address=self._system_address)
        async for state in self._drone.core.connection_state():
            if state.is_connected:
                self.get_logger().info("MAVSDK connected.")
                break

    async def _start_telemetry(self) -> None:
        self._require_drone()
        asyncio.create_task(self._track_position())
        asyncio.create_task(self._track_landed_state())
        asyncio.create_task(self._track_armed())

    async def _track_position(self) -> None:
        drone = self._require_drone()
        async for pos in drone.telemetry.position_velocity_ned():
            self._last_position = pos

    async def _track_landed_state(self) -> None:
        drone = self._require_drone()
        async for state in drone.telemetry.landed_state():
            self._last_landed_state = state

    async def _track_armed(self) -> None:
        drone = self._require_drone()
        async for armed in drone.telemetry.armed():
            self._last_armed = armed

    async def _handle_ready(self) -> None:
        if not self._start_requested:
            return
        self._transition(DeliveryState.GO_TO_PROVIDER)

    async def _handle_go_to_provider(self) -> None:
        target = await self._get_target_for("provider_global")
        if target is None:
            self.get_logger().error("Missing provider_global in config.")
            self._transition(DeliveryState.ABORT)
            return
        success = await self._fly_to_local(target, "provider")
        self._transition(DeliveryState.WAITING_PICKUP if success else DeliveryState.ABORT)

    async def _handle_waiting_pickup(self) -> None:
        wait_s = self._config["wait_s"]["pickup"]
        self.get_logger().info(f"Waiting pickup ({wait_s}s) or /confirm_pickup.")
        await self._wait_for_confirmation(wait_s, "_pickup_confirmed")
        self._transition(DeliveryState.GO_TO_CONSUMER)

    async def _handle_go_to_consumer(self) -> None:
        target = await self._get_target_for("consumer_global")
        if target is None:
            self.get_logger().error("Missing consumer_global in config.")
            self._transition(DeliveryState.ABORT)
            return
        success = await self._fly_to_local(target, "consumer")
        self._transition(DeliveryState.WAITING_DELIVER if success else DeliveryState.ABORT)

    async def _handle_waiting_deliver(self) -> None:
        wait_s = self._config["wait_s"]["deliver"]
        self.get_logger().info(f"Waiting delivery ({wait_s}s) or /confirm_deliver.")
        await self._wait_for_confirmation(wait_s, "_deliver_confirmed")
        self._transition(DeliveryState.DONE)

    async def _handle_done(self) -> None:
        self.get_logger().info("Delivery cycle complete.")
        self._start_requested = False
        self._pickup_confirmed = False
        self._deliver_confirmed = False
        self._auto_start_deadline = None

        if self.get_parameter("loop_delivery").get_parameter_value().bool_value:
            self._transition(DeliveryState.READY)
        else:
            await asyncio.sleep(1.0)

    async def _handle_abort(self) -> None:
        self.get_logger().error("Mission aborted. Landing for safety.")
        try:
            await self._stop_offboard()
        except Exception:
            pass
        try:
            drone = self._require_drone()
            await drone.action.hold()
        except Exception:
            pass
        try:
            drone = self._require_drone()
            await drone.action.land()
        except Exception as exc:
            self.get_logger().error(f"Land command failed: {exc}")
        self._transition(DeliveryState.DONE)

    def _require_drone(self) -> System:
        if self._drone is None:
            raise RuntimeError("MAVSDK system is not connected.")
        return self._drone

    async def _wait_for_confirmation(self, wait_s: float, flag_name: str) -> None:
        start = time.monotonic()
        while time.monotonic() - start < wait_s:
            if getattr(self, flag_name):
                return
            await asyncio.sleep(0.2)

    async def _prepare_targets(self) -> None:
        mode = self._config.get("mode", "local")
        if mode != "global":
            raise ValueError(f"Unsupported mode: {mode}")

    async def _get_target_for(self, key: str) -> Optional[Dict[str, float]]:
        target_global = self._config.get(key)
        if target_global is None:
            return None

        reference = await self._get_global_reference()
        if reference is None:
            self.get_logger().error("Failed to get global reference for conversion.")
            return None

        self.get_logger().info(
            "Global reference set (lat=%.7f, lon=%.7f, alt=%.2f)."
            % (reference["lat"], reference["lon"], reference["alt"])
        )
        self.get_logger().info(
            "Global target set (lat=%.7f, lon=%.7f, alt=%.2f)."
            % (target_global["lat"], target_global["lon"], target_global["alt"])
        )
        target_local = self.global_to_local_ned_m(reference, target_global)
        self.get_logger().info(
            "Converted target to local NED (x=%.3f, y=%.3f, z=%.3f)."
            % (target_local["x"], target_local["y"], target_local["z"])
        )

        max_range = self._config["max_global_range_m"]
        if not self._within_range(target_local, max_range):
            self.get_logger().error("Global target is too far from reference.")
            return None

        return target_local

    async def _get_global_reference(self) -> Optional[Dict[str, float]]:
        drone = self._require_drone()

        async def _first_position():
            async for pos in drone.telemetry.position():
                return pos

        try:
            pos = await asyncio.wait_for(_first_position(), timeout=5.0)
            if pos is None:
                self.get_logger().error("Failed to get position.")
                return None

            return {
                "lat": pos.latitude_deg,
                "lon": pos.longitude_deg,
                "alt": pos.absolute_altitude_m,
            }
        except Exception:
            return None

    def global_to_local_enu_m(
        self,
        reference: Dict[str, float],
        target: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Convert global GPS (lat, lon, alt) -> local ENU meters relative to reference.

        Inputs:
          reference: (lat_deg, lon_deg, alt_m)
          target:    (lat_deg, lon_deg, alt_m)

        Output:
          (x_m, y_m, z_m) in meters where:
            x = East  (meters)
            y = North (meters)
            z = Up    (meters)

        If you want PX4 NED instead:
          north = y
          east  = x
          down  = -z
        """
        # WGS84 constants
        a = 6378137.0                    # semi-major axis (m)
        f = 1.0 / 298.257223563          # flattening
        e2 = f * (2.0 - f)               # eccentricity^2

        def deg2rad(d: float) -> float:
            return d * math.pi / 180.0

        def lla_to_ecef(lat_deg: float, lon_deg: float, alt_m: float):
            lat = deg2rad(lat_deg)
            lon = deg2rad(lon_deg)

            sin_lat = math.sin(lat)
            cos_lat = math.cos(lat)
            sin_lon = math.sin(lon)
            cos_lon = math.cos(lon)

            N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)

            x = (N + alt_m) * cos_lat * cos_lon
            y = (N + alt_m) * cos_lat * sin_lon
            z = (N * (1.0 - e2) + alt_m) * sin_lat
            return x, y, z

        lat0, lon0, h0 = reference["lat"], reference["lon"], reference["alt"]
        lat1, lon1, h1 = target["lat"], target["lon"], target["alt"]

        # ECEF positions
        x0, y0, z0 = lla_to_ecef(lat0, lon0, h0)
        x1, y1, z1 = lla_to_ecef(lat1, lon1, h1)

        # Delta in ECEF
        dx, dy, dz = (x1 - x0), (y1 - y0), (z1 - z0)

        # ECEF -> ENU rotation at reference
        lat0r = deg2rad(lat0)
        lon0r = deg2rad(lon0)

        sin_lat0 = math.sin(lat0r)
        cos_lat0 = math.cos(lat0r)
        sin_lon0 = math.sin(lon0r)
        cos_lon0 = math.cos(lon0r)

        east  = -sin_lon0 * dx + cos_lon0 * dy
        north = -sin_lat0 * cos_lon0 * dx - sin_lat0 * sin_lon0 * dy + cos_lat0 * dz
        up    =  cos_lat0 * cos_lon0 * dx + cos_lat0 * sin_lon0 * dy + sin_lat0 * dz

        return {"x": east, "y": north, "z": up}

    def global_to_local_ned_m(
        self,
        reference: Dict[str, float],
        target: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Convert global GPS (lat, lon, alt) -> local NED meters relative to reference.

        Output:
          (x_m, y_m, z_m) where:
            x = North (meters)
            y = East  (meters)
            z = Down  (meters)  [PX4 convention]
        """
        local = self.global_to_local_enu_m(reference, target)
        return {"x": local["y"], "y": local["x"], "z": -local["z"]}

    def _within_range(self, target: Dict[str, float], max_range_m: float) -> bool:
        distance = (target["x"] ** 2 + target["y"] ** 2) ** 0.5
        return distance <= max_range_m

    def _transition(self, new_state: DeliveryState) -> None:
        if self._state != new_state:
            self.get_logger().info(f"{self._state.value} -> {new_state.value}")
            self._state = new_state

    async def _fly_to_local(self, target: Dict[str, float], label: str) -> bool:
        drone = self._require_drone()
        if not await self._ensure_armed():
            return False

        current = await self._wait_for_position_ready()
        if current is None:
            self.get_logger().error("No position telemetry; aborting.")
            return False

        safe_down = target["z"] - self._config["safe_alt_m"]
        approach_down = target["z"] - self._config["approach_alt_m"]

        await self._start_offboard_stream(
            PositionNedYaw(
                current.position.north_m,
                current.position.east_m,
                current.position.down_m,
                self._yaw_deg,
            )
        )
        if not await self._start_offboard():
            await self._stop_streaming()
            return False

        self.get_logger().info(f"{label}: TAKEOFF to {safe_down:.2f} down.")
        self._update_target(current.position.north_m, current.position.east_m, safe_down)
        if not await self._wait_for_position(
            current.position.north_m,
            current.position.east_m,
            safe_down,
            check_xy=False,
            timeout_s=self._config["timeouts_s"]["takeoff"],
        ):
            await self._abort_offboard("takeoff timeout")
            return False

        self.get_logger().info(f"{label}: CRUISE to ({target['x']:.6f}, {target['y']:.6f}).")
        self._update_target(target["x"], target["y"], safe_down)
        if not await self._wait_for_position(
            target["x"],
            target["y"],
            safe_down,
            check_xy=True,
            timeout_s=self._config["timeouts_s"]["cruise"],
        ):
            await self._abort_offboard("cruise timeout")
            return False

        self.get_logger().info(f"{label}: APPROACH to {approach_down:.3f} down.")
        self._update_target(target["x"], target["y"], approach_down)
        if not await self._wait_for_position(
            target["x"],
            target["y"],
            approach_down,
            check_xy=True,
            timeout_s=self._config["timeouts_s"]["approach"],
        ):
            await self._abort_offboard("approach timeout")
            return False

        await self._stop_offboard()
        try:
            await drone.action.hold()
        except Exception:
            pass
        self.get_logger().info(f"{label}: LAND.")
        try:
            await drone.action.land()
        except Exception as exc:
            self.get_logger().error(f"Land command failed: {exc}")
            return False

        landed = await self._wait_for_landed(self._config["timeouts_s"]["land"])
        if not landed:
            self.get_logger().error("Land timeout.")
            return False

        return True

    async def _ensure_armed(self) -> bool:
        drone = self._require_drone()
        if self._last_armed:
            return True
        try:
            await drone.action.arm()
        except Exception as exc:
            self.get_logger().error(f"Arm failed: {exc}")
            return False

        start = time.monotonic()
        while time.monotonic() - start < 5.0:
            if self._last_armed:
                return True
            await asyncio.sleep(0.1)
        self.get_logger().error("Arm timeout.")
        return False

    async def _wait_for_position_ready(self) -> Optional[Any]:
        start = time.monotonic()
        while time.monotonic() - start < 5.0:
            if self._last_position is not None:
                return self._last_position
            await asyncio.sleep(0.1)
        return None

    def _update_target(self, x: float, y: float, z: float) -> None:
        self._target_setpoint = PositionNedYaw(x, y, z, self._yaw_deg)

    async def _start_offboard_stream(self, initial_setpoint: PositionNedYaw) -> None:
        self._target_setpoint = initial_setpoint
        self._streaming = True
        self._stream_task = asyncio.create_task(self._stream_setpoints())
        await asyncio.sleep(0.1)

    async def _prime_offboard_setpoints(self) -> None:
        drone = self._require_drone()
        if self._target_setpoint is None:
            return
        for _ in range(15):
            try:
                await drone.offboard.set_position_ned(self._target_setpoint)
            except Exception:
                pass
            await asyncio.sleep(0.05)

    async def _stream_setpoints(self) -> None:
        drone = self._require_drone()
        period = 1.0 / max(self._setpoint_rate_hz, 1.0)
        while self._streaming:
            if self._target_setpoint is None:
                await asyncio.sleep(period)
                continue
            try:
                await drone.offboard.set_position_ned(self._target_setpoint)
            except Exception:
                pass
            await asyncio.sleep(period)

    async def _start_offboard(self) -> bool:
        drone = self._require_drone()
        await self._prime_offboard_setpoints()
        try:
            await drone.offboard.start()
            return True
        except OffboardError as exc:
            self.get_logger().error(f"Offboard start failed: {exc}")
            await self._prime_offboard_setpoints()
            try:
                await drone.offboard.start()
                return True
            except OffboardError as exc_retry:
                self.get_logger().error(f"Offboard retry failed: {exc_retry}")
                return False

    async def _stop_offboard(self) -> None:
        drone = self._require_drone()
        await self._stop_streaming()
        try:
            await drone.offboard.stop()
        except Exception:
            pass

    async def _stop_streaming(self) -> None:
        self._streaming = False
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None

    async def _abort_offboard(self, reason: str) -> None:
        self.get_logger().error(f"Abort offboard: {reason}")
        await self._stop_offboard()

    async def _wait_for_position(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        check_xy: bool,
        timeout_s: float,
    ) -> bool:
        start = time.monotonic()
        xy_accept = self._config["xy_accept_m"]
        z_accept = self._config["z_accept_m"]

        while time.monotonic() - start < timeout_s:
            pos = self._last_position
            if pos is None:
                await asyncio.sleep(0.1)
                continue

            dx = target_x - pos.position.north_m
            dy = target_y - pos.position.east_m
            dz = target_z - pos.position.down_m

            xy_ok = True if not check_xy else (dx * dx + dy * dy) ** 0.5 <= xy_accept
            z_ok = abs(dz) <= z_accept
            if xy_ok and z_ok:
                return True
            await asyncio.sleep(0.1)
        return False

    async def _wait_for_landed(self, timeout_s: float) -> bool:
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            if self._last_landed_state == LandedState.ON_GROUND:
                return True
            await asyncio.sleep(0.2)
        return False


async def _async_main() -> None:
    rclpy.init()
    node = DeliveryNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        await node.run()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
