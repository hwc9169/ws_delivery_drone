import argparse
import asyncio
import logging
import os
from enum import Enum
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw
from mavsdk.telemetry import LandedState
from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError

from ..common.mqtt_client import MqttClient
from ..common.schemas import (
    AbortCommand,
    AssignOrderCommand,
    Command,
    ConfirmDeliverCommand,
    ConfirmPickupCommand,
    GPSCoord,
    LocalNEDCoord,
    parse_command,
)
from ..common.math_utils import distance_xy


class DroneState(str, Enum):
    READY = "READY"
    GO_TO_PROVIDER = "GO_TO_PROVIDER"
    WAITING_PICKUP = "WAITING_PICKUP"
    GO_TO_CONSUMER = "GO_TO_CONSUMER"
    WAITING_DELIVER = "WAITING_DELIVER"
    DONE = "DONE"
    ABORT = "ABORT"


class DroneStatus(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    drone_id: str
    order_id: str | None = None
    state: DroneState
    pos_ned: list[float]
    gps: GPSCoord | None = None
    armed: bool
    last_error: str | None = None
    ts: float


@dataclass
class FlightConfig:
    safe_alt_m: float
    approach_alt_m: float
    setpoint_rate_hz: float
    xy_accept_m: float
    z_accept_m: float
    timeouts_s: Dict[str, float]


@dataclass
class WaitingConfig:
    pickup: float
    deliver: float


class DroneAgent:
    def __init__(
        self,
        drone_id: str,
        system_address: str,
        mqtt_host: str,
        mqtt_port: int,
        config: Dict[str, Any],
    ) -> None:
        self.drone_id = drone_id
        self.system_address = system_address
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        flight = config["flight"]
        waiting = config["waiting_s"]
        self.flight = FlightConfig(
            safe_alt_m=float(flight["safe_alt_m"]),
            approach_alt_m=float(flight["approach_alt_m"]),
            setpoint_rate_hz=float(flight["setpoint_rate_hz"]),
            xy_accept_m=float(flight["xy_accept_m"]),
            z_accept_m=float(flight["z_accept_m"]),
            timeouts_s={k: float(v) for k, v in flight["timeouts_s"].items()},
        )
        self.waiting = WaitingConfig(
            pickup=float(waiting["pickup"]),
            deliver=float(waiting["deliver"]),
        )
        self.abort_return_ready = bool(config.get("abort", {}).get("return_ready", True))
        self.abort_action_default = str(config.get("abort", {}).get("action", "LAND"))
        self.status_rate_hz = float(config.get("status", {}).get("rate_hz", 3.0))
        self.logger = self._setup_logger()

        self.state = DroneState.READY
        self.order_id: Optional[str] = None
        self.last_error: Optional[str] = None
        self.pos_ned = [0.0, 0.0, 0.0]
        self.gps: Optional[GPSCoord] = None
        self.armed = False
        self.assignment: Optional[AssignOrderCommand] = None

        self._assign_event = asyncio.Event()
        self._pickup_event = asyncio.Event()
        self._deliver_event = asyncio.Event()
        self._abort_event = asyncio.Event()
        self._abort_action = self.abort_action_default
        self._status_event = asyncio.Event()

        self.drone: Optional[System] = None
        self._offboard_active = False

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"delivery.drone_agent.{self.drone_id}")
        if logger.handlers:
            return logger
        logger.setLevel(logging.INFO)
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"drone_agent_{self.drone_id}.log"),
            encoding="utf-8",
        )
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        return logger

    async def run(self) -> None:
        await self._connect_drone()
        async with MqttClient(self.mqtt_host, self.mqtt_port, client_id=self.drone_id) as mqtt:
            self.mqtt = mqtt
            tasks = [
                asyncio.create_task(self._telemetry_loop()),
                asyncio.create_task(self._gps_loop()),
                asyncio.create_task(self._armed_loop()),
                asyncio.create_task(self._mqtt_command_loop()),
                asyncio.create_task(self._status_loop()),
                asyncio.create_task(self._state_machine()),
            ]
            await asyncio.gather(*tasks)

    async def _connect_drone(self) -> None:
        self.drone = System()
        await self.drone.connect(system_address=self.system_address)
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                self.logger.info("Drone connected")
                break

    def _set_state(self, new_state: DroneState) -> None:
        if self.state != new_state:
            self.state = new_state
            self._status_event.set()

    async def _telemetry_loop(self) -> None:
      # set pos_ned fom drone telemetry
      assert self.drone
      async for pos in self.drone.telemetry.position_velocity_ned():
        self.pos_ned = [pos.position.north_m, pos.position.east_m, pos.position.down_m]

    async def _gps_loop(self) -> None:
      assert self.drone
      async for pos in self.drone.telemetry.position():
        self.gps = GPSCoord(lat=pos.latitude_deg, lon=pos.longitude_deg, alt=pos.absolute_altitude_m)

    async def _armed_loop(self) -> None:
      # set armed from drone telemetry
      assert self.drone
      async for armed in self.drone.telemetry.armed():
        self.armed = bool(armed)

    async def _mqtt_command_loop(self) -> None:
      # subscribe to MQTT command topic and handle commands
      topic = f"delivery/{self.drone_id}/cmd"
      async for _, payload in self.mqtt.subscribe_json(topic):
        if not isinstance(payload, dict):
          self.logger.warning("Invalid command payload type: %s", type(payload))
          continue
        try:
          cmd: Command = parse_command(payload)
        except ValidationError as exc:
          self.last_error = str(exc)
          self.logger.error("Command parse error: %s", exc)
          self._status_event.set()
          continue

        if isinstance(cmd, AssignOrderCommand):
          if self.state != DroneState.READY or self.assignment is not None:
            self.last_error = "Received ASSIGN_ORDER while busy"
            self.logger.warning("Assign received while busy order_id=%s", cmd.order_id)
            self._status_event.set()
            continue
          self.logger.info("Assigned order_id=%s", cmd.order_id)
          self.assignment = cmd
          self.order_id = cmd.order_id
          self._assign_event.set()
        elif isinstance(cmd, ConfirmPickupCommand):
          if self.order_id == cmd.order_id:
            self._pickup_event.set()
        elif isinstance(cmd, ConfirmDeliverCommand):
          if self.order_id == cmd.order_id:
            self._deliver_event.set()
        elif isinstance(cmd, AbortCommand):
          if self.order_id == cmd.order_id:
            self.logger.warning("Abort order_id=%s action=%s", cmd.order_id, cmd.action)
            self._abort_action = cmd.action
            self._abort_event.set()

    async def _status_loop(self) -> None:
      # publish status at regular intervals 
      interval = 1.0 / max(self.status_rate_hz, 0.5)
      while True:
        status = DroneStatus(
          drone_id=self.drone_id,
          order_id=self.order_id,
          state=self.state,
          pos_ned=self.pos_ned,
          gps=self.gps,
          armed=self.armed,
          last_error=self.last_error,
          ts=time.time(),
        )
        await self.mqtt.publish_json(
            f"delivery/{self.drone_id}/status",
            status.model_dump(mode="json"),
        )
        try:
          await asyncio.wait_for(self._status_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
        self._status_event.clear()

    async def _state_machine(self) -> None:
        while True:
          await self._assign_event.wait()
          self._assign_event.clear()
          if not self.assignment:
              continue
          try:
            self.last_error = None
            self._set_state(DroneState.GO_TO_PROVIDER)
            ref_gps = self.gps
            if not ref_gps:
                raise RuntimeError("GPS unavailable for provider conversion")
            offset = self.assignment.provider.to_local_ned(ref_gps)
            pos_ned = self.pos_ned
            provider_target = LocalNEDCoord(
                north=pos_ned[0] + offset.north,
                east=pos_ned[1] + offset.east,
                down=pos_ned[2] + offset.down,
            )
            # logging start point gps
            self.logger.info(
                "Takeoff Current gps=(%.7f,%.7f,%.2f)",
                ref_gps.lat,
                ref_gps.lon,
                ref_gps.alt,
            )
            await self._fly_to(provider_target, self.assignment.safe_alt_m, self.assignment.approach_alt_m, self.assignment.yaw_deg)
            self._set_state(DroneState.WAITING_PICKUP)
            await self._wait_for_event(self._pickup_event, self.waiting.pickup)
            self._set_state(DroneState.GO_TO_CONSUMER)
            ref_gps = self.gps
            if not ref_gps:
                raise RuntimeError("GPS unavailable for consumer conversion")
            offset = self.assignment.consumer.to_local_ned(ref_gps)
            pos_ned = self.pos_ned
            # logging start point gps
            self.logger.info(
                "Takeoff Current gps=(%.7f,%.7f,%.2f)",
                ref_gps.lat,
                ref_gps.lon,
                ref_gps.alt,
            )
            consumer_target = LocalNEDCoord(
                north=pos_ned[0] + offset.north,
                east=pos_ned[1] + offset.east,
                down=pos_ned[2] + offset.down,
            )
            await self._fly_to(consumer_target, self.assignment.safe_alt_m, self.assignment.approach_alt_m, self.assignment.yaw_deg)
            self._set_state(DroneState.WAITING_DELIVER)
            await self._wait_for_event(self._deliver_event, self.waiting.deliver)
            self._set_state(DroneState.DONE)
            if self.drone:
              await self.drone.action.disarm()
          except Exception as exc:
            self.last_error = str(exc)
            self.logger.error("State machine error: %s", exc)
            await self._handle_abort(self.abort_action_default)
          finally:
              if self.state == DroneState.DONE:
                  self._reset_to_ready()
              elif self.state == DroneState.ABORT and self.abort_return_ready:
                  self._reset_to_ready()

    def _reset_to_ready(self) -> None:
        self.assignment = None
        self.order_id = None
        self._pickup_event.clear()
        self._deliver_event.clear()
        self._abort_event.clear()
        self._set_state(DroneState.READY)

    async def _wait_for_event(self, event: asyncio.Event, timeout_s: float) -> None:
        if self._abort_event.is_set():
            await self._handle_abort(self._abort_action)
            return
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return
        finally:
            event.clear()

    async def _fly_to(
        self,
        target: LocalNEDCoord,
        safe_alt_m: float,
        approach_alt_m: float,
        yaw_deg: float,
    ) -> None:
        # arm and start offboard(yaw_deg)
        assert self.drone
        await self.drone.action.arm()
        self.armed = True
        self.logger.info("Takeoff start")
        await self.drone.offboard.set_position_ned(PositionNedYaw(
            self.pos_ned[0], self.pos_ned[1], self.pos_ned[2], yaw_deg
        ))
        try:
            await self.drone.offboard.start()
            self._offboard_active = True
        except OffboardError as exc:
            self.logger.error("Offboard start failed: %s", exc)
            raise RuntimeError(f"Offboard start failed: {exc._result.result}") from exc

        try:
            self.logger.info("Fly phase takeoff to alt=%.2f", safe_alt_m)
            await self._goto_position(
                [self.pos_ned[0], self.pos_ned[1], -safe_alt_m],
                yaw_deg,
                self.flight.xy_accept_m,
                self.flight.z_accept_m,
                self.flight.timeouts_s["takeoff"],
            )
            await self._goto_position(
                [target.north, target.east, -safe_alt_m],
                yaw_deg,
                self.flight.xy_accept_m,
                self.flight.z_accept_m,
                self.flight.timeouts_s["cruise"],
            )
            await self._goto_position(
                [target.north, target.east, -approach_alt_m],
                yaw_deg,
                self.flight.xy_accept_m,
                self.flight.z_accept_m,
                self.flight.timeouts_s["approach"],
            )
        finally:
            await self._stop_offboard()
            self.logger.info("Fly complete target=(%.2f,%.2f,%.2f)", target.north, target.east, target.down)
        await self._land_and_wait()

    async def _stop_offboard(self) -> None:
        if self.drone and self._offboard_active:
            try:
                await self.drone.offboard.stop()
            except OffboardError:
                pass
            self._offboard_active = False

    async def _goto_position(
        self,
        target_ned: list[float],
        yaw_deg: float,
        xy_accept_m: float,
        z_accept_m: float,
        timeout_s: float,
    ) -> None:
        start = time.monotonic()
        interval = 1.0 / max(self.flight.setpoint_rate_hz, 2.0)
        while True:
            if self._abort_event.is_set():
                await self._handle_abort(self._abort_action)
                return
            if not self.drone:
                return
            await self.drone.offboard.set_position_ned(
                PositionNedYaw(target_ned[0], target_ned[1], target_ned[2], yaw_deg)
            )
            if distance_xy(self.pos_ned, target_ned) <= xy_accept_m and abs(self.pos_ned[2] - target_ned[2]) <= z_accept_m:
                return
            if time.monotonic() - start > timeout_s:
                self.last_error = "Timeout reaching target"
                await self._handle_abort(self.abort_action_default)
                return
            await asyncio.sleep(interval)

    async def _land_and_wait(self) -> None:
        assert self.drone
        await self.drone.action.land()
        try:
            await asyncio.wait_for(self._wait_landed_state(), timeout=self.flight.timeouts_s["land"])
            assert self.gps
            self.logger.info(
                "Land complete gps=(%.7f,%.7f,%.2f)",
                self.gps.lat,
                self.gps.lon,
                self.gps.alt,
            )
        except asyncio.TimeoutError:
            self.last_error = "Timeout during land"
            await self._handle_abort(self.abort_action_default)

    async def _wait_landed_state(self) -> None:
        assert self.drone
        async for state in self.drone.telemetry.landed_state():
            if state == LandedState.ON_GROUND:
                return

    async def _handle_abort(self, action: str) -> None:
        self._set_state(DroneState.ABORT)
        await self._stop_offboard()
        if not self.drone:
            return
        if action.upper() == "HOLD":
            await self.drone.action.hold()
        else:
            await self.drone.action.land()
        self._abort_event.clear()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PX4 MAVSDK delivery drone agent")
    parser.add_argument("--drone-id", required=True)
    parser.add_argument("--system-address", required=True)
    parser.add_argument("--mqtt-host", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--config", default="src/delivery/agent/config.yaml")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    agent = DroneAgent(
        drone_id=args.drone_id,
        system_address=args.system_address,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        config=config,
    )
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
