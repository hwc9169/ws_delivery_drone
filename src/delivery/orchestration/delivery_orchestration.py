import argparse
import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, List

import yaml

from ..agent.drone_agent import DroneState, DroneStatus
from ..common.schemas import AssignOrderCommand, GPSCoord, Order, OrderStatus
from ..common.mqtt_client import MqttClient



class DeliveryOrchestrator:
    def __init__(self, mqtt_host: str, mqtt_port: int, config: Dict[str, Any]) -> None:
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.config = config
        self.logger = self._setup_logger()
        self.stations: Dict[str, GPSCoord] = {
            "provider": self._gps_from_config(config["stations"]["provider"]),
            "consumer": self._gps_from_config(config["stations"]["consumer"]),
        }
        self.flight = config["flight"]
        self.delivery_timeout_s = float(config["orders"].get("delivery_timeout_s", 300))
        self.auto_interval_s = float(config["orders"].get("auto_interval_s", 0))

        self.drone_status: Dict[str, DroneStatus] = {}
        self.orders: Dict[str, Order] = {}
        self.order_queue: List[str] = []

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("delivery.orchestration")
        if logger.handlers:
            return logger
        logger.setLevel(logging.INFO)
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "delivery_orchestration.log"),
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

    def _gps_from_config(self, station: Dict[str, Any]) -> GPSCoord:
        return GPSCoord.model_validate(station)

    async def run(self) -> None:
        async with MqttClient(self.mqtt_host, self.mqtt_port, client_id="orchestration") as mqtt:
            self.mqtt = mqtt
            tasks = [
                asyncio.create_task(self._status_listener()),
                # asyncio.create_task(self._timeout_monitor()),
                asyncio.create_task(self._cli_loop()),
            ]
            if self.auto_interval_s > 0:
                tasks.append(asyncio.create_task(self._auto_order_loop()))
            await asyncio.gather(*tasks)

    async def _status_listener(self) -> None:
      # subscribe to drone status updates
      async for topic, payload in self.mqtt.subscribe_json("delivery/+/status"):
        if not isinstance(payload, dict):
          self.logger.warning("Invalid status payload type: %s", type(payload))
          continue
        try:
          drone_status = DroneStatus.model_validate(payload)
        except Exception as exc:
          self.logger.error("Status parse error: %s", exc)
          continue
        drone_id = drone_status.drone_id
        if not drone_id:
          continue
        prev_status = self.drone_status.get(drone_id)
        self.drone_status[drone_id] = drone_status

        # update order status based on drone status
        order_id = drone_status.order_id
        drone_state = drone_status.state
        if order_id and order_id in self.orders and drone_state is not None:
            order = self.orders[order_id]
            if drone_state in {
                DroneState.GO_TO_PROVIDER,
                DroneState.WAITING_PICKUP,
                DroneState.GO_TO_CONSUMER,
                DroneState.WAITING_DELIVER
              }:
                order.status = OrderStatus.IN_PROGRESS
            elif drone_state == DroneState.DONE:
                order.status = OrderStatus.COMPLETED
                order.assigned_drone = None
                self.logger.info("Order completed order_id=%s drone_id=%s", order_id, drone_id)
            elif drone_state == DroneState.ABORT:
                order.status = OrderStatus.FAILED
                self.logger.warning("Order failed order_id=%s drone_id=%s", order_id, drone_id)
                await self._maybe_reassign(order, exclude_drone=drone_id)

        # try to assign queued orders
        if not self.order_queue:
          continue
        available = self._available_drones()
        if not available:
          continue
        order_id = self.order_queue.pop(0)
        order = self.orders[order_id]
        await self._assign_order(order, available)

    def _available_drones(self) -> List[str]:
        available = []
        for drone_id, status in self.drone_status.items():
            if status.state == DroneState.READY and status.order_id in (None, "", "null"):
                available.append(drone_id)
        return available

    async def _assign_order(self, order: Order, available_drones: List[str]) -> None:
        chosen = self._choose_nearest(available_drones)
        if not chosen:
            self.order_queue.insert(0, order.order_id)
            return
        status = self.drone_status.get(chosen)
        if not status:
            self.order_queue.insert(0, order.order_id)
            return
        drone_gps = status.gps
        if not drone_gps:
            self.logger.warning("Missing GPS for drone_id=%s", chosen)
            self.order_queue.insert(0, order.order_id)
            return

        provider_gps = self.stations["provider"]
        consumer_gps = self.stations["consumer"]
        self.logger.info(
            "Assign order order_id=%s\ndrone_id=%s\ndrone_gps=(%.7f,%.7f,%.7f)\nprovider_gps=(%.7f,%.7f,%.7f)\nconsumer_gps=(%.7f,%.7f,%.7f)",
            order.order_id,
            chosen,
            drone_gps.lat,
            drone_gps.lon,
            drone_gps.alt,
            provider_gps.lat,
            provider_gps.lon,
            provider_gps.alt,
            consumer_gps.lat,
            consumer_gps.lon,
            consumer_gps.alt,
        )
        assign_cmd = AssignOrderCommand(
            order_id=order.order_id,
            provider=provider_gps,
            consumer=consumer_gps,
            safe_alt_m=float(self.flight["safe_alt_m"]),
            approach_alt_m=float(self.flight["approach_alt_m"]),
            yaw_deg=0.0,
        )
        await self.mqtt.publish_json(
            f"delivery/{chosen}/cmd",
            assign_cmd.model_dump(mode="json"),
        )
        order.status = OrderStatus.ASSIGNED
        order.assigned_drone = chosen
        order.assigned_ts = time.time()
        self.logger.info("Assigned order_id=%s to drone_id=%s", order.order_id, chosen)

    def _choose_nearest(self, drone_ids: List[str]) -> str:
        return drone_ids[0]

    async def _maybe_reassign(self, order: Order, exclude_drone: str) -> None:
        available = [d for d in self._available_drones() if d != exclude_drone]
        if not available:
            self.logger.info("No available drones to reassign order_id=%s", order.order_id)
            return
        order.status = OrderStatus.CREATED
        self.logger.info("Reassigning order_id=%s exclude_drone=%s", order.order_id, exclude_drone)
        await self._assign_order(order, available)

    async def _timeout_monitor(self) -> None:
        while True:
            now = time.time()
            for order in list(self.orders.values()):
                if order.status in {OrderStatus.ASSIGNED, OrderStatus.IN_PROGRESS} and order.assigned_ts:
                    if now - order.assigned_ts > self.delivery_timeout_s:
                        order.status = OrderStatus.FAILED
                        self.logger.warning("Order timeout order_id=%s", order.order_id)
                        await self._maybe_reassign(order, exclude_drone=order.assigned_drone or "")
            await asyncio.sleep(1.0)

    async def _cli_loop(self) -> None:
        while True:
            cmd = await asyncio.to_thread(input, "orchestration> ")
            cmd = cmd.strip().lower()
            if cmd in {"order", "o", ""}:
                await self._create_order()
            elif cmd in {"status", "s"}:
                self._print_status()
            elif cmd in {"drones", "d"}:
                self._print_drones()
            elif cmd in {"quit", "exit"}:
                raise SystemExit(0)

    async def _auto_order_loop(self) -> None:
        while True:
            await asyncio.sleep(self.auto_interval_s)
            await self._create_order()

    async def _create_order(self) -> None:
        order_id = str(uuid.uuid4())
        order = Order(
            order_id=order_id,
            provider=self.stations["provider"],
            consumer=self.stations["consumer"],
            status=OrderStatus.CREATED,
            assigned_drone=None,
            created_ts=time.time(),
            assigned_ts=None,
        )
        self.orders[order_id] = order
        self.logger.info("Order created order_id=%s", order_id)
        available = self._available_drones()
        if not available:
            self.order_queue.append(order_id)
            self.logger.info("Order queued order_id=%s", order_id)
            return
        await self._assign_order(order, available)

    def _print_status(self) -> None:
        print("Orders:")
        for order in self.orders.values():
            print(f"  {order.order_id} {order.status.value} drone={order.assigned_drone}")
        self._print_drones()

    def _print_drones(self) -> None:
        print("Drones:")
        for drone_id, status in self.drone_status.items():
          pos = status.pos_ned
          gps = status.gps or GPSCoord(lat=0.0, lon=0.0, alt=0.0)
          print(
              f"  {drone_id} state={status.state} order={status.order_id} "
              f"pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})"
              f"gps=({gps.lat:.6f},{gps.lon:.6f},{gps.alt:.6f})"
          )


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delivery orchestration for PX4 SITL")
    parser.add_argument("--mqtt-host", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--config", default="src/delivery/orchestration/config.yaml")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    orchestrator = DeliveryOrchestrator(args.mqtt_host, args.mqtt_port, config)
    print("Starting delivery orchestration")
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
