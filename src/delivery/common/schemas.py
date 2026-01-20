from __future__ import annotations

import math
from enum import Enum
from typing import Any, ClassVar, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, TypeAdapter


#### Coordinate Models ####
class GPSCoord(BaseModel):
    model_config = ConfigDict(frozen=True)

    _WGS84_A: ClassVar[float] = 6378137.0
    _WGS84_F: ClassVar[float] = 1.0 / 298.257223563
    _WGS84_E2: ClassVar[float] = _WGS84_F * (2.0 - _WGS84_F)

    lat: float
    lon: float
    alt: float = 0.0

    def to_local_ned(
        self,
        reference: "GPSCoord",
        ref_ecef: Optional[tuple[float, float, float]] = None,
        ref_trig: Optional[tuple[float, float, float, float]] = None,
    ) -> LocalNEDCoord:
        if ref_ecef is None:
            ref_ecef = reference._to_ecef()
        if ref_trig is None:
            ref_lat = math.radians(reference.lat)
            ref_lon = math.radians(reference.lon)
            ref_trig = (
                math.sin(ref_lat),
                math.cos(ref_lat),
                math.sin(ref_lon),
                math.cos(ref_lon),
            )
        x1, y1, z1 = self._to_ecef()
        dx = x1 - ref_ecef[0]
        dy = y1 - ref_ecef[1]
        dz = z1 - ref_ecef[2]
        sin_lat, cos_lat, sin_lon, cos_lon = ref_trig
        east = -sin_lon * dx + cos_lon * dy
        north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
        return LocalNEDCoord(north=north, east=east, down=-up)

    def _to_ecef(self) -> tuple[float, float, float]:
        lat = math.radians(self.lat)
        lon = math.radians(self.lon)
        sin_lat = math.sin(lat)
        cos_lat = math.cos(lat)
        sin_lon = math.sin(lon)
        cos_lon = math.cos(lon)
        n = self._WGS84_A / math.sqrt(1.0 - self._WGS84_E2 * sin_lat * sin_lat)
        x = (n + self.alt) * cos_lat * cos_lon
        y = (n + self.alt) * cos_lat * sin_lon
        z = (n * (1.0 - self._WGS84_E2) + self.alt) * sin_lat
        return x, y, z


class LocalNEDCoord(BaseModel):
    north: float
    east: float
    down: float


#### Order Models ####
class Order(BaseModel):
    order_id: str
    provider: GPSCoord
    consumer: GPSCoord
    status: OrderStatus
    assigned_drone: Optional[str]
    created_ts: float
    assigned_ts: Optional[float]


class OrderStatus(str, Enum):
    CREATED = "CREATED"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


#### Command Models ####
class AssignOrderCommand(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str = "ASSIGN_ORDER"
    order_id: str
    provider: GPSCoord  # Local NED coordinates (north, east, down)
    consumer: GPSCoord  # Local NED coordinates (north, east, down)
    safe_alt_m: float
    approach_alt_m: float
    yaw_deg: float


class ConfirmPickupCommand(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str = "CONFIRM_PICKUP"
    order_id: str


class ConfirmDeliverCommand(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str = "CONFIRM_DELIVER"
    order_id: str


class AbortCommand(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str = "ABORT"
    order_id: str
    action: str


Command = Union[AssignOrderCommand, ConfirmPickupCommand, ConfirmDeliverCommand, AbortCommand]

_COMMAND_ADAPTER = TypeAdapter(Command)


def parse_command(payload: Dict[str, Any]) -> Command:
    return _COMMAND_ADAPTER.validate_python(payload)
