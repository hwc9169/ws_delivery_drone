import math
from typing import Iterable


def distance_xy(a_ned: Iterable[float], b_ned: Iterable[float]) -> float:
    a_n, a_e = a_ned[0], a_ned[1]
    b_n, b_e = b_ned[0], b_ned[1]
    return math.hypot(a_n - b_n, a_e - b_e)


def distance_xyz(a_ned: Iterable[float], b_ned: Iterable[float]) -> float:
    a_n, a_e, a_d = a_ned[0], a_ned[1], a_ned[2]
    b_n, b_e, b_d = b_ned[0], b_ned[1], b_ned[2]
    return math.sqrt((a_n - b_n) ** 2 + (a_e - b_e) ** 2 + (a_d - b_d) ** 2)
