from __future__ import annotations
from typing import Optional, Tuple
import math


def l2(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def distance_to(target: Tuple[float, float, float], pos: Tuple[float, float, float]) -> float:
    return l2(target, pos)

