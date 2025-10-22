from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple


@dataclass
class StepLog:
    t: float
    agent_pos: Tuple[float, float, float]
    agent_coord: Tuple[int, int, int]
    yaw: float
    command_len: int
    collisions_count: int
    distance_to_target: float | None
    frame_time_ms: float
    policy_log: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeSummary:
    success: bool
    steps: int
    collisions_total: int
    avg_command_len: float
    path_len_est: float | None
    coverage: float | None
    time_to_goal: float | None
    reasons: str | None
    policy_summary: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

