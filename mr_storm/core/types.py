"""
Core data structures for simulator- and hardware-agnostic MR-STORM execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import json
import numpy as np


@dataclass
class RobotJointState:
    """Simulator- and hardware-agnostic joint state representation."""

    positions: np.ndarray  # shape: (n_dofs,) in radians
    velocities: np.ndarray  # shape: (n_dofs,) in rad/s
    accelerations: Optional[np.ndarray] = None  # shape: (n_dofs,) in rad/s^2
    joint_names: Optional[List[str]] = None
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        self.positions = np.asarray(self.positions, dtype=np.float32).reshape(-1)
        self.velocities = np.asarray(self.velocities, dtype=np.float32).reshape(-1)
        if self.accelerations is None:
            self.accelerations = np.zeros_like(self.positions)
        else:
            self.accelerations = np.asarray(self.accelerations, dtype=np.float32).reshape(-1)


@dataclass
class JointCommand:
    """Command output produced by an MR-STORM agent step."""

    positions: np.ndarray  # shape: (n_dofs,) target joint positions (rad)
    velocities: np.ndarray  # shape: (n_dofs,) target joint velocities (rad/s)
    accelerations: np.ndarray  # shape: (n_dofs,) commanded accelerations (rad/s^2)
    joint_names: List[str]
    dt: float = 0.033  # control step duration in seconds
    ee_position_error: float = 0.0
    ee_rotation_error: float = 0.0
    solve_time_ms: float = 0.0


@dataclass
class BroadcastPlanPacket:
    """
    Compact plan broadcast packet shared between decentralized MR-STORM agents.

    Contains the predicted task-space collision spheres over the MPC horizon H
    expressed in the common world frame W, along with the current goal error
    metrics used for dynamic priority resolution (Winner-Takes-All / tau).
    """

    robot_id: int
    timestamp: float
    horizon: int
    n_spheres: int
    sphere_positions: np.ndarray  # shape: (H, S, 3) in world frame (meters)
    sphere_radii: np.ndarray  # shape: (S,) in meters
    pos_error: float  # Euclidean distance to EE goal (meters)
    rot_error: float  # Geodesic rotation error to EE goal (radians or degrees)
    base_pose: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def to_bytes(self) -> bytes:
        """Serialize packet to a compact binary payload for ROS 2 / ZMQ / UDP transport."""
        header = {
            "robot_id": int(self.robot_id),
            "timestamp": float(self.timestamp),
            "horizon": int(self.horizon),
            "n_spheres": int(self.n_spheres),
            "pos_error": float(self.pos_error),
            "rot_error": float(self.rot_error),
            "base_pose": [float(x) for x in self.base_pose],
        }
        header_bytes = json.dumps(header).encode("utf-8")
        header_len = len(header_bytes).to_bytes(4, byteorder="little")
        pos_bytes = np.asarray(self.sphere_positions, dtype=np.float32).tobytes()
        rad_bytes = np.asarray(self.sphere_radii, dtype=np.float32).tobytes()
        return header_len + header_bytes + pos_bytes + rad_bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> "BroadcastPlanPacket":
        """Deserialize a binary payload back into a BroadcastPlanPacket."""
        header_len = int.from_bytes(data[:4], byteorder="little")
        header = json.loads(data[4 : 4 + header_len].decode("utf-8"))
        H = header["horizon"]
        S = header["n_spheres"]
        offset = 4 + header_len
        pos_size = H * S * 3 * 4
        pos_buf = data[offset : offset + pos_size]
        rad_buf = data[offset + pos_size : offset + pos_size + S * 4]
        sphere_positions = np.frombuffer(pos_buf, dtype=np.float32).reshape(H, S, 3).copy()
        sphere_radii = np.frombuffer(rad_buf, dtype=np.float32).reshape(S).copy()
        return cls(
            robot_id=header["robot_id"],
            timestamp=header["timestamp"],
            horizon=H,
            n_spheres=S,
            sphere_positions=sphere_positions,
            sphere_radii=sphere_radii,
            pos_error=header["pos_error"],
            rot_error=header["rot_error"],
            base_pose=header.get("base_pose", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        )


@dataclass
class ArmConfig:
    """Configuration for a single arm in an MR-STORM setup."""

    robot_id: int
    robot_yml: str = "ur5e.yml"
    base_pose: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    )  # [x, y, z, qw, qx, qy, qz] in world frame
    particle_yml: str = "projects_root/experiments/benchmarks/cfgs/particle/O.yml"
    retract_cfg: Optional[List[float]] = None
    namespace: str = ""
    override_col_With: Optional[List[int]] = None

    def __post_init__(self):
        if not self.namespace:
            self.namespace = f"arm_{self.robot_id}"
