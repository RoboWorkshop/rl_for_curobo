"""
Inter-robot communication layer for MR-STORM (`plans_board` and `mean_goal_err`).

Supports:
1. `LocalPlanBoardComm`: In-memory shared `plans_board` & `mean_goal_err` (for
   `synchronous-decentralized` pseudo-distributed mode or local multi-thread async).
2. `ROSPlanBoardComm`: Distributed ROS 2 (`rclpy`) / ROS 1 (`rospy`) communication
   for `asynchronous-decentralized` mode where each robot publishes its estimated
   horizon sphere plan (`plan['task_space']['spheres']`) and goal error
   (`mean_goal_err`) over ROS topics and populates `plans_board` and `mean_goal_err`
   in the exact format expected by `CuAgent.update_col_pred(plans_board, mean_goal_err)`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import struct
from threading import Lock, Thread
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from curobo.types.base import TensorDeviceType


def pack_sphere_plan_bytes(
    robot_idx: int,
    p_tensor: torch.Tensor,
    r_tensor: torch.Tensor,
    p_err: float = -1.0,
    q_err: float = -1.0,
) -> bytes:
    """
    Serialize `plan['task_space']['spheres']['p']` ([H, S, 3]) and `['r']` ([H, S])
    plus optional `(p_err, q_err)` into a compact byte buffer for ROS transport.
    """
    p_np = np.ascontiguousarray(p_tensor.detach().cpu().numpy(), dtype=np.float32)
    r_np = np.ascontiguousarray(r_tensor.detach().cpu().numpy(), dtype=np.float32)
    if r_np.ndim == 1:
        r_np = np.broadcast_to(r_np[np.newaxis, :], (p_np.shape[0], r_np.shape[0])).copy()
    H, S, _ = p_np.shape
    header = struct.pack("<iiidff", int(robot_idx), int(H), int(S), time.time(), float(p_err), float(q_err))
    return header + p_np.tobytes() + r_np.tobytes()


def unpack_sphere_plan_bytes(
    raw: bytes, tensor_args: TensorDeviceType
) -> Tuple[int, dict, Optional[Tuple[float, float]], float]:
    """
    Deserialize a compact byte buffer into `(robot_idx, plan_dict, goal_err_tuple, timestamp)`
    where `plan_dict` has the exact structure expected by `MpcPlanner.update_col_pred`:
    `{'task_space': {'spheres': {'p': Tensor[H, S, 3], 'r': Tensor[H, S]}}}`.
    """
    header_size = struct.calcsize("<iiidff")
    robot_idx, H, S, ts, p_err, q_err = struct.unpack("<iiidff", raw[:header_size])
    offset = header_size
    p_bytes = H * S * 3 * 4
    r_bytes = H * S * 4
    p_np = np.frombuffer(raw, dtype=np.float32, count=H * S * 3, offset=offset).reshape(H, S, 3).copy()
    offset += p_bytes
    r_np = np.frombuffer(raw, dtype=np.float32, count=H * S, offset=offset).reshape(H, S).copy()

    p_t = tensor_args.to_device(torch.from_numpy(p_np))
    r_t = tensor_args.to_device(torch.from_numpy(r_np))
    plan_dict = {
        "task_space": {
            "spheres": {
                "p": p_t,
                "r": r_t,
            }
        }
    }
    goal_err = (float(p_err), float(q_err)) if p_err >= 0.0 else None
    return robot_idx, plan_dict, goal_err, ts


class PlanCommBackend(ABC):
    """
    Abstract communication layer providing `plans_board` and `mean_goal_err`
    between decentralized `CuAgent` instances.
    """

    def __init__(self, n_agents: int, tensor_args: Optional[TensorDeviceType] = None):
        self.n_agents = n_agents
        self.tensor_args = tensor_args or TensorDeviceType()
        self.plans_board: List[Optional[dict]] = [None for _ in range(n_agents)]
        self.mean_goal_err: List[Optional[Tuple[float, float]]] = [
            None for _ in range(n_agents)
        ]
        self.lock = Lock()

    @abstractmethod
    def publish_plan(self, agent_idx: int, plan: dict) -> None:
        """Publish `agent_idx`'s horizon plan (`get_estimated_plan` output)."""

    @abstractmethod
    def publish_goal_error(self, agent_idx: int, goal_err: Tuple[float, float]) -> None:
        """Publish `agent_idx`'s `(p_err, q_err)` for `pose_wta` conflict resolution."""

    @abstractmethod
    def get_boards(
        self,
    ) -> Tuple[List[Optional[dict]], List[Optional[Tuple[float, float]]]]:
        """Return `(plans_board, mean_goal_err)` in exact `benchmark_sim.py` format."""

    def close(self) -> None:
        """Clean up any background threads or ROS nodes."""


class LocalPlanBoardComm(PlanCommBackend):
    """
    In-memory `plans_board` and `mean_goal_err` communication layer.

    Used by:
    - `synchronous-decentralized` mode (exact sequential pseudo-distributed loop from `benchmark_sim.py`)
    - `asynchronous-decentralized` mode when running multiple robot threads inside a single process
      without ROS.
    """

    def publish_plan(self, agent_idx: int, plan: dict) -> None:
        with self.lock:
            self.plans_board[agent_idx] = plan

    def publish_goal_error(self, agent_idx: int, goal_err: Tuple[float, float]) -> None:
        with self.lock:
            self.mean_goal_err[agent_idx] = goal_err

    def get_boards(
        self,
    ) -> Tuple[List[Optional[dict]], List[Optional[Tuple[float, float]]]]:
        return self.plans_board, self.mean_goal_err


class ROSPlanBoardComm(PlanCommBackend):
    """
    ROS communication backend for `asynchronous-decentralized` execution.

    Automatically detects and supports **ROS 2 (`rclpy`)** or **ROS 1 (`rospy`)**.
    Each robot publishes its estimated plan (`plan['task_space']['spheres']`) and
    `mean_goal_err` over ROS topics:
      - `/mr_storm/robot_<i>/plan` (`std_msgs/ByteMultiArray` or `UInt8MultiArray`)
      - `/mr_storm/robot_<i>/goal_error` (`std_msgs/Float32MultiArray`)
    and subscribes to all peer robots `j != i`, reconstructing `self.plans_board[j]`
    and `self.mean_goal_err[j]` on GPU so `a.update_col_pred(plans_board, mean_goal_err)`
    runs with 0 changes to the original MR-STORM algorithm!
    """

    def __init__(
        self,
        n_agents: int,
        tensor_args: Optional[TensorDeviceType] = None,
        active_agent_indices: Optional[List[int]] = None,
        topic_prefix: str = "/mr_storm",
        ros_version: str = "auto",
    ):
        super().__init__(n_agents=n_agents, tensor_args=tensor_args)
        self.topic_prefix = topic_prefix.rstrip("/")
        self.active_agent_indices = (
            active_agent_indices if active_agent_indices is not None else list(range(n_agents))
        )
        self._stop = False
        self._spin_thread: Optional[Thread] = None
        self._ros_mode = self._init_ros(ros_version)

    def _init_ros(self, ros_version: str) -> str:
        if ros_version in ("auto", "ros2"):
            try:
                import rclpy
                from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
                from std_msgs.msg import ByteMultiArray, Float32MultiArray

                self._rclpy = rclpy
                self._ByteMultiArray = ByteMultiArray
                self._Float32MultiArray = Float32MultiArray

                if not rclpy.ok():
                    rclpy.init()

                node_suffix = (
                    f"robot_{self.active_agent_indices[0]}"
                    if len(self.active_agent_indices) == 1
                    else f"fleet_{int(time.time() * 1000) % 10000}"
                )
                self.node = rclpy.create_node(f"mr_storm_comm_{node_suffix}")

                qos = QoSProfile(
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                )

                self._plan_pubs = {}
                self._err_pubs = {}
                for i in self.active_agent_indices:
                    self._plan_pubs[i] = self.node.create_publisher(
                        ByteMultiArray, f"{self.topic_prefix}/robot_{i}/plan", qos
                    )
                    self._err_pubs[i] = self.node.create_publisher(
                        Float32MultiArray, f"{self.topic_prefix}/robot_{i}/goal_error", qos
                    )

                self._plan_subs = []
                self._err_subs = []
                for j in range(self.n_agents):
                    self._plan_subs.append(
                        self.node.create_subscription(
                            ByteMultiArray,
                            f"{self.topic_prefix}/robot_{j}/plan",
                            lambda msg, r_idx=j: self._on_ros2_plan(msg, r_idx),
                            qos,
                        )
                    )
                    self._err_subs.append(
                        self.node.create_subscription(
                            Float32MultiArray,
                            f"{self.topic_prefix}/robot_{j}/goal_error",
                            lambda msg, r_idx=j: self._on_ros2_err(msg, r_idx),
                            qos,
                        )
                    )

                self._spin_thread = Thread(target=self._spin_ros2, daemon=True)
                self._spin_thread.start()
                return "ros2"
            except ImportError:
                if ros_version == "ros2":
                    raise

        if ros_version in ("auto", "ros1"):
            try:
                import rospy
                from std_msgs.msg import Float32MultiArray, UInt8MultiArray

                self._rospy = rospy
                self._UInt8MultiArray = UInt8MultiArray
                self._Float32MultiArray = Float32MultiArray

                if not rospy.core.is_initialized():
                    rospy.init_node("mr_storm_comm_node", anonymous=True, disable_signals=True)

                self._plan_pubs = {}
                self._err_pubs = {}
                for i in self.active_agent_indices:
                    self._plan_pubs[i] = rospy.Publisher(
                        f"{self.topic_prefix}/robot_{i}/plan",
                        UInt8MultiArray,
                        queue_size=1,
                    )
                    self._err_pubs[i] = rospy.Publisher(
                        f"{self.topic_prefix}/robot_{i}/goal_error",
                        Float32MultiArray,
                        queue_size=1,
                    )

                self._plan_subs = []
                self._err_subs = []
                for j in range(self.n_agents):
                    self._plan_subs.append(
                        rospy.Subscriber(
                            f"{self.topic_prefix}/robot_{j}/plan",
                            UInt8MultiArray,
                            self._on_ros1_plan,
                            callback_args=j,
                            queue_size=1,
                        )
                    )
                    self._err_subs.append(
                        rospy.Subscriber(
                            f"{self.topic_prefix}/robot_{j}/goal_error",
                            Float32MultiArray,
                            self._on_ros1_err,
                            callback_args=j,
                            queue_size=1,
                        )
                    )
                return "ros1"
            except ImportError as exc:
                raise ImportError(
                    "Neither ROS 2 (`rclpy`) nor ROS 1 (`rospy`) was found in the current Python "
                    "environment. Source your ROS setup.bash or use `--comm local`."
                ) from exc

        raise ValueError(f"Unsupported ros_version: {ros_version}")

    def _spin_ros2(self):
        while not self._stop and self._rclpy.ok():
            try:
                self._rclpy.spin_once(self.node, timeout_sec=0.002)
            except Exception:
                break

    def _on_ros2_plan(self, msg, robot_idx: int):
        try:
            raw = bytes(b[0] if isinstance(b, (bytes, bytearray)) else int(b) & 0xFF for b in msg.data)
            r_idx, plan_dict, goal_err, _ = unpack_sphere_plan_bytes(raw, self.tensor_args)
            with self.lock:
                self.plans_board[r_idx] = plan_dict
                if goal_err is not None:
                    self.mean_goal_err[r_idx] = goal_err
        except Exception:
            pass

    def _on_ros2_err(self, msg, robot_idx: int):
        if len(msg.data) >= 2:
            with self.lock:
                self.mean_goal_err[robot_idx] = (float(msg.data[0]), float(msg.data[1]))

    def _on_ros1_plan(self, msg, robot_idx: int):
        try:
            raw = bytes(msg.data)
            r_idx, plan_dict, goal_err, _ = unpack_sphere_plan_bytes(raw, self.tensor_args)
            with self.lock:
                self.plans_board[r_idx] = plan_dict
                if goal_err is not None:
                    self.mean_goal_err[r_idx] = goal_err
        except Exception:
            pass

    def _on_ros1_err(self, msg, robot_idx: int):
        if len(msg.data) >= 2:
            with self.lock:
                self.mean_goal_err[robot_idx] = (float(msg.data[0]), float(msg.data[1]))

    def publish_plan(self, agent_idx: int, plan: dict) -> None:
        with self.lock:
            self.plans_board[agent_idx] = plan
            cur_err = self.mean_goal_err[agent_idx]

        if plan is None or "task_space" not in plan or "spheres" not in plan["task_space"]:
            return

        spheres = plan["task_space"]["spheres"]
        p_err, q_err = cur_err if cur_err is not None else (-1.0, -1.0)
        raw = pack_sphere_plan_bytes(agent_idx, spheres["p"], spheres["r"], p_err, q_err)

        if self._ros_mode == "ros2" and agent_idx in self._plan_pubs:
            msg = self._ByteMultiArray()
            msg.data = [bytes([b]) for b in raw]
            self._plan_pubs[agent_idx].publish(msg)
        elif self._ros_mode == "ros1" and agent_idx in self._plan_pubs:
            msg = self._UInt8MultiArray()
            msg.data = raw
            self._plan_pubs[agent_idx].publish(msg)

    def publish_goal_error(self, agent_idx: int, goal_err: Tuple[float, float]) -> None:
        with self.lock:
            self.mean_goal_err[agent_idx] = goal_err

        if agent_idx in self._err_pubs:
            msg = self._Float32MultiArray()
            msg.data = [float(goal_err[0]), float(goal_err[1])]
            self._err_pubs[agent_idx].publish(msg)

    def get_boards(
        self,
    ) -> Tuple[List[Optional[dict]], List[Optional[Tuple[float, float]]]]:
        with self.lock:
            return list(self.plans_board), list(self.mean_goal_err)

    def close(self) -> None:
        self._stop = True
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=0.5)
        if getattr(self, "_ros_mode", None) == "ros2" and hasattr(self, "node"):
            try:
                self.node.destroy_node()
            except Exception:
                pass
