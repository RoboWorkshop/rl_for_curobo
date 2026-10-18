"""
Simulator- and Hardware-Agnostic Embodiment Adapters (`BaseEnvAdapter`).

Bridges any simulator (Isaac Sim, Gazebo, Kinematic/MuJoCo) or real robot hardware
with `MRStormAgnosticLayer` by providing `RobotObservation` and applying
`EmbodimentAction` for each robot in both:
- `synchronous-decentralized` mode (`get_observations()` -> `step_synchronous_decentralized()` -> `apply_actions()`)
- `asynchronous-decentralized` mode (`get_robot_observation(i)` -> `step_single_agent(i)` -> `apply_robot_action(i)`)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from threading import Lock
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from curobo.types.math import Pose

from mr_storm.backends.real_robot_backend import (
    SafetyLimits,
    SafetySupervisor,
    SingleArmHardwareDriver,
)
from mr_storm.core.agnostic_layer import (
    EmbodimentAction,
    MRStormAgnosticLayer,
    RobotObservation,
)
from mr_storm.core.planners import PoseUtils
from mr_storm.core.types import JointCommand, RobotJointState


class BaseEnvAdapter(ABC):
    """
    Base interface for any simulator or real-hardware environment adapter.
    """

    def __init__(self, agnostic_layer: MRStormAgnosticLayer):
        self.layer = agnostic_layer
        self.tensor_args = agnostic_layer.tensor_args

    @abstractmethod
    def get_robot_observation(self, robot_idx: int) -> RobotObservation:
        """Return the latest `RobotObservation` (joint states, goals_W, obstacles) for `robot_idx`."""

    @abstractmethod
    def apply_robot_action(self, robot_idx: int, action: EmbodimentAction) -> None:
        """Command `robot_idx`'s embodiment with `action`."""

    def get_observations(self) -> Dict[int, RobotObservation]:
        """Return observations for all active robots."""
        return {a.idx: self.get_robot_observation(a.idx) for a in self.layer.cu_agents}

    def apply_actions(self, actions: Dict[int, Optional[EmbodimentAction]]) -> None:
        """Apply actions for all active robots."""
        for r_idx, act in actions.items():
            if act is not None:
                self.apply_robot_action(r_idx, act)

    def step_environment(self) -> None:
        """Advance the simulator step (no-op for real hardware or external async simulators)."""

    def is_running(self) -> bool:
        return True

    def close(self) -> None:
        pass


class KinematicEnvAdapter(BaseEnvAdapter):
    """
    Pure-GPU Kinematic Environment Adapter.

    Integrates `EmbodimentAction` directly into joint states and manages world-frame
    goal poses (`goals_W`) with automatic reach/swap target updates when robots converge
    to their targets (matching `ReachTask` / `CbsMp1Task` from `benchmark_sim.py`).
    Ideal for testing `main.py` without Isaac Sim or Gazebo installed, or for dry-running
    custom embodiments.
    """

    def __init__(
        self,
        agnostic_layer: MRStormAgnosticLayer,
        auto_swap_goals: bool = True,
        goal_reach_thresh_m: float = 0.04,
    ):
        super().__init__(agnostic_layer)
        self.auto_swap_goals = auto_swap_goals
        self.goal_reach_thresh_m = goal_reach_thresh_m
        self._lock = Lock()
        self._pose_utils = PoseUtils(seed=42)

        self._obs_by_idx: Dict[int, RobotObservation] = (
            self.layer.get_retract_observations()
        )
        # Set initial cross-workspace / swapped goals so robots actively coordinate
        if len(self._obs_by_idx) > 1 and auto_swap_goals:
            indices = sorted(self._obs_by_idx.keys())
            initial_ee_goals = {
                idx: deepcopy(
                    self._obs_by_idx[idx].goals_W[
                        self.layer.agent_by_idx[idx].planner.ee_link_name
                    ]
                )
                for idx in indices
            }
            for k, idx in enumerate(indices):
                partner_idx = indices[(k + 1) % len(indices)]
                ee_link = self.layer.agent_by_idx[idx].planner.ee_link_name
                # Blend 60% toward partner's retract EE pose so it stays reachable for each base
                own_p = initial_ee_goals[idx].position.clone()
                partner_p = initial_ee_goals[partner_idx].position.clone()
                target_p = 0.4 * own_p + 0.6 * partner_p
                self._obs_by_idx[idx].goals_W[ee_link] = Pose(
                    position=target_p,
                    quaternion=initial_ee_goals[idx].quaternion.clone(),
                )

    def set_robot_goal_W(
        self,
        robot_idx: int,
        position_W: List[float],
        quaternion_W: List[float],
        link_name: Optional[str] = None,
    ) -> None:
        with self._lock:
            a = self.layer.agent_by_idx[robot_idx]
            ln = link_name or a.planner.ee_link_name
            self._obs_by_idx[robot_idx].goals_W[ln] = Pose(
                position=self.tensor_args.to_device(
                    torch.tensor(position_W, dtype=torch.float32).unsqueeze(0)
                ),
                quaternion=self.tensor_args.to_device(
                    torch.tensor(quaternion_W, dtype=torch.float32).unsqueeze(0)
                ),
            )

    def get_robot_observation(self, robot_idx: int) -> RobotObservation:
        with self._lock:
            obs = self._obs_by_idx[robot_idx]
            return RobotObservation(
                dof_names=list(obs.dof_names),
                positions=obs.positions.copy(),
                velocities=obs.velocities.copy(),
                prev_positions=(
                    obs.prev_positions.copy() if obs.prev_positions is not None else None
                ),
                prev_velocities=(
                    obs.prev_velocities.copy() if obs.prev_velocities is not None else None
                ),
                goals_W={k: v.clone() for k, v in obs.goals_W.items()},
                obstacle_poses_W=obs.obstacle_poses_W,
            )

    def apply_robot_action(self, robot_idx: int, action: EmbodimentAction) -> None:
        with self._lock:
            obs = self._obs_by_idx[robot_idx]
            obs.prev_positions = obs.positions.copy()
            obs.prev_velocities = obs.velocities.copy()
            obs.positions = action.positions.astype(np.float32).copy()
            if action.velocities is not None:
                obs.velocities = action.velocities.astype(np.float32).copy()
            else:
                obs.velocities = (obs.positions - obs.prev_positions) / 0.03

            if (
                self.auto_swap_goals
                and action.mean_goal_err is not None
                and action.mean_goal_err[0] < self.goal_reach_thresh_m
            ):
                a = self.layer.agent_by_idx[robot_idx]
                ee_link = a.planner.ee_link_name
                base_xyz = a.base_pose[:3]
                new_pos = self._pose_utils.sample_pos_in_box(
                    [base_xyz[0], base_xyz[1], base_xyz[2] + 0.45], 0.45
                )
                cur_q = obs.goals_W[ee_link].quaternion.clone()
                obs.goals_W[ee_link] = Pose(
                    position=self.tensor_args.to_device(
                        torch.tensor(new_pos, dtype=torch.float32).unsqueeze(0)
                    ),
                    quaternion=cur_q,
                )


class GazeboROSAdapter(BaseEnvAdapter):
    """
    Gazebo (`ros2_control` / `ros_control`) Environment Adapter.

    Subscribes to `/<robot_ns>/joint_states` (`sensor_msgs/JointState`) and
    `/<robot_ns>/goal_pose` (`geometry_msgs/PoseStamped`) for each robot, and
    publishes `EmbodimentAction` to `/<robot_ns>/<controller_name>/joint_trajectory`
    (or `/commands`).
    Supports both `synchronous-decentralized` and `asynchronous-decentralized` modes!
    """

    def __init__(
        self,
        agnostic_layer: MRStormAgnosticLayer,
        namespace_prefix: str = "arm",
        controller_name: str = "joint_trajectory_controller",
        command_interface: str = "trajectory",
        step_dt: float = 0.033,
    ):
        super().__init__(agnostic_layer)
        self.namespace_prefix = namespace_prefix
        self.controller_name = controller_name
        self.command_interface = command_interface
        self.step_dt = step_dt
        self._lock = Lock()
        self._obs_by_idx: Dict[int, RobotObservation] = (
            self.layer.get_retract_observations()
        )
        self._init_ros_interfaces()

    def _init_ros_interfaces(self):
        import rclpy
        from rclpy.duration import Duration
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState as RosJointState
        from std_msgs.msg import Float64MultiArray
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        self._rclpy = rclpy
        self._Duration = Duration
        self._JointTrajectory = JointTrajectory
        self._JointTrajectoryPoint = JointTrajectoryPoint
        self._Float64MultiArray = Float64MultiArray

        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node("mr_storm_gazebo_adapter")
        self._cmd_pubs = {}
        self._js_subs = []
        self._goal_subs = []

        for a in self.layer.cu_agents:
            ns = f"/{self.namespace_prefix}_{a.idx}"
            self._js_subs.append(
                self.node.create_subscription(
                    RosJointState,
                    f"{ns}/joint_states",
                    lambda msg, idx=a.idx: self._on_js(msg, idx),
                    10,
                )
            )
            self._goal_subs.append(
                self.node.create_subscription(
                    PoseStamped,
                    f"{ns}/goal_pose",
                    lambda msg, idx=a.idx: self._on_goal(msg, idx),
                    10,
                )
            )
            if self.command_interface == "trajectory":
                self._cmd_pubs[a.idx] = self.node.create_publisher(
                    JointTrajectory,
                    f"{ns}/{self.controller_name}/joint_trajectory",
                    10,
                )
            else:
                self._cmd_pubs[a.idx] = self.node.create_publisher(
                    Float64MultiArray,
                    f"{ns}/{self.controller_name}/commands",
                    10,
                )

    def _on_js(self, msg, robot_idx: int):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        with self._lock:
            obs = self._obs_by_idx[robot_idx]
            if not all(jn in name_to_idx for jn in obs.dof_names):
                return
            obs.prev_positions = obs.positions.copy()
            obs.prev_velocities = obs.velocities.copy()
            obs.positions = np.array(
                [msg.position[name_to_idx[jn]] for jn in obs.dof_names], dtype=np.float32
            )
            if len(msg.velocity) >= len(msg.name):
                obs.velocities = np.array(
                    [msg.velocity[name_to_idx[jn]] for jn in obs.dof_names],
                    dtype=np.float32,
                )

    def _on_goal(self, msg, robot_idx: int):
        p = msg.pose.position
        q = msg.pose.orientation
        a = self.layer.agent_by_idx[robot_idx]
        ee_link = a.planner.ee_link_name
        with self._lock:
            self._obs_by_idx[robot_idx].goals_W[ee_link] = Pose(
                position=self.tensor_args.to_device(
                    torch.tensor([[p.x, p.y, p.z]], dtype=torch.float32)
                ),
                quaternion=self.tensor_args.to_device(
                    torch.tensor([[q.w, q.x, q.y, q.z]], dtype=torch.float32)
                ),
            )

    def get_robot_observation(self, robot_idx: int) -> RobotObservation:
        self._rclpy.spin_once(self.node, timeout_sec=0.001)
        with self._lock:
            obs = self._obs_by_idx[robot_idx]
            return RobotObservation(
                dof_names=list(obs.dof_names),
                positions=obs.positions.copy(),
                velocities=obs.velocities.copy(),
                prev_positions=(
                    obs.prev_positions.copy() if obs.prev_positions is not None else None
                ),
                prev_velocities=(
                    obs.prev_velocities.copy() if obs.prev_velocities is not None else None
                ),
                goals_W={k: v.clone() for k, v in obs.goals_W.items()},
                obstacle_poses_W=obs.obstacle_poses_W,
            )

    def apply_robot_action(self, robot_idx: int, action: EmbodimentAction) -> None:
        pub = self._cmd_pubs.get(robot_idx)
        if pub is None:
            return
        if self.command_interface == "trajectory":
            traj = self._JointTrajectory()
            traj.joint_names = list(action.joint_names)
            pt = self._JointTrajectoryPoint()
            pt.positions = [float(x) for x in action.positions]
            if action.velocities is not None:
                pt.velocities = [float(x) for x in action.velocities]
            if action.accelerations is not None:
                pt.accelerations = [float(x) for x in action.accelerations]
            sec = int(self.step_dt)
            nanosec = int((self.step_dt - sec) * 1e9)
            pt.time_from_start = self._Duration(seconds=sec, nanoseconds=nanosec).to_msg()
            traj.points = [pt]
            pub.publish(traj)
        else:
            arr = self._Float64MultiArray()
            arr.data = [float(x) for x in action.positions]
            pub.publish(arr)

    def close(self) -> None:
        if hasattr(self, "node"):
            try:
                self.node.destroy_node()
            except Exception:
                pass


class RealHardwareEnvAdapter(BaseEnvAdapter):
    """
    Real Robot Hardware Environment Adapter with `SafetySupervisor`.

    Connects each `CuAgent` to any physical robot driver (`SingleArmHardwareDriver`,
    `URRTDEHardwareDriver`, `CallbackHardwareDriver`, or custom user driver) and
    automatically clamps joint position steps, velocities, and accelerations before
    commanding hardware.
    """

    def __init__(
        self,
        agnostic_layer: MRStormAgnosticLayer,
        drivers: Dict[int, SingleArmHardwareDriver],
        safety_limits: Optional[SafetyLimits] = None,
    ):
        super().__init__(agnostic_layer)
        self.drivers = drivers
        self.safety = SafetySupervisor(safety_limits or SafetyLimits())
        self._obs_by_idx: Dict[int, RobotObservation] = (
            self.layer.get_retract_observations()
        )
        for drv in self.drivers.values():
            drv.connect()

    def get_robot_observation(self, robot_idx: int) -> RobotObservation:
        drv = self.drivers[robot_idx]
        hw_js = drv.read_joint_state()
        obs = self._obs_by_idx[robot_idx]
        obs.prev_positions = obs.positions.copy()
        obs.prev_velocities = obs.velocities.copy()
        obs.positions = np.asarray(hw_js.positions, dtype=np.float32)
        obs.velocities = np.asarray(hw_js.velocities, dtype=np.float32)
        return RobotObservation(
            dof_names=list(obs.dof_names),
            positions=obs.positions.copy(),
            velocities=obs.velocities.copy(),
            prev_positions=obs.prev_positions.copy(),
            prev_velocities=obs.prev_velocities.copy(),
            goals_W={k: v.clone() for k, v in obs.goals_W.items()},
            obstacle_poses_W=obs.obstacle_poses_W,
        )

    def apply_robot_action(self, robot_idx: int, action: EmbodimentAction) -> None:
        obs = self._obs_by_idx[robot_idx]
        cur_js = RobotJointState(
            positions=obs.positions,
            velocities=obs.velocities,
            joint_names=action.joint_names,
        )
        raw_cmd = JointCommand(
            positions=action.positions,
            velocities=(
                action.velocities
                if action.velocities is not None
                else np.zeros_like(action.positions)
            ),
            accelerations=(
                action.accelerations
                if action.accelerations is not None
                else np.zeros_like(action.positions)
            ),
            joint_names=action.joint_names,
            dt=0.03,
        )
        safe_cmd = self.safety.filter_command(cur_js, raw_cmd)
        self.drivers[robot_idx].send_joint_command(safe_cmd)

    def close(self) -> None:
        for drv in self.drivers.values():
            try:
                drv.stop()
            except Exception:
                pass
