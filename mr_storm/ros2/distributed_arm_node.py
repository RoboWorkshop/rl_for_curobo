#!/usr/bin/env python3
"""
True Distributed Multi-Arm ROS 2 Node (`MRStormDistributedArmNode`).

Each arm `i in {0 .. N-1}` runs in its own independent OS process (or on its own
dedicated GPU / PC connected over a LAN) and coordinates purely via decentralized
ROS 2 DDS plan broadcasting (`/mr_storm/plan_broadcast`):

1. Subscribes to `/<arm_ns>/joint_states` (`sensor_msgs/msg/JointState`)
2. Subscribes to `/<arm_ns>/goal_pose` (`geometry_msgs/msg/PoseStamped`)
3. Subscribes to `/mr_storm/plan_broadcast` (`std_msgs/msg/ByteMultiArray`) from all peer arms `j != i`
   - Automatically compensates for network latency by shifting delayed peer horizon
     predictions forward in time (`shift_steps = int(latency / dt)`).
4. Runs GPU MPPI (`MRStormAgent.step`) with dynamic priority conflict resolution (`pose_wta` / tau)
5. Broadcasts its own predicted horizon collision spheres (`BroadcastPlanPacket`) on `/mr_storm/plan_broadcast`
6. Publishes safety-filtered joint commands to Gazebo (`ros2_control`) or physical robot drivers
7. Publishes RViz2 `visualization_msgs/msg/MarkerArray` of predicted horizon collision spheres
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import numpy as np

from mr_storm.backends.real_robot_backend import SafetyLimits, SafetySupervisor
from mr_storm.core.agent import MRStormAgent, resolve_repo_path
from mr_storm.core.types import (
    ArmConfig,
    BroadcastPlanPacket,
    JointCommand,
    RobotJointState,
)


# Distinct RGBA colors for visualizing each arm's predicted horizon spheres in RViz2
ARM_RVIZ_COLORS = [
    (1.0, 0.45, 0.1, 0.55),  # Orange
    (0.15, 0.55, 1.0, 0.55),  # Blue
    (0.15, 0.85, 0.35, 0.55),  # Green
    (0.95, 0.2, 0.3, 0.55),  # Red
    (0.7, 0.3, 0.95, 0.55),  # Purple
    (0.95, 0.85, 0.15, 0.55),  # Yellow
]


def load_fleet_arm_configs(fleet_yaml_path: str) -> List[ArmConfig]:
    """Load a list of `ArmConfig` definitions from a fleet YAML file."""
    resolved = resolve_repo_path(fleet_yaml_path)
    with open(resolved, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    default_robot = data.get("robot_yml", "ur5e.yml")
    default_particle = data.get(
        "particle_yml", "projects_root/experiments/benchmarks/cfgs/particle/O.yml"
    )
    arms_raw = data["arms"]
    configs: List[ArmConfig] = []
    for idx, item in enumerate(arms_raw):
        r_id = int(item.get("robot_id", idx))
        configs.append(
            ArmConfig(
                robot_id=r_id,
                robot_yml=item.get("robot_yml", default_robot),
                base_pose=list(item["base_pose"]),
                particle_yml=item.get("particle_yml", default_particle),
                retract_cfg=item.get("retract_cfg"),
                namespace=item.get("namespace", f"arm_{r_id}"),
            )
        )
    return configs


def shift_peer_packet_for_latency(
    pkt: BroadcastPlanPacket,
    now_ts: float,
    step_dt: float,
    max_stale_sec: float = 0.5,
) -> Optional[BroadcastPlanPacket]:
    """
    Compensate for communication delay in distributed ROS 2 setups by shifting
    the peer's horizon sphere trajectory forward by `k = int(age / step_dt)` steps
    and repeating the terminal horizon state.
    """
    age = max(0.0, now_ts - pkt.timestamp)
    if age > max_stale_sec:
        return None
    shift_k = int(age / max(step_dt, 1e-3))
    if shift_k <= 0:
        return pkt
    shift_k = min(shift_k, pkt.horizon - 1)
    shifted_pos = np.empty_like(pkt.sphere_positions)
    shifted_pos[: pkt.horizon - shift_k] = pkt.sphere_positions[shift_k:]
    shifted_pos[pkt.horizon - shift_k :] = pkt.sphere_positions[-1:]
    return BroadcastPlanPacket(
        robot_id=pkt.robot_id,
        timestamp=pkt.timestamp,
        horizon=pkt.horizon,
        n_spheres=pkt.n_spheres,
        sphere_positions=shifted_pos,
        sphere_radii=pkt.sphere_radii,
        pos_error=pkt.pos_error,
        rot_error=pkt.rot_error,
        base_pose=pkt.base_pose,
    )


class MRStormDistributedArmNode:
    """
    Decentralized ROS 2 controller node for a single robotic arm `robot_id`.
    """

    def __init__(
        self,
        robot_id: int,
        all_arm_cfgs: List[ArmConfig],
        control_freq_hz: float = 30.0,
        command_interface: str = "trajectory",
        controller_name: str = "joint_trajectory_controller",
        enable_safety_supervisor: bool = True,
        publish_rviz_spheres: bool = True,
        initial_goal_W: Optional[List[float]] = None,
    ):
        try:
            import rclpy
            from rclpy.duration import Duration
            from rclpy.node import Node
            from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
            from geometry_msgs.msg import PoseStamped
            from sensor_msgs.msg import JointState as RosJointState
            from std_msgs.msg import ByteMultiArray, Float64MultiArray
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
            from visualization_msgs.msg import Marker, MarkerArray
        except ImportError as exc:
            raise ImportError(
                "ROS 2 (`rclpy`, `sensor_msgs`, `trajectory_msgs`, `geometry_msgs`, "
                "`visualization_msgs`) is required to run `MRStormDistributedArmNode`."
            ) from exc

        self._rclpy = rclpy
        self._Duration = Duration
        self._ByteMultiArray = ByteMultiArray
        self._Float64MultiArray = Float64MultiArray
        self._JointTrajectory = JointTrajectory
        self._JointTrajectoryPoint = JointTrajectoryPoint
        self._Marker = Marker
        self._MarkerArray = MarkerArray

        if not rclpy.ok():
            rclpy.init()

        self.arm_cfg = all_arm_cfgs[robot_id]
        self.robot_id = robot_id
        self.ns = self.arm_cfg.namespace
        self.node: Node = rclpy.create_node(f"mr_storm_arm_{robot_id}_node")

        self.node.get_logger().info(
            f"Initializing decentralized MR-STORM GPU solver for arm {robot_id} (namespace='/{self.ns}')..."
        )
        self.agent = MRStormAgent.create_distributed(self.arm_cfg, all_arm_cfgs)

        if initial_goal_W is not None and len(initial_goal_W) == 7:
            self.agent.set_goal_world(initial_goal_W[:3], initial_goal_W[3:])

        self.control_freq_hz = control_freq_hz
        self.dt = 1.0 / max(control_freq_hz, 1.0)
        self.command_interface = command_interface
        self.enable_safety = enable_safety_supervisor
        self.safety_supervisor = SafetySupervisor(SafetyLimits())
        self.publish_rviz_spheres = publish_rviz_spheres

        # Initialize current joint state at retract configuration until first JointState arrives
        retract_np = (
            self.agent.solver.rollout_fn.dynamics_model.retract_config.detach().cpu().numpy().reshape(-1)
        )
        self.current_joint_state = RobotJointState(
            positions=retract_np.copy(),
            velocities=np.zeros_like(retract_np),
            joint_names=list(self.agent.joint_names),
        )
        self.peer_packets: Dict[int, BroadcastPlanPacket] = {}

        # Low-latency Best-Effort QoS for inter-arm plan broadcasting
        plan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # 1. Subscribe to own joint states
        self.sub_js = self.node.create_subscription(
            RosJointState,
            f"/{self.ns}/joint_states",
            self._on_joint_state,
            10,
        )
        # 2. Subscribe to runtime goal pose updates
        self.sub_goal = self.node.create_subscription(
            PoseStamped,
            f"/{self.ns}/goal_pose",
            self._on_goal_pose,
            10,
        )
        # 3. Subscribe to peer plan broadcasts
        self.sub_plans = self.node.create_subscription(
            ByteMultiArray,
            "/mr_storm/plan_broadcast",
            self._on_plan_broadcast,
            plan_qos,
        )

        # 4. Publisher for own plan broadcasts
        self.pub_plan = self.node.create_publisher(
            ByteMultiArray,
            "/mr_storm/plan_broadcast",
            plan_qos,
        )

        # 5. Publisher for joint commands (Gazebo ros2_control or real robot driver)
        if command_interface == "trajectory":
            self.pub_cmd = self.node.create_publisher(
                JointTrajectory,
                f"/{self.ns}/{controller_name}/joint_trajectory",
                10,
            )
        else:
            self.pub_cmd = self.node.create_publisher(
                Float64MultiArray,
                f"/{self.ns}/{controller_name}/commands",
                10,
            )

        # 6. Optional RViz2 MarkerArray publisher for horizon spheres
        if publish_rviz_spheres:
            self.pub_markers = self.node.create_publisher(
                MarkerArray,
                f"/{self.ns}/horizon_spheres",
                5,
            )

        # Control loop timer
        self.timer = self.node.create_timer(self.dt, self._control_step)
        self.node.get_logger().info(
            f"MR-STORM Arm {robot_id} ready! Running decentralized MPC at {control_freq_hz:.1f} Hz."
        )

    def _on_joint_state(self, msg):
        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        target_names = self.agent.joint_names
        if not all(jn in name_to_idx for jn in target_names):
            return
        pos = np.array([msg.position[name_to_idx[jn]] for jn in target_names], dtype=np.float32)
        if len(msg.velocity) >= len(msg.name):
            vel = np.array([msg.velocity[name_to_idx[jn]] for jn in target_names], dtype=np.float32)
        else:
            vel = np.zeros_like(pos)
        self.current_joint_state = RobotJointState(
            positions=pos,
            velocities=vel,
            joint_names=list(target_names),
            timestamp=time.time(),
        )

    def _on_goal_pose(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation
        pos_W = np.array([p.x, p.y, p.z], dtype=np.float32)
        quat_W = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
        self.agent.set_goal_world(pos_W, quat_W)
        self.node.get_logger().info(
            f"[Arm {self.robot_id}] Updated goal pose in world frame: pos={pos_W.round(3)}, quat={quat_W.round(3)}"
        )

    def _on_plan_broadcast(self, msg):
        raw_bytes = bytes(msg.data)
        pkt = BroadcastPlanPacket.from_bytes(raw_bytes)
        if pkt.robot_id == self.robot_id:
            return
        self.peer_packets[pkt.robot_id] = pkt

    def _control_step(self):
        now = time.time()
        # 1. Build and broadcast own horizon collision spheres + priority goal error
        own_pkt = self.agent.build_broadcast_plan(self.current_joint_state)
        b_msg = self._ByteMultiArray()
        b_msg.data = [bytes([b]) if isinstance(b, int) else b for b in own_pkt.to_bytes()]
        self.pub_plan.publish(b_msg)

        # 2. Shift peer plans for network latency and update DynamicObsCollPredictor
        active_peers: Dict[int, BroadcastPlanPacket] = {}
        for peer_id, pkt in list(self.peer_packets.items()):
            shifted = shift_peer_packet_for_latency(pkt, now, self.dt)
            if shifted is not None:
                active_peers[peer_id] = shifted
        self.agent.update_from_peer_plans(active_peers)

        # 3. Solve local GPU MPPI step
        raw_cmd = self.agent.step(self.current_joint_state)
        cmd = (
            self.safety_supervisor.filter_command(self.current_joint_state, raw_cmd)
            if self.enable_safety
            else raw_cmd
        )

        # 4. Publish command to Gazebo / Real Robot controller
        if self.command_interface == "trajectory":
            traj = self._JointTrajectory()
            traj.joint_names = list(cmd.joint_names)
            pt = self._JointTrajectoryPoint()
            pt.positions = [float(x) for x in cmd.positions]
            pt.velocities = [float(x) for x in cmd.velocities]
            pt.accelerations = [float(x) for x in cmd.accelerations]
            sec = int(cmd.dt)
            nanosec = int((cmd.dt - sec) * 1e9)
            pt.time_from_start = self._Duration(seconds=sec, nanoseconds=nanosec).to_msg()
            traj.points = [pt]
            self.pub_cmd.publish(traj)
        else:
            arr = self._Float64MultiArray()
            arr.data = [float(x) for x in cmd.positions]
            self.pub_cmd.publish(arr)

        # 5. Publish RViz2 collision sphere markers (terminal + mid-horizon steps)
        if self.publish_rviz_spheres:
            self._publish_rviz_markers(own_pkt)

    def _publish_rviz_markers(self, pkt: BroadcastPlanPacket):
        ma = self._MarkerArray()
        rgba = ARM_RVIZ_COLORS[self.robot_id % len(ARM_RVIZ_COLORS)]
        sample_steps = [0, pkt.horizon // 2, pkt.horizon - 1]
        marker_id = 0
        for h_idx in sample_steps:
            alpha_scale = 1.0 if h_idx == 0 else (0.5 if h_idx == pkt.horizon // 2 else 0.25)
            for s_idx in range(pkt.n_spheres):
                rad = float(pkt.sphere_radii[s_idx])
                if rad <= 0.0:
                    continue
                pos = pkt.sphere_positions[h_idx, s_idx]
                m = self._Marker()
                m.header.frame_id = "world"
                m.ns = f"arm_{self.robot_id}_h{h_idx}"
                m.id = marker_id
                marker_id += 1
                m.type = self._Marker.SPHERE
                m.action = self._Marker.ADD
                m.pose.position.x = float(pos[0])
                m.pose.position.y = float(pos[1])
                m.pose.position.z = float(pos[2])
                m.pose.orientation.w = 1.0
                m.scale.x = rad * 2.0
                m.scale.y = rad * 2.0
                m.scale.z = rad * 2.0
                m.color.r = rgba[0]
                m.color.g = rgba[1]
                m.color.b = rgba[2]
                m.color.a = rgba[3] * alpha_scale
                ma.markers.append(m)
        self.pub_markers.publish(ma)

    def spin(self):
        try:
            self._rclpy.spin(self.node)
        except KeyboardInterrupt:
            pass
        finally:
            self.node.destroy_node()


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Run a decentralized MR-STORM arm controller node over ROS 2."
    )
    parser.add_argument(
        "--robot_id",
        type=int,
        required=True,
        help="Index of this arm in the fleet (0 .. N-1).",
    )
    parser.add_argument(
        "--fleet_config",
        type=str,
        default="mr_storm/ros2/config/quad_ur5e_fleet.yaml",
        help="Path to the multi-arm fleet YAML configuration.",
    )
    parser.add_argument(
        "--control_freq",
        type=float,
        default=30.0,
        help="MPC control frequency in Hz (default: 30.0).",
    )
    parser.add_argument(
        "--command_interface",
        type=str,
        default="trajectory",
        choices=["trajectory", "position"],
        help="ROS 2 controller interface type ('trajectory' or 'position').",
    )
    parser.add_argument(
        "--controller_name",
        type=str,
        default="joint_trajectory_controller",
        help="Controller name under /<arm_ns>/<controller_name>.",
    )
    parser.add_argument(
        "--no_safety_supervisor",
        action="store_true",
        help="Disable real-robot safety step clamping.",
    )
    args, _ = parser.parse_known_args(argv)

    all_cfgs = load_fleet_arm_configs(args.fleet_config)
    node = MRStormDistributedArmNode(
        robot_id=args.robot_id,
        all_arm_cfgs=all_cfgs,
        control_freq_hz=args.control_freq,
        command_interface=args.command_interface,
        controller_name=args.controller_name,
        enable_safety_supervisor=not args.no_safety_supervisor,
    )
    node.spin()


if __name__ == "__main__":
    main()
