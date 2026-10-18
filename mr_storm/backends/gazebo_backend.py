"""
Gazebo Simulator Backend (`GazeboROS2Backend`) for MR-STORM.

Supports both modern Gazebo (Gazebo Sim / Ignition: Fortress, Garden, Harmonic)
via `gz_ros2_control` and Gazebo Classic via `gazebo_ros2_control`.

Connects each robotic arm `i` to:
- State subscription: `/<arm_namespace>/joint_states` (`sensor_msgs/msg/JointState`)
  or a shared `/joint_states` topic with joint-name filtering.
- Command publication:
  * `"trajectory"` mode: `/<arm_namespace>/joint_trajectory_controller/joint_trajectory`
    (`trajectory_msgs/msg/JointTrajectory`)
  * `"position"` mode: `/<arm_namespace>/forward_position_controller/commands`
    (`std_msgs/msg/Float64MultiArray`)
  * `"velocity"` mode: `/<arm_namespace>/velocity_controller/commands`
    (`std_msgs/msg/Float64MultiArray`)
"""

from __future__ import annotations

from typing import Dict, List, Optional
import threading
import time
import numpy as np

from mr_storm.backends.base import RobotBackend
from mr_storm.core.types import JointCommand, RobotJointState


class GazeboROS2Backend(RobotBackend):
    """
    Multi-arm Gazebo backend communicating over ROS 2 (`rclpy`) and `ros2_control`.
    """

    def __init__(
        self,
        arm_joint_names: Dict[int, List[str]],
        arm_namespaces: Optional[Dict[int, str]] = None,
        command_interface: str = "trajectory",  # "trajectory" | "position" | "velocity"
        shared_joint_states_topic: Optional[str] = None,
        controller_name: Optional[str] = None,
        control_freq_hz: float = 30.0,
        node_name: str = "mr_storm_gazebo_bridge",
    ):
        super().__init__(n_arms=len(arm_joint_names), control_freq_hz=control_freq_hz)
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.duration import Duration
            from sensor_msgs.msg import JointState as RosJointState
            from std_msgs.msg import Float64MultiArray
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        except ImportError as exc:
            raise ImportError(
                "ROS 2 Python packages (`rclpy`, `sensor_msgs`, `trajectory_msgs`) are required "
                "for GazeboROS2Backend. Source your ROS 2 workspace (e.g. `source /opt/ros/humble/setup.bash`)."
            ) from exc

        self._rclpy = rclpy
        self._Duration = Duration
        self._Float64MultiArray = Float64MultiArray
        self._JointTrajectory = JointTrajectory
        self._JointTrajectoryPoint = JointTrajectoryPoint

        if not rclpy.ok():
            rclpy.init()

        self.node: Node = rclpy.create_node(node_name)
        self.arm_joint_names = arm_joint_names
        self.arm_namespaces = arm_namespaces or {
            arm_id: f"arm_{arm_id}" for arm_id in arm_joint_names
        }
        assert command_interface in ("trajectory", "position", "velocity"), (
            f"Unsupported command_interface '{command_interface}'. Choose from 'trajectory', 'position', 'velocity'."
        )
        self.command_interface = command_interface

        # Default latest states initialized to zeros until first `/joint_states` message arrives
        self._states_lock = threading.Lock()
        self._latest_states: Dict[int, RobotJointState] = {
            arm_id: RobotJointState(
                positions=np.zeros(len(j_names), dtype=np.float32),
                velocities=np.zeros(len(j_names), dtype=np.float32),
                joint_names=list(j_names),
            )
            for arm_id, j_names in arm_joint_names.items()
        }
        self._received_first_state: Dict[int, bool] = {arm_id: False for arm_id in arm_joint_names}

        # Set up subscribers
        self._subs = []
        if shared_joint_states_topic:
            sub = self.node.create_subscription(
                RosJointState,
                shared_joint_states_topic,
                self._on_shared_joint_state,
                10,
            )
            self._subs.append(sub)
        else:
            for arm_id, ns in self.arm_namespaces.items():
                topic = f"/{ns}/joint_states" if ns else "/joint_states"
                sub = self.node.create_subscription(
                    RosJointState,
                    topic,
                    lambda msg, a_id=arm_id: self._on_arm_joint_state(a_id, msg),
                    10,
                )
                self._subs.append(sub)

        # Set up command publishers per arm
        self._pubs = {}
        for arm_id, ns in self.arm_namespaces.items():
            prefix = f"/{ns}" if ns else ""
            if command_interface == "trajectory":
                ctrl = controller_name or "joint_trajectory_controller"
                topic = f"{prefix}/{ctrl}/joint_trajectory"
                self._pubs[arm_id] = self.node.create_publisher(JointTrajectory, topic, 10)
            elif command_interface == "position":
                ctrl = controller_name or "forward_position_controller"
                topic = f"{prefix}/{ctrl}/commands"
                self._pubs[arm_id] = self.node.create_publisher(Float64MultiArray, topic, 10)
            else:
                ctrl = controller_name or "velocity_controller"
                topic = f"{prefix}/{ctrl}/commands"
                self._pubs[arm_id] = self.node.create_publisher(Float64MultiArray, topic, 10)

        # Background spin thread for non-blocking state reception
        self._spin_running = True
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

    def _spin_loop(self):
        while self._spin_running and self._rclpy.ok():
            self._rclpy.spin_once(self.node, timeout_sec=0.005)

    def _extract_ordered_state(self, arm_id: int, msg) -> Optional[RobotJointState]:
        target_names = self.arm_joint_names[arm_id]
        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        if not all(jn in name_to_idx for jn in target_names):
            return None

        pos = np.array([msg.position[name_to_idx[jn]] for jn in target_names], dtype=np.float32)
        if len(msg.velocity) >= len(msg.name):
            vel = np.array([msg.velocity[name_to_idx[jn]] for jn in target_names], dtype=np.float32)
        else:
            vel = np.zeros_like(pos)
        return RobotJointState(
            positions=pos,
            velocities=vel,
            joint_names=list(target_names),
            timestamp=time.time(),
        )

    def _on_arm_joint_state(self, arm_id: int, msg):
        st = self._extract_ordered_state(arm_id, msg)
        if st is not None:
            with self._states_lock:
                self._latest_states[arm_id] = st
                self._received_first_state[arm_id] = True

    def _on_shared_joint_state(self, msg):
        for arm_id in self.arm_joint_names:
            self._on_arm_joint_state(arm_id, msg)

    def wait_for_joint_states(self, timeout_sec: float = 10.0) -> bool:
        """Block until all arms have received at least one `/joint_states` message from Gazebo."""
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            with self._states_lock:
                if all(self._received_first_state.values()):
                    return True
            time.sleep(0.02)
        return False

    def get_joint_states(self) -> Dict[int, RobotJointState]:
        with self._states_lock:
            return {k: v for k, v in self._latest_states.items()}

    def send_joint_commands(self, commands: Dict[int, JointCommand]) -> None:
        for arm_id, cmd in commands.items():
            pub = self._pubs[arm_id]
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
                pub.publish(traj)
            elif self.command_interface == "position":
                msg = self._Float64MultiArray()
                msg.data = [float(x) for x in cmd.positions]
                pub.publish(msg)
            else:
                msg = self._Float64MultiArray()
                msg.data = [float(x) for x in cmd.velocities]
                pub.publish(msg)

    def shutdown(self) -> None:
        self._spin_running = False
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)
        self.node.destroy_node()


def generate_gazebo_ros2_control_yaml(
    arm_joint_names: Dict[int, List[str]],
    update_rate_hz: int = 100,
) -> str:
    """
    Generate a ready-to-use `ros2_control` controller manager YAML configuration
    for running N arms in Gazebo (Harmonic / Ignition / Classic) with MR-STORM.
    """
    lines = [
        "# Auto-generated ros2_control configuration for MR-STORM in Gazebo",
        "controller_manager:",
        "  ros__parameters:",
        f"    update_rate: {update_rate_hz}",
        "    joint_state_broadcaster:",
        "      type: joint_state_broadcaster/JointStateBroadcaster",
    ]
    for arm_id in arm_joint_names:
        lines.extend(
            [
                f"    arm_{arm_id}_trajectory_controller:",
                "      type: joint_trajectory_controller/JointTrajectoryController",
                f"    arm_{arm_id}_position_controller:",
                "      type: position_controllers/JointGroupPositionController",
            ]
        )
    lines.append("")
    for arm_id, j_names in arm_joint_names.items():
        joints_list_str = ", ".join(f'"{jn}"' for jn in j_names)
        lines.extend(
            [
                f"arm_{arm_id}_trajectory_controller:",
                "  ros__parameters:",
                f"    joints: [{joints_list_str}]",
                '    command_interfaces: ["position", "velocity"]',
                '    state_interfaces: ["position", "velocity"]',
                "    allow_nonzero_velocity_at_trajectory_end: true",
                "",
                f"arm_{arm_id}_position_controller:",
                "  ros__parameters:",
                f"    joints: [{joints_list_str}]",
                "",
            ]
        )
    return "\n".join(lines)
