"""
MR-STORM: Scalable Multi-Arm Control By Distributed MPC (IEEE/RSJ IROS 2026)

Simulator- and hardware-agnostic Python & ROS 2 package for decentralized
multi-arm trajectory optimization using GPU MPPI, horizon plan-sharing,
and dynamic priority conflict resolution.
"""

from mr_storm.core import (
    ArmConfig,
    BroadcastPlanPacket,
    JointCommand,
    MRStormAgent,
    MRStormCoordinator,
    RobotJointState,
    calculate_robot_sphere_count,
    resolve_repo_path,
)
from mr_storm.backends import (
    CallbackHardwareDriver,
    GazeboROS2Backend,
    KinematicBackend,
    RealRobotBackend,
    RobotBackend,
    SafetyLimits,
    SafetySupervisor,
    SingleArmHardwareDriver,
    URRTDEHardwareDriver,
    generate_gazebo_ros2_control_yaml,
)

__version__ = "1.0.0"

__all__ = [
    "ArmConfig",
    "BroadcastPlanPacket",
    "JointCommand",
    "MRStormAgent",
    "MRStormCoordinator",
    "RobotJointState",
    "calculate_robot_sphere_count",
    "resolve_repo_path",
    "CallbackHardwareDriver",
    "GazeboROS2Backend",
    "KinematicBackend",
    "RealRobotBackend",
    "RobotBackend",
    "SafetyLimits",
    "SafetySupervisor",
    "SingleArmHardwareDriver",
    "URRTDEHardwareDriver",
    "generate_gazebo_ros2_control_yaml",
]
