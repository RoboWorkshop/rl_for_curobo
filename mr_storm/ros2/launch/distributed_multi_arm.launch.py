"""
ROS 2 Launch file to spawn N independent decentralized MR-STORM arm controller processes.

Usage:
    ros2 launch mr_storm/ros2/launch/distributed_multi_arm.launch.py num_arms:=4
"""

from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
NODE_SCRIPT = str(REPO_ROOT / "mr_storm" / "ros2" / "distributed_arm_node.py")


def _launch_setup(context, *args, **kwargs):
    num_arms = int(LaunchConfiguration("num_arms").perform(context))
    fleet_config = LaunchConfiguration("fleet_config").perform(context)
    control_freq = LaunchConfiguration("control_freq").perform(context)
    command_interface = LaunchConfiguration("command_interface").perform(context)

    processes = []
    for arm_id in range(num_arms):
        processes.append(
            ExecuteProcess(
                cmd=[
                    sys.executable,
                    NODE_SCRIPT,
                    "--robot_id",
                    str(arm_id),
                    "--fleet_config",
                    fleet_config,
                    "--control_freq",
                    control_freq,
                    "--command_interface",
                    command_interface,
                ],
                name=f"mr_storm_arm_{arm_id}",
                output="screen",
            )
        )
    return processes


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "num_arms",
                default_value="4",
                description="Number of decentralized MR-STORM arm nodes to launch.",
            ),
            DeclareLaunchArgument(
                "fleet_config",
                default_value="mr_storm/ros2/config/quad_ur5e_fleet.yaml",
                description="Path to the fleet YAML configuration.",
            ),
            DeclareLaunchArgument(
                "control_freq",
                default_value="30.0",
                description="MPC control frequency in Hz.",
            ),
            DeclareLaunchArgument(
                "command_interface",
                default_value="trajectory",
                description="ROS 2 controller interface ('trajectory' or 'position').",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
