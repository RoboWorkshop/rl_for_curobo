#!/usr/bin/env python3
"""
Example 1: Multi-Arm MR-STORM in Gazebo (Gazebo Harmonic / Ignition / Classic).

Demonstrates connecting `MRStormCoordinator` to a multi-arm Gazebo simulation
via `GazeboROS2Backend` (`ros2_control`).

Usage:
    # 1. Start your multi-arm Gazebo simulation with ros2_control controllers active
    # 2. Run this script:
    python3 examples/01_gazebo_multi_arm.py --fleet_config mr_storm/ros2/config/quad_ur5e_fleet.yaml
"""

from __future__ import annotations

import argparse
import numpy as np

from mr_storm.backends.gazebo_backend import (
    GazeboROS2Backend,
    generate_gazebo_ros2_control_yaml,
)
from mr_storm.core.coordinator import MRStormCoordinator
from mr_storm.ros2.distributed_arm_node import load_fleet_arm_configs


def main():
    parser = argparse.ArgumentParser(description="Run MR-STORM with Gazebo via ROS 2.")
    parser.add_argument(
        "--fleet_config",
        type=str,
        default="mr_storm/ros2/config/quad_ur5e_fleet.yaml",
        help="Path to the multi-arm fleet YAML configuration.",
    )
    parser.add_argument(
        "--command_interface",
        type=str,
        default="trajectory",
        choices=["trajectory", "position", "velocity"],
        help="Gazebo ros2_control controller interface.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of MPC control steps to execute.",
    )
    parser.add_argument(
        "--print_ros2_control_yaml",
        action="store_true",
        help="Print a ready-to-use Gazebo ros2_control YAML config and exit.",
    )
    args = parser.parse_args()

    arm_cfgs = load_fleet_arm_configs(args.fleet_config)

    if args.print_ros2_control_yaml:
        ur5e_joints = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        joint_map = {
            cfg.robot_id: [f"{cfg.namespace}_{jn}" for jn in ur5e_joints]
            for cfg in arm_cfgs
        }
        print(generate_gazebo_ros2_control_yaml(joint_map))
        return

    print(f"Initializing MR-STORM Coordinator for {len(arm_cfgs)} arms...")
    coordinator = MRStormCoordinator(arm_cfgs)

    arm_joint_names = {
        idx: agent.joint_names for idx, agent in enumerate(coordinator.agents)
    }
    arm_namespaces = {cfg.robot_id: cfg.namespace for cfg in arm_cfgs}

    backend = GazeboROS2Backend(
        arm_joint_names=arm_joint_names,
        arm_namespaces=arm_namespaces,
        command_interface=args.command_interface,
        control_freq_hz=30.0,
    )

    print("Waiting for /joint_states from Gazebo...")
    backend.wait_for_joint_states(timeout_sec=5.0)

    # Example crossing goals in the shared central workspace (world frame W)
    goals_W = {}
    for idx, cfg in enumerate(arm_cfgs):
        bx, by, bz = cfg.base_pose[:3]
        # Send each arm across the workspace center to test inter-arm collision avoidance
        target_pos = np.array([-bx * 0.8, -by * 0.8, bz + 0.45], dtype=np.float32)
        target_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # pointing down
        goals_W[idx] = (target_pos, target_quat)

    coordinator.set_goals_world(goals_W)
    print("Running decentralized MR-STORM control loop with Gazebo...")

    try:
        for step_idx in range(args.steps):
            joint_states = backend.get_joint_states()
            commands = coordinator.step(joint_states)
            backend.send_joint_commands(commands)
            backend.step()

            if step_idx % 30 == 0:
                errs = [f"Arm {i}: {cmd.ee_position_error:.3f}m" for i, cmd in commands.items()]
                print(f"[Step {step_idx:04d}] EE Goal Errors -> " + " | ".join(errs))
    except KeyboardInterrupt:
        print("\nStopping Gazebo MR-STORM controller...")
    finally:
        backend.shutdown()


if __name__ == "__main__":
    main()
