#!/usr/bin/env python3
"""
Example 2: Multi-Arm MR-STORM on Real Robot Hardware (`RealRobotBackend`).

Shows two ways to run MR-STORM on physical robotic arms:
1. Using the built-in Universal Robots RTDE driver (`URRTDEHardwareDriver`)
2. Using `CallbackHardwareDriver` to wrap ANY robot SDK (Unitree G1, Franka,
   Kinova, Dynamixel, CAN bus) with just two functions: `read_fn` and `write_fn`.

Includes real-time `SafetySupervisor` protection (per-step joint clamping,
EMA smoothing, velocity/acceleration limits, and instant E-Stop).
"""

from __future__ import annotations

import argparse
import numpy as np

from mr_storm.backends.real_robot_backend import (
    CallbackHardwareDriver,
    RealRobotBackend,
    SafetyLimits,
    URRTDEHardwareDriver,
)
from mr_storm.core.coordinator import MRStormCoordinator
from mr_storm.core.types import ArmConfig


def main():
    parser = argparse.ArgumentParser(description="Run MR-STORM on physical multi-arm hardware.")
    parser.add_argument(
        "--mode",
        type=str,
        default="custom_sdk_demo",
        choices=["ur_rtde", "custom_sdk_demo"],
        help="Use 'ur_rtde' for physical UR5e arms or 'custom_sdk_demo' for custom SDK callback template.",
    )
    parser.add_argument("--arm0_ip", type=str, default="192.168.1.101", help="IP of UR5e Arm 0")
    parser.add_argument("--arm1_ip", type=str, default="192.168.1.102", help="IP of UR5e Arm 1")
    parser.add_argument("--steps", type=int, default=200, help="Number of control steps")
    args = parser.parse_args()

    # 1. Define physical base poses of each arm in the shared world calibration frame
    arm_configs = [
        ArmConfig(
            robot_id=0,
            robot_yml="ur5e.yml",
            base_pose=[0.0, -0.25, 0.0, 1.0, 0.0, 0.0, 0.0],
        ),
        ArmConfig(
            robot_id=1,
            robot_yml="ur5e.yml",
            base_pose=[0.0, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0],
        ),
    ]

    coordinator = MRStormCoordinator(arm_configs)

    # 2. Configure hardware drivers & safety limits per arm
    if args.mode == "ur_rtde":
        drivers = {
            0: URRTDEHardwareDriver(robot_ip=args.arm0_ip, joint_names=coordinator.agents[0].joint_names),
            1: URRTDEHardwareDriver(robot_ip=args.arm1_ip, joint_names=coordinator.agents[1].joint_names),
        }
    else:
        # Template showing how to wrap ANY robot hardware SDK in 5 lines of code:
        hw_pos = {
            0: np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float32),
            1: np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float32),
        }
        hw_vel = {
            0: np.zeros(6, dtype=np.float32),
            1: np.zeros(6, dtype=np.float32),
        }

        def make_driver(arm_id: int):
            def read_from_sdk():
                # Replace with e.g. my_robot_sdk.get_joint_positions(), my_robot_sdk.get_joint_velocities()
                return hw_pos[arm_id], hw_vel[arm_id]

            def write_to_sdk(q_target: np.ndarray, qd_target: np.ndarray):
                # Replace with e.g. my_robot_sdk.set_servo_targets(q_target)
                hw_vel[arm_id] = qd_target.copy()
                hw_pos[arm_id] = q_target.copy()

            return CallbackHardwareDriver(
                joint_names=coordinator.agents[arm_id].joint_names,
                read_fn=read_from_sdk,
                write_fn=write_to_sdk,
            )

        drivers = {0: make_driver(0), 1: make_driver(1)}

    safety_limits = {
        0: SafetyLimits(max_vel_rad_s=1.5, max_step_rad=0.05, ema_smoothing_alpha=0.6),
        1: SafetyLimits(max_vel_rad_s=1.5, max_step_rad=0.05, ema_smoothing_alpha=0.6),
    }

    backend = RealRobotBackend(
        drivers=drivers,
        safety_limits=safety_limits,
        control_freq_hz=30.0,
    )

    # 3. Set crossing target poses in the shared world frame
    coordinator.set_goals_world(
        {
            0: (np.array([0.45, 0.15, 0.35], dtype=np.float32), np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)),
            1: (np.array([0.45, -0.15, 0.35], dtype=np.float32), np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)),
        }
    )

    print("Starting real-robot MR-STORM control loop (Ctrl+C triggers immediate E-Stop)...")
    try:
        for step_idx in range(args.steps):
            states = backend.get_joint_states()
            cmds = coordinator.step(states)
            backend.send_joint_commands(cmds)
            backend.step()

            if step_idx % 20 == 0:
                print(
                    f"[Step {step_idx:03d}] Arm 0 err: {cmds[0].ee_position_error:.3f}m | "
                    f"Arm 1 err: {cmds[1].ee_position_error:.3f}m | "
                    f"Solve: {cmds[0].solve_time_ms:.1f}ms"
                )
    except KeyboardInterrupt:
        backend.trigger_estop_all("Operator KeyboardInterrupt")
    finally:
        backend.shutdown()


if __name__ == "__main__":
    main()
