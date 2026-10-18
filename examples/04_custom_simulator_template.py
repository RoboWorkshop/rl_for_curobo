#!/usr/bin/env python3
"""
Example 4: Custom Simulator Adapter Template (MuJoCo / PyBullet / CoppeliaSim / Drake).

Demonstrates how to connect MR-STORM to ANY simulator by subclassing `RobotBackend`
in ~25 lines of code.
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np

from mr_storm.backends.base import RobotBackend
from mr_storm.core.coordinator import MRStormCoordinator
from mr_storm.core.types import ArmConfig, JointCommand, RobotJointState


class CustomSimulatorBackend(RobotBackend):
    """
    Minimal template for connecting MR-STORM to MuJoCo, PyBullet, or any custom simulator.
    """

    def __init__(self, arm_joint_names: Dict[int, List[str]], control_freq_hz: float = 30.0):
        super().__init__(n_arms=len(arm_joint_names), control_freq_hz=control_freq_hz)
        self.arm_joint_names = arm_joint_names
        # Initialize your simulator here (e.g. `self.mj_model = mujoco.MjModel.from_xml_path(...)`)
        self._q = {
            arm_id: np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float32)
            for arm_id in arm_joint_names
        }
        self._qd = {
            arm_id: np.zeros(len(names), dtype=np.float32)
            for arm_id, names in arm_joint_names.items()
        }

    def get_joint_states(self) -> Dict[int, RobotJointState]:
        # Read joint positions and velocities from your simulator (e.g. `mj_data.qpos`, `mj_data.qvel`)
        return {
            arm_id: RobotJointState(
                positions=self._q[arm_id].copy(),
                velocities=self._qd[arm_id].copy(),
                joint_names=self.arm_joint_names[arm_id],
            )
            for arm_id in self.arm_joint_names
        }

    def send_joint_commands(self, commands: Dict[int, JointCommand]) -> None:
        # Apply control commands to your simulator actuators (e.g. `mj_data.ctrl[:] = cmd.positions`)
        for arm_id, cmd in commands.items():
            self._q[arm_id] = cmd.positions.copy()
            self._qd[arm_id] = cmd.velocities.copy()

    def step(self) -> None:
        # Step your physics engine (e.g. `mujoco.mj_step(self.mj_model, self.mj_data)`)
        super().step()


if __name__ == "__main__":
    arm_configs = [
        ArmConfig(robot_id=0, robot_yml="ur5e.yml", base_pose=[-0.25, -0.25, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ArmConfig(robot_id=1, robot_yml="ur5e.yml", base_pose=[0.25, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0]),
    ]
    coordinator = MRStormCoordinator(arm_configs)
    backend = CustomSimulatorBackend(
        {i: agent.joint_names for i, agent in enumerate(coordinator.agents)}
    )
    coordinator.set_goals_world(
        {
            0: (np.array([0.20, 0.20, 0.40], dtype=np.float32), np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)),
            1: (np.array([-0.20, -0.20, 0.40], dtype=np.float32), np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)),
        }
    )
    for step_i in range(100):
        states = backend.get_joint_states()
        cmds = coordinator.step(states)
        backend.send_joint_commands(cmds)
        if step_i % 20 == 0:
            print(f"[Step {step_i:03d}] Arm 0 err: {cmds[0].ee_position_error:.3f}m | Arm 1 err: {cmds[1].ee_position_error:.3f}m")
