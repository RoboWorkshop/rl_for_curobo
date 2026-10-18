"""
Lightweight Kinematic Simulator Backend (`KinematicBackend`).

Enables running, testing, and benchmarking MR-STORM multi-arm trajectories
without launching Isaac Sim or Gazebo.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import time
import numpy as np

from mr_storm.backends.base import RobotBackend
from mr_storm.core.types import JointCommand, RobotJointState


class KinematicBackend(RobotBackend):
    """
    Pure NumPy kinematic integrator backend that tracks commanded joint states.
    Ideal for rapid prototyping, unit testing, CI pipelines, or driving external
    visualizers (e.g., MeshCat / RViz / Matplotlib).
    """

    def __init__(
        self,
        initial_states: Dict[int, RobotJointState],
        control_freq_hz: float = 30.0,
        realtime_sleep: bool = False,
    ):
        super().__init__(n_arms=len(initial_states), control_freq_hz=control_freq_hz)
        self._states: Dict[int, RobotJointState] = {
            arm_id: RobotJointState(
                positions=st.positions.copy(),
                velocities=st.velocities.copy(),
                accelerations=st.accelerations.copy() if st.accelerations is not None else None,
                joint_names=list(st.joint_names) if st.joint_names else None,
            )
            for arm_id, st in initial_states.items()
        }
        self.realtime_sleep = realtime_sleep

    def get_joint_states(self) -> Dict[int, RobotJointState]:
        now = time.time()
        for st in self._states.values():
            st.timestamp = now
        return self._states

    def send_joint_commands(self, commands: Dict[int, JointCommand]) -> None:
        for arm_id, cmd in commands.items():
            self._states[arm_id] = RobotJointState(
                positions=cmd.positions.copy(),
                velocities=cmd.velocities.copy(),
                accelerations=cmd.accelerations.copy(),
                joint_names=list(cmd.joint_names),
                timestamp=time.time(),
            )

    def step(self) -> None:
        if self.realtime_sleep:
            super().step()
