"""
Abstract Backend Interface (`RobotBackend`) for connecting MR-STORM to any
simulator (Gazebo, Isaac Sim, MuJoCo, PyBullet) or physical robot hardware.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import time

from mr_storm.core.types import JointCommand, RobotJointState


class RobotBackend(ABC):
    """
    Base interface for any multi-arm simulator or real-robot hardware driver.

    To connect MR-STORM to a new simulator or custom robot SDK, subclass
    `RobotBackend` and implement `get_joint_states` and `send_joint_commands`.
    """

    def __init__(self, n_arms: int, control_freq_hz: float = 30.0):
        self.n_arms = n_arms
        self.control_freq_hz = control_freq_hz
        self.dt = 1.0 / max(control_freq_hz, 1.0)
        self._last_step_time: float = 0.0

    @abstractmethod
    def get_joint_states(self) -> Dict[int, RobotJointState]:
        """Return the latest `RobotJointState` for each arm `0 .. n_arms-1`."""

    @abstractmethod
    def send_joint_commands(self, commands: Dict[int, JointCommand]) -> None:
        """Send target `JointCommand` to each arm `0 .. n_arms-1`."""

    def step(self) -> None:
        """
        Advance the simulation or sleep to maintain `control_freq_hz` on real hardware.
        """
        now = time.perf_counter()
        if self._last_step_time > 0.0:
            elapsed = now - self._last_step_time
            remaining = self.dt - elapsed
            if remaining > 0.0:
                time.sleep(remaining)
        self._last_step_time = time.perf_counter()

    def reset(self, initial_positions: Optional[Dict[int, List[float]]] = None) -> Dict[int, RobotJointState]:
        """Reset arms to initial joint configurations and return their states."""
        return self.get_joint_states()

    def shutdown(self) -> None:
        """Cleanly release simulator or hardware resources."""
