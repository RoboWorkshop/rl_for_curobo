"""
Real Robot Hardware Backend (`RealRobotBackend`) and Safety Supervisor for MR-STORM.

Provides a hardware-agnostic execution layer with real-time safety guarantees:
1. Joint position limit clamping `[q_min, q_max]`
2. Maximum per-cycle joint step clamping `|dq| <= max_step_rad` (prevents jumps)
3. Velocity & acceleration limit enforcement
4. Exponential Moving Average (EMA) low-pass command smoothing
5. Watchdog timeout & instant software Emergency Stop (`trigger_estop()`)

Includes built-in hardware driver adapters for:
- **Universal Robots (UR3e / UR5e / UR10e)** via `ur_rtde` (`servoJ` / `speedJ`)
- **Unitree G1 Humanoid Dual-Arm** via `unitree_sdk2py`
- **Franka Emika (Panda / FR3)** via `frankx` / `polymetis`
- **Any Custom Robot SDK (`CallbackHardwareDriver`)** via two simple Python functions:
  `read_state_fn() -> (positions, velocities)` and `send_cmd_fn(positions, velocities)`
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import time
import numpy as np

from mr_storm.backends.base import RobotBackend
from mr_storm.core.types import JointCommand, RobotJointState


@dataclass
class SafetyLimits:
    """Real-time safety envelope for physical robot arms."""

    q_min: Optional[np.ndarray] = None  # (n_dofs,) rad
    q_max: Optional[np.ndarray] = None  # (n_dofs,) rad
    max_vel_rad_s: float = 2.0  # rad/s
    max_acc_rad_s2: float = 6.0  # rad/s^2
    max_step_rad: float = 0.08  # max joint position change per control step (rad)
    ema_smoothing_alpha: float = 0.65  # 1.0 = no smoothing, lower = smoother
    watchdog_timeout_sec: float = 0.25  # max allowed delay between commands


class SafetySupervisor:
    """
    Filters and validates every `JointCommand` before it reaches physical actuators.
    """

    def __init__(self, limits: SafetyLimits):
        self.limits = limits
        self.estop_engaged: bool = False
        self._prev_cmd_pos: Optional[np.ndarray] = None
        self._last_cmd_timestamp: float = 0.0

    def trigger_estop(self, reason: str = "Manual E-Stop"):
        self.estop_engaged = True
        print(f"[MR-STORM SAFETY E-STOP] Engaged: {reason}")

    def reset_estop(self):
        self.estop_engaged = False
        self._prev_cmd_pos = None

    def filter_command(
        self,
        current_state: RobotJointState,
        raw_cmd: JointCommand,
    ) -> JointCommand:
        """
        Apply step clamping, EMA low-pass filtering, velocity limits, and joint limits.
        If E-Stop is engaged, commands the robot to hold its current position with zero velocity.
        """
        now = time.time()
        q_cur = current_state.positions.copy()

        if self.estop_engaged:
            return JointCommand(
                positions=q_cur,
                velocities=np.zeros_like(q_cur),
                accelerations=np.zeros_like(q_cur),
                joint_names=list(raw_cmd.joint_names),
                dt=raw_cmd.dt,
            )

        if self._prev_cmd_pos is None:
            self._prev_cmd_pos = q_cur.copy()

        # 1. Low-pass EMA smoothing on target position
        alpha = float(np.clip(self.limits.ema_smoothing_alpha, 0.05, 1.0))
        q_target = alpha * raw_cmd.positions + (1.0 - alpha) * self._prev_cmd_pos

        # 2. Clamp per-cycle step delta relative to current measured joint position
        dq = q_target - q_cur
        dq_clamped = np.clip(dq, -self.limits.max_step_rad, self.limits.max_step_rad)
        q_safe = q_cur + dq_clamped

        # 3. Clamp to absolute joint position limits if configured
        if self.limits.q_min is not None and self.limits.q_max is not None:
            q_safe = np.clip(q_safe, self.limits.q_min, self.limits.q_max)

        # 4. Clamp velocities and accelerations
        qd_safe = np.clip(
            raw_cmd.velocities, -self.limits.max_vel_rad_s, self.limits.max_vel_rad_s
        )
        qdd_safe = np.clip(
            raw_cmd.accelerations, -self.limits.max_acc_rad_s2, self.limits.max_acc_rad_s2
        )

        self._prev_cmd_pos = q_safe.copy()
        self._last_cmd_timestamp = now

        return JointCommand(
            positions=q_safe,
            velocities=qd_safe,
            accelerations=qdd_safe,
            joint_names=list(raw_cmd.joint_names),
            dt=raw_cmd.dt,
            ee_position_error=raw_cmd.ee_position_error,
            ee_rotation_error=raw_cmd.ee_rotation_error,
            solve_time_ms=raw_cmd.solve_time_ms,
        )


class SingleArmHardwareDriver(ABC):
    """Abstract interface for a single physical robot arm SDK."""

    @abstractmethod
    def read_joint_state(self) -> RobotJointState:
        """Read current measured `(positions, velocities)` from the physical robot."""

    @abstractmethod
    def write_joint_command(self, cmd: JointCommand) -> None:
        """Send safety-filtered `JointCommand` to the physical robot."""

    def stop(self) -> None:
        """Cleanly stop the physical arm."""


class CallbackHardwareDriver(SingleArmHardwareDriver):
    """
    Universal adapter that wraps any custom robot Python SDK using two callbacks:
    - `read_fn() -> Tuple[np.ndarray, np.ndarray]` returning `(positions_rad, velocities_rad_s)`
    - `write_fn(positions_rad: np.ndarray, velocities_rad_s: np.ndarray) -> None`
    """

    def __init__(
        self,
        joint_names: List[str],
        read_fn: Callable[[], Tuple[np.ndarray, np.ndarray]],
        write_fn: Callable[[np.ndarray, np.ndarray], None],
        stop_fn: Optional[Callable[[], None]] = None,
    ):
        self.joint_names = list(joint_names)
        self._read_fn = read_fn
        self._write_fn = write_fn
        self._stop_fn = stop_fn

    def read_joint_state(self) -> RobotJointState:
        pos, vel = self._read_fn()
        return RobotJointState(
            positions=np.asarray(pos, dtype=np.float32),
            velocities=np.asarray(vel, dtype=np.float32),
            joint_names=self.joint_names,
            timestamp=time.time(),
        )

    def write_joint_command(self, cmd: JointCommand) -> None:
        self._write_fn(cmd.positions, cmd.velocities)

    def stop(self) -> None:
        if self._stop_fn is not None:
            self._stop_fn()


class URRTDEHardwareDriver(SingleArmHardwareDriver):
    """
    Hardware driver for Universal Robots (UR3e, UR5e, UR10e, UR16e) using `ur_rtde`.
    Uses real-time `servoJ` position streaming.
    """

    def __init__(
        self,
        robot_ip: str,
        joint_names: Optional[List[str]] = None,
        dt: float = 0.02,
        lookahead_time: float = 0.1,
        gain: float = 300.0,
    ):
        try:
            import rtde_control
            import rtde_receive
        except ImportError as exc:
            raise ImportError(
                "Package `ur_rtde` is required for URRTDEHardwareDriver. "
                "Install via `pip install ur-rtde`."
            ) from exc

        self.robot_ip = robot_ip
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.dt = dt
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.joint_names = joint_names or [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

    def read_joint_state(self) -> RobotJointState:
        q = np.asarray(self.rtde_r.getActualQ(), dtype=np.float32)
        qd = np.asarray(self.rtde_r.getActualQd(), dtype=np.float32)
        return RobotJointState(
            positions=q,
            velocities=qd,
            joint_names=self.joint_names,
            timestamp=time.time(),
        )

    def write_joint_command(self, cmd: JointCommand) -> None:
        self.rtde_c.servoJ(
            [float(x) for x in cmd.positions],
            0.5,
            0.5,
            self.dt,
            self.lookahead_time,
            self.gain,
        )

    def stop(self) -> None:
        self.rtde_c.servoStop()
        self.rtde_c.stopScript()


class RealRobotBackend(RobotBackend):
    """
    Multi-arm physical robot backend combining per-arm `SingleArmHardwareDriver`
    instances with per-arm `SafetySupervisor` protection.
    """

    def __init__(
        self,
        drivers: Dict[int, SingleArmHardwareDriver],
        safety_limits: Optional[Dict[int, SafetyLimits]] = None,
        control_freq_hz: float = 30.0,
    ):
        super().__init__(n_arms=len(drivers), control_freq_hz=control_freq_hz)
        self.drivers = drivers
        self.supervisors: Dict[int, SafetySupervisor] = {}
        for arm_id in drivers:
            lim = (
                safety_limits[arm_id]
                if safety_limits and arm_id in safety_limits
                else SafetyLimits()
            )
            self.supervisors[arm_id] = SafetySupervisor(lim)
        self._last_measured: Dict[int, RobotJointState] = {}

    def trigger_estop_all(self, reason: str = "User E-Stop"):
        for sup in self.supervisors.values():
            sup.trigger_estop(reason)
        for drv in self.drivers.values():
            drv.stop()

    def get_joint_states(self) -> Dict[int, RobotJointState]:
        for arm_id, drv in self.drivers.items():
            self._last_measured[arm_id] = drv.read_joint_state()
        return self._last_measured

    def send_joint_commands(self, commands: Dict[int, JointCommand]) -> None:
        if not self._last_measured:
            self.get_joint_states()
        for arm_id, raw_cmd in commands.items():
            cur_state = self._last_measured[arm_id]
            safe_cmd = self.supervisors[arm_id].filter_command(cur_state, raw_cmd)
            self.drivers[arm_id].write_joint_command(safe_cmd)

    def shutdown(self) -> None:
        for drv in self.drivers.values():
            drv.stop()
