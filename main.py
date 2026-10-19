#!/usr/bin/env python3
"""
MR-STORM Unified Entry Point (`main.py`)
IEEE/RSJ IROS 2026 — "MR-STORM: Scalable Multi-Arm Control By Distributed MPC"

Loyal to the original `projects_root/experiments/core_api/benchmark_sim.py` algorithm
(`CuAgent`, `MpcPlanner`, `CumotionPlanner`, `PlanPubSub`, `WorldModelWrapper`,
`get_estimated_plan`, `update_col_pred`) while providing a dedicated simulator- and
hardware-agnostic layer (`MRStormAgnosticLayer`) under the hood.

Supported Execution Modes (for BOTH simulation and real robots):
1. `synchronous-decentralized` (`--mode sync`):
   Robots communicate and compute paths decentrally (`get_estimated_plan` -> `plans_board`
   -> `update_col_pred` -> `yield_action`), planning one after the next in a
   pseudo-distributed round-robin loop (identical to `benchmark_sim.py`).
2. `asynchronous-decentralized` (`--mode async`):
   Robots plan asynchronously without waiting for each other and communicate their
   horizon collision sphere plans and goal errors over ROS (`ROSPlanBoardComm`,
   `--comm ros`) or local memory (`--comm local`). Can run either as one OS process
   per robot (`--robot_id 0`, `--robot_id 1`, ...) or as parallel async agent threads.

Supported Environments / Embodiments (`--backend`):
- `kinematic`: Pure-GPU kinematic simulation (runs out of the box without Isaac/Gazebo)
- `gazebo`:    Gazebo (Harmonic / Classic) via `ros2_control` (`GazeboROSAdapter`)
- `real`:      Physical multi-arm hardware with `SafetySupervisor` (`RealHardwareEnvAdapter`)
- `isaac`:     NVIDIA Isaac Sim (`benchmark_sim.py` stage & `ArticulationAction`)
"""

from __future__ import annotations

import argparse
import os
import signal
from threading import Event, Thread
import time
from typing import Dict, List, Optional

import numpy as np

from curobo.util.logger import setup_curobo_logger
from curobo.util_file import load_yaml

from mr_storm.backends.env_adapters import (
    BaseEnvAdapter,
    GazeboROSAdapter,
    KinematicEnvAdapter,
    RealHardwareEnvAdapter,
)
from mr_storm.backends.real_robot_backend import (
    CallbackHardwareDriver,
    SafetyLimits,
    SingleArmHardwareDriver,
    URRTDEHardwareDriver,
    UnitreeG1SocketDriver,
)
from mr_storm.core.agnostic_layer import MRStormAgnosticLayer
from mr_storm.core.planners import resolve_repo_path


DEFAULT_META_CFG = (
    "projects_root/experiments/core_api/meta_conf_variants/meta_cfg_arms_50steps.yml"
)


def build_env_adapter(
    backend: str,
    layer: MRStormAgnosticLayer,
    ur_ips: Optional[List[str]] = None,
    control_freq_hz: float = 33.3,
) -> BaseEnvAdapter:
    """
    Instantiate the requested simulator or real-hardware adapter on top of `MRStormAgnosticLayer`.
    """
    if backend == "kinematic":
        return KinematicEnvAdapter(layer, auto_swap_goals=True)
    elif backend == "gazebo":
        return GazeboROSAdapter(
            layer,
            namespace_prefix="arm",
            controller_name="joint_trajectory_controller",
            command_interface="trajectory",
            step_dt=1.0 / max(control_freq_hz, 1.0),
        )
    elif backend in ("real", "g1_real"):
        drivers: Dict[int, SingleArmHardwareDriver] = {}
        for idx, a in enumerate(layer.cu_agents):
            dof_names = list(a.planner.ordered_j_names)
            retract = (
                a.planner.solver.rollout_fn.dynamics_model.retract_config.detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float32)
            )
            robot_cfg_name = str(a.a_cfg.get("robot", ""))
            is_g1_arm = backend == "g1_real" or ("g1" in robot_cfg_name.lower() and not ur_ips)
            if is_g1_arm and a.idx in (0, 1):
                drivers[a.idx] = UnitreeG1SocketDriver(
                    arm_idx=a.idx,
                    joint_names=dof_names,
                    fallback_retract=retract,
                )
            elif ur_ips and idx < len(ur_ips) and ur_ips[idx]:
                drivers[a.idx] = URRTDEHardwareDriver(
                    robot_id=a.idx,
                    robot_ip=ur_ips[idx],
                    joint_names=dof_names,
                    control_dt=1.0 / max(control_freq_hz, 1.0),
                )
            else:
                # Default safe callback driver template (replace with your robot SDK or pass --ur_ips)
                state_buf = {"pos": retract.copy(), "vel": np.zeros_like(retract)}

                def _read_fn(buf=state_buf):
                    return buf["pos"].copy(), buf["vel"].copy()

                def _write_fn(pos, vel, acc=None, buf=state_buf):
                    buf["vel"] = (
                        vel.copy()
                        if vel is not None
                        else (pos - buf["pos"]) * control_freq_hz
                    )
                    buf["pos"] = pos.copy()

                drivers[a.idx] = CallbackHardwareDriver(
                    robot_id=a.idx,
                    joint_names=dof_names,
                    read_fn=_read_fn,
                    write_fn=_write_fn,
                )
        return RealHardwareEnvAdapter(
            layer,
            drivers=drivers,
            safety_limits=SafetyLimits(),
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def run_synchronous_decentralized(
    layer: MRStormAgnosticLayer,
    adapter: BaseEnvAdapter,
    max_steps: int = 500,
    log_every: int = 10,
) -> None:
    """
    Mode 1: `synchronous-decentralized`
    Robots communicate via `plans_board` / `mean_goal_err` and compute paths decentrally,
    planning one after the next in a pseudo-distributed loop — identical to
    `benchmark_sim.py` lines 3614-3878.
    """
    stop_flag = False

    def _sig_handler(_sig, _frame):
        nonlocal stop_flag
        stop_flag = True

    signal.signal(signal.SIGINT, _sig_handler)

    print(
        f"\n[MR-STORM] Starting SYNCHRONOUS-DECENTRALIZED control loop "
        f"({len(layer.cu_agents)} active robots, comm={type(layer.comm).__name__}, "
        f"adapter={type(adapter).__name__})"
    )

    for t in range(max_steps):
        if stop_flag or not adapter.is_running():
            break

        # 1. Step environment / task
        adapter.step_environment()

        # 2. Sense observations for all robots
        observations = adapter.get_observations()

        # 3. Pseudo-distributed sequential planning across `cu_agents`:
        #    get_estimated_plan -> plans_board -> update_col_pred -> yield_action
        actions = layer.step_synchronous_decentralized(observations, t)

        # 4. Command embodiments
        adapter.apply_actions(actions)

        if log_every > 0 and t % log_every == 0:
            err_strs = []
            for r_idx, act in actions.items():
                if act is not None and act.mean_goal_err is not None:
                    p_e, q_e = act.mean_goal_err
                    err_strs.append(
                        f"R{r_idx}: pos_err={p_e:.3f}m, rot_err={q_e:.1f}deg ({act.solve_time_sec*1000:.1f}ms)"
                    )
            print(f"[Sync Step {t:04d}] " + " | ".join(err_strs))


def run_asynchronous_decentralized(
    layer: MRStormAgnosticLayer,
    adapter: BaseEnvAdapter,
    max_steps: int = 500,
    control_freq_hz: float = 33.3,
    log_every: int = 10,
) -> None:
    """
    Mode 2: `asynchronous-decentralized`
    Each robot runs its own independent asynchronous control loop (either in its own
    OS process via `--robot_id <i>` or its own independent worker thread when running
    multiple robots) and communicates its horizon sphere plans (`plan['task_space']['spheres']`)
    and `mean_goal_err` over ROS (`ROSPlanBoardComm`).
    """
    stop_event = Event()

    def _sig_handler(_sig, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _sig_handler)

    dt_target = 1.0 / max(control_freq_hz, 1.0)
    print(
        f"\n[MR-STORM] Starting ASYNCHRONOUS-DECENTRALIZED control loop "
        f"(active_robots={[a.idx for a in layer.cu_agents]}, comm={type(layer.comm).__name__}, "
        f"adapter={type(adapter).__name__}, rate={control_freq_hz:.1f}Hz)"
    )

    def _agent_async_loop(robot_idx: int):
        t_local = 0
        while not stop_event.is_set() and (max_steps <= 0 or t_local < max_steps):
            t_start = time.perf_counter()
            obs = adapter.get_robot_observation(robot_idx)
            act = layer.step_single_agent(robot_idx, obs, t_local)
            if act is not None:
                adapter.apply_robot_action(robot_idx, act)
                if log_every > 0 and t_local % log_every == 0 and act.mean_goal_err is not None:
                    p_e, q_e = act.mean_goal_err
                    print(
                        f"[Async R{robot_idx} | step={t_local:04d}] "
                        f"pos_err={p_e:.3f}m, rot_err={q_e:.1f}deg, "
                        f"solve={act.solve_time_sec*1000:.1f}ms"
                    )
            t_local += 1
            elapsed = time.perf_counter() - t_start
            if elapsed < dt_target:
                time.sleep(dt_target - elapsed)

    if len(layer.cu_agents) == 1:
        # Single robot process (e.g., distributed ROS node `--robot_id i`)
        _agent_async_loop(layer.cu_agents[0].idx)
    else:
        # Multi-robot asynchronous threads communicating via `layer.comm` (ROS or local)
        threads = [
            Thread(target=_agent_async_loop, args=(a.idx,), daemon=True)
            for a in layer.cu_agents
        ]
        for th in threads:
            th.start()
        try:
            while any(th.is_alive() for th in threads) and not stop_event.is_set():
                adapter.step_environment()
                time.sleep(0.01)
        finally:
            stop_event.set()
            for th in threads:
                th.join(timeout=1.0)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="MR-STORM Unified Simulator- & Hardware-Agnostic Main Script"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=DEFAULT_META_CFG,
        help="Path to MR-STORM meta configuration YAML (default: meta_cfg_arms_50steps.yml).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sync",
        choices=[
            "sync",
            "async",
            "synchronous-decentralized",
            "asynchronous-decentralized",
        ],
        help=(
            "Execution mode: 'sync' ('synchronous-decentralized', pseudo-distributed sequential "
            "planning) or 'async' ('asynchronous-decentralized', independent async planning "
            "with inter-robot communication over ROS)."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="kinematic",
        choices=["kinematic", "gazebo", "real", "g1_real", "isaac"],
        help="Simulator or hardware backend ('kinematic', 'gazebo', 'real', 'g1_real', 'isaac').",
    )
    parser.add_argument(
        "--comm",
        type=str,
        default="auto",
        choices=["auto", "local", "ros"],
        help=(
            "Inter-robot communication backend for plans_board and mean_goal_err: "
            "'local' (shared memory), 'ros' (ROS 2 / ROS 1 topics), or 'auto' "
            "('local' in sync mode, 'ros' in async mode)."
        ),
    )
    parser.add_argument(
        "--robot_id",
        type=str,
        default="all",
        help=(
            "Robot index to run in this process ('all' for all robots, or an integer '0', '1', ... "
            "when running each robot in its own distributed ROS process/machine)."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Maximum number of control steps per robot (default: 200; <=0 for infinite).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=33.3,
        help="Target control loop frequency in Hz for async/real/gazebo execution (default: 33.3).",
    )
    parser.add_argument(
        "--ur_ips",
        type=str,
        default="",
        help="Comma-separated UR robot IP addresses when using `--backend real` (e.g. '192.168.1.101,192.168.1.102').",
    )
    parser.add_argument(
        "--vis",
        type=str,
        default="headless",
        choices=["headless", "gui", "livestream"],
        help="Visualization mode when using `--backend isaac` (default: headless).",
    )
    args = parser.parse_args(argv)

    # If `--backend isaac` is selected, delegate to the full Isaac Sim stage runner in `benchmark_sim.py`
    if args.backend == "isaac":
        import multiprocessing as mp
        from projects_root.experiments.core_api.benchmark_sim import root as isaac_root

        cfg_path = resolve_repo_path(args.cfg)
        meta_cfg = load_yaml(cfg_path)
        meta_cfg["async"] = args.mode in ("async", "asynchronous-decentralized")
        if args.steps > 0:
            meta_cfg["termination"]["max_tsteps"] = args.steps
        out_dir = os.path.join("out_isaac", time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(out_dir, exist_ok=True)
        stop_ev = mp.Event()
        isaac_root(meta_cfg, out_dir, stop_ev, args.vis)
        return

    setup_curobo_logger("warn")
    cfg_path = resolve_repo_path(args.cfg)
    meta_cfg = load_yaml(cfg_path)

    active_indices: Optional[List[int]] = None
    if args.robot_id.lower() != "all":
        active_indices = [int(x.strip()) for x in args.robot_id.split(",") if x.strip()]

    layer = MRStormAgnosticLayer(
        meta_cfg=meta_cfg,
        mode=args.mode,
        comm_type=args.comm,
        active_agent_indices=active_indices,
    )

    ur_ip_list = [ip.strip() for ip in args.ur_ips.split(",") if ip.strip()] if args.ur_ips else None
    adapter = build_env_adapter(
        backend=args.backend,
        layer=layer,
        ur_ips=ur_ip_list,
        control_freq_hz=args.rate,
    )

    try:
        if layer.mode == "synchronous-decentralized":
            run_synchronous_decentralized(
                layer=layer,
                adapter=adapter,
                max_steps=args.steps,
            )
        else:
            run_asynchronous_decentralized(
                layer=layer,
                adapter=adapter,
                max_steps=args.steps,
                control_freq_hz=args.rate,
            )
    finally:
        adapter.close()
        layer.close()


if __name__ == "__main__":
    main()
