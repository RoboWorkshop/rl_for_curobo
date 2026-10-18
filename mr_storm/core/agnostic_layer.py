"""
Simulator- and Hardware-Agnostic Execution Layer for MR-STORM (`MRStormAgnosticLayer`).

Loyal to the original control loop in `projects_root/experiments/core_api/benchmark_sim.py`
(lines 3631-3759 and 2607-2741):
- Takes observations (`RobotObservation`: joint positions, velocities, controlled DOF names,
  world-frame goal poses, and optional obstacle poses) from any simulator (Isaac Sim,
  Gazebo, MuJoCo, Kinematic) or physical robot hardware.
- Returns actions (`EmbodimentAction`: ordered joint position/velocity/acceleration commands
  and raw CuRobo `JointState`) ready to command any robot embodiment.
- Supports both execution modes in both simulation and real world:
  1. `synchronous-decentralized` (`sync`): Robots communicate and compute paths decentrally,
     planning one after the next in a pseudo-distributed manner.
  2. `asynchronous-decentralized` (`async`): Robots run asynchronously and communicate
     their horizon sphere plans and goal errors over ROS (`ROSPlanBoardComm`) or another
     `PlanCommBackend`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState

from projects_root.projects.dynamic_obs.dynamic_obs_predictor.runtime_topics import get_topics

from mr_storm.comm.plan_comm import LocalPlanBoardComm, PlanCommBackend, ROSPlanBoardComm
from mr_storm.core.planners import (
    CuAgent,
    CumotionPlanner,
    JointsStateWrap,
    MpcPlanner,
    build_cu_agents_from_meta_cfg,
    get_per_axis_euler_error,
)


@dataclass
class RobotObservation:
    """
    Simulator- and hardware-agnostic observation for a single robot `robot_idx`.
    """

    dof_names: List[str]
    positions: np.ndarray
    velocities: np.ndarray
    goals_W: Dict[str, Pose]
    prev_positions: Optional[np.ndarray] = None
    prev_velocities: Optional[np.ndarray] = None
    obstacle_poses_W: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
    link_name_to_pose_W: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
    dof_index_fn: Optional[Callable[[str], int]] = None


@dataclass
class EmbodimentAction:
    """
    Simulator- and hardware-agnostic action returned for a single robot `robot_idx`.
    Can be applied directly to any embodiment (UR5e, UR10e, Franka, Unitree G1, etc.)
    in Isaac Sim, Gazebo, MuJoCo, or real hardware.
    """

    robot_idx: int
    joint_names: List[str]
    joint_indices: List[int]
    positions: np.ndarray
    velocities: Optional[np.ndarray]
    accelerations: Optional[np.ndarray]
    cu_action: JointState
    mean_goal_err: Optional[Tuple[float, float]] = None
    plan: Optional[dict] = None
    solve_time_sec: float = 0.0
    extra_debug: Dict[str, Any] = field(default_factory=dict)


class MRStormAgnosticLayer:
    """
    Dedicated simulator- and hardware-agnostic layer under the hood of `main.py`.

    Wraps the exact original `CuAgent`, `MpcPlanner`, `CumotionPlanner`, `PlanPubSub`,
    and `WorldModelWrapper` classes from `benchmark_sim.py` and pairs them with a
    pluggable communication backend (`LocalPlanBoardComm` or `ROSPlanBoardComm`).
    """

    def __init__(
        self,
        meta_cfg: dict,
        mode: str = "synchronous-decentralized",
        comm_backend: Optional[PlanCommBackend] = None,
        comm_type: str = "auto",
        active_agent_indices: Optional[List[int]] = None,
        world_cfg: Optional[WorldConfig] = None,
        tensor_args: Optional[TensorDeviceType] = None,
        ros_topic_prefix: str = "/mr_storm",
    ):
        """
        Args:
            meta_cfg: Parsed MR-STORM meta configuration dictionary (same schema as `benchmark_sim.py`).
            mode: Either `'synchronous-decentralized'` (`'sync'`) or `'asynchronous-decentralized'` (`'async'`).
            comm_backend: Optional pre-constructed `PlanCommBackend`.
            comm_type: `'local'`, `'ros'`, or `'auto'` (defaults to `'local'` for sync and `'ros'` for async).
            active_agent_indices: Indices of robots controlled by this process (`None` = all robots `0..N-1`).
            world_cfg: Optional initial static `WorldConfig`.
            tensor_args: Optional `TensorDeviceType`.
            ros_topic_prefix: Topic prefix when communicating over ROS (default: `'/mr_storm'`).
        """
        if mode in ("sync", "synchronous", "synchronous-decentralized"):
            self.mode = "synchronous-decentralized"
        elif mode in ("async", "asynchronous", "asynchronous-decentralized"):
            self.mode = "asynchronous-decentralized"
        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Choose 'synchronous-decentralized' ('sync') "
                "or 'asynchronous-decentralized' ('async')."
            )

        self.tensor_args = tensor_args or TensorDeviceType()
        self.meta_cfg = meta_cfg
        self.cu_agents, self.init_info = build_cu_agents_from_meta_cfg(
            meta_cfg=meta_cfg,
            tensor_args=self.tensor_args,
            active_agent_indices=active_agent_indices,
            world_cfg=world_cfg,
        )
        self.n_agents: int = self.init_info["n_agents"]
        self.agent_by_idx: Dict[int, CuAgent] = {a.idx: a for a in self.cu_agents}
        self._last_js_by_idx: Dict[int, JointsStateWrap] = {}

        if comm_backend is not None:
            self.comm = comm_backend
        else:
            resolved_comm = comm_type
            if resolved_comm == "auto":
                resolved_comm = (
                    "ros" if self.mode == "asynchronous-decentralized" else "local"
                )
            if resolved_comm == "ros":
                self.comm = ROSPlanBoardComm(
                    n_agents=self.n_agents,
                    tensor_args=self.tensor_args,
                    active_agent_indices=[a.idx for a in self.cu_agents],
                    topic_prefix=ros_topic_prefix,
                )
            elif resolved_comm == "local":
                self.comm = LocalPlanBoardComm(
                    n_agents=self.n_agents,
                    tensor_args=self.tensor_args,
                )
            else:
                raise ValueError(
                    f"Unsupported comm_type '{comm_type}'. Use 'local', 'ros', or 'auto'."
                )

    def get_retract_observations(self) -> Dict[int, RobotObservation]:
        """
        Create initial `RobotObservation` for each active `CuAgent` at its retract
        configuration and initial retract goals. Useful for initializing any simulator
        or dry-running the control loop.
        """
        obs_dict: Dict[int, RobotObservation] = {}
        for a in self.cu_agents:
            dof_names = list(a.planner.ordered_j_names)
            retract_pos = (
                a.planner.solver.rollout_fn.dynamics_model.retract_config.detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float32)
            )
            obs_dict[a.idx] = RobotObservation(
                dof_names=dof_names,
                positions=retract_pos.copy(),
                velocities=np.zeros_like(retract_pos),
                goals_W={k: v.clone() for k, v in a.planner.plan_goals.items()},
            )
        return obs_dict

    def _compute_link_poses_and_goal_errors(
        self,
        a: CuAgent,
        cu_js: JointState,
        goals_W: Dict[str, Pose],
        measured_link_poses_W: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Tuple[
        Dict[str, Tuple[float, float]],
        Dict[str, Tuple[np.ndarray, np.ndarray]],
        Dict[str, str],
        Dict[str, Tuple[np.ndarray, np.ndarray]],
    ]:
        """
        Compute per-link `(pos_err, rot_err)` and populate `link_name_to_pose`,
        `name_link_to_target`, and `target_name_to_pose` using the exact error
        definition from `SimTask._get_link_errors` in `benchmark_sim.py` (lines 504-517).

        If `measured_link_poses_W` is not provided by the simulator, uses
        `a.planner.get_state_in_task_space(cu_js, frame='W')` (GPU forward kinematics).
        """
        if measured_link_poses_W is None:
            ts_W = a.planner.get_state_in_task_space(cu_js, frame="W")
            p_ee_W = ts_W["ee"]["p"].squeeze(0).detach().cpu().numpy().reshape(3)
            q_ee_W = ts_W["ee"]["q"].squeeze(0).detach().cpu().numpy().reshape(4)
            measured_link_poses_W = {a.planner.ee_link_name: (p_ee_W, q_ee_W)}
            if a.planner.constrained_links_names and "links" in ts_W:
                crm_link_names = list(a.planner.crm.kinematics_config.link_names)
                p_links_W = ts_W["links"]["p"].squeeze(0).detach().cpu().numpy()
                q_links_W = ts_W["links"]["q"].squeeze(0).detach().cpu().numpy()
                for l_idx, l_name in enumerate(crm_link_names):
                    measured_link_poses_W[l_name] = (
                        p_links_W[l_idx].reshape(3),
                        q_links_W[l_idx].reshape(4),
                    )

        link_errors: Dict[str, Tuple[float, float]] = {}
        link_name_to_pose: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        name_link_to_target: Dict[str, str] = {}
        target_name_to_pose: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for link_name, goal_pose in goals_W.items():
            if link_name in measured_link_poses_W:
                p_link, q_link = measured_link_poses_W[link_name]
            else:
                p_link, q_link = measured_link_poses_W[a.planner.ee_link_name]
            p_target = goal_pose.position.flatten().detach().cpu().numpy().reshape(3)
            q_target = goal_pose.quaternion.flatten().detach().cpu().numpy().reshape(4)

            target_name = f"target_{a.idx}_{link_name}"
            link_name_to_pose[link_name] = (p_link, q_link)
            name_link_to_target[link_name] = target_name
            target_name_to_pose[target_name] = (p_target, q_target)

            pos_err = float(np.linalg.norm(p_target - p_link[:3]))
            rot_err = float(np.linalg.norm(get_per_axis_euler_error(list(q_link), list(q_target))))
            link_errors[link_name] = (pos_err, rot_err)

        return link_errors, link_name_to_pose, name_link_to_target, target_name_to_pose

    def step_single_agent(
        self,
        agent_idx: int,
        obs: RobotObservation,
        t: int,
    ) -> Optional[EmbodimentAction]:
        """
        Execute one complete decentralized Sense -> Publish Plan -> Update ColPred -> Plan -> Act
        iteration for `agent_idx`, following lines 3645-3759 of `benchmark_sim.py` verbatim.
        """
        a = self.agent_by_idx[agent_idx]
        planner = a.planner
        psw = a.planning_stopwatch
        ctrl_dof_names = obs.dof_names

        # Current joint state wrapped in JointsStateWrap (same interface as isaac_JointsState)
        js_curr = JointsStateWrap(
            positions=np.asarray(obs.positions, dtype=np.float32),
            velocities=np.asarray(obs.velocities, dtype=np.float32),
        )
        # Previous step's joint state (`get_js(sync_new=False)` in benchmark_sim.py)
        if obs.prev_positions is not None and obs.prev_velocities is not None:
            js_prev = JointsStateWrap(
                positions=np.asarray(obs.prev_positions, dtype=np.float32),
                velocities=np.asarray(obs.prev_velocities, dtype=np.float32),
            )
        else:
            js_prev = self._last_js_by_idx.get(agent_idx, js_curr)

        # 1) Publish estimated plan from last step's policy means (lines 3654-3669 of benchmark_sim.py)
        plan = None
        agent_decentralized = a.plan_pub_sub is not None
        if agent_decentralized:
            if js_prev is not None and a.plan_pub_sub.should_pub_now(t):
                share_full_plan = a.is_full_plan_publisher()
                psw.on()
                plan = planner.get_estimated_plan(
                    ctrl_dof_names,
                    a.plan_pub_sub.valid_spheres,
                    js_prev,
                    valid_spheres_only=False,
                    naive=not share_full_plan,
                )
                psw.off()
                if plan is not None:
                    self.comm.publish_plan(a.idx, plan)

        # 2) Sense obstacles (lines 3673-3680 of benchmark_sim.py)
        if obs.obstacle_poses_W is not None:
            a.update_col_model_from_pose_dict(obs.obstacle_poses_W)

        # 3) Prepare current CuRobo JointState and update planner state (lines 3688-3695 of benchmark_sim.py)
        self._last_js_by_idx[agent_idx] = js_curr
        psw.on()
        pos_t = self.tensor_args.to_device(torch.from_numpy(js_curr.positions))
        vel_t = self.tensor_args.to_device(torch.from_numpy(js_curr.velocities))
        _0 = pos_t * 0.0
        cu_js = JointState(
            pos_t,
            vel_t,
            _0,
            ctrl_dof_names,
            _0,
        ).get_ordered_joint_state(planner.ordered_j_names)
        if isinstance(planner, MpcPlanner):
            planner.update_state(cu_js)
        psw.off()

        # 4) Compute goal errors for pose_wta conflict resolution & update robot_context (lines 3705-3734)
        (
            l_name_to_goal_err,
            link_name_to_pose,
            name_link_to_target,
            target_name_to_pose,
        ) = self._compute_link_poses_and_goal_errors(
            a, cu_js, obs.goals_W, obs.link_name_to_pose_W
        )

        agent_mean_err: Optional[Tuple[float, float]] = None
        if a.is_full_plan_publisher() and len(l_name_to_goal_err) > 0:
            psw.on()
            a_n_links = len(l_name_to_goal_err)
            p_err, q_err = 0.0, 0.0
            for p_err_link, q_err_link in l_name_to_goal_err.values():
                p_err += p_err_link
                q_err += q_err_link
            agent_mean_err = (p_err / a_n_links, q_err / a_n_links)
            self.comm.publish_goal_error(a.idx, agent_mean_err)
            psw.off()

        # 5) Sense plans from peer robots and update DynamicObsCollPredictor (lines 3720-3725)
        if a.plan_pub_sub is not None:
            psw.on()
            plans_board, mean_goal_err = self.comm.get_boards()
            a.update_col_pred(plans_board, mean_goal_err)
            psw.off()

        # Update runtime_topics context (lines 3730-3733)
        topics = get_topics()
        if topics is not None:
            robot_context = topics.get_default_env()[a.idx]
            robot_context["link_name_to_pose"] = link_name_to_pose
            robot_context["name_link_to_target"] = name_link_to_target
            robot_context["target_name_to_pose"] = target_name_to_pose

        # 6) Plan / yield action (lines 3738-3752)
        goals = obs.goals_W
        if isinstance(planner, CumotionPlanner):
            psw.on()
            action = planner.yield_action(goals, cu_js, js_curr.velocities)
            solve_elapsed = psw.off()
        elif isinstance(planner, MpcPlanner):
            psw.on()
            action = planner.yield_action(goals)
            solve_elapsed = psw.off()
        else:
            raise ValueError(f"Invalid planner type: {type(planner)}")

        if action is None:
            return None

        # 7) Convert action to embodiment joint ordering (lines 3754-3758)
        converted = planner.convert_action_to_embodiment(
            action, ctrl_dof_names, obs.dof_index_fn
        )
        a.step_count += 1

        return EmbodimentAction(
            robot_idx=a.idx,
            joint_names=converted["joint_names"],
            joint_indices=converted["joint_indices"],
            positions=converted["positions"],
            velocities=converted["velocities"],
            accelerations=converted["accelerations"],
            cu_action=action,
            mean_goal_err=agent_mean_err,
            plan=plan,
            solve_time_sec=solve_elapsed,
        )

    def step_synchronous_decentralized(
        self,
        observations: Dict[int, RobotObservation],
        t: int,
    ) -> Dict[int, Optional[EmbodimentAction]]:
        """
        Execute one round-robin step of `synchronous-decentralized` mode across all
        robots (`for a_idx, a in enumerate(cu_agents): ...`), planning one robot
        after the next in a pseudo-distributed manner — identical to `benchmark_sim.py`
        lines 3631-3759.
        """
        actions: Dict[int, Optional[EmbodimentAction]] = {}
        for a in self.cu_agents:
            if a.idx in observations:
                actions[a.idx] = self.step_single_agent(a.idx, observations[a.idx], t)
        return actions

    def close(self) -> None:
        self.comm.close()
