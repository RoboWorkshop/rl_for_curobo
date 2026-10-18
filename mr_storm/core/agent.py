"""
Simulator- and Hardware-Agnostic MR-STORM Agent.

Decouples the core MR-STORM algorithm (GPU MPPI trajectory optimization,
forward-integrated collision sphere rollout broadcasting, and dynamic priority
conflict resolution) from Isaac Sim so it can run with Gazebo, ROS 2, MuJoCo,
or directly on physical robot hardware.
"""

from __future__ import annotations

import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from mr_storm.core.types import (
    ArmConfig,
    BroadcastPlanPacket,
    JointCommand,
    RobotJointState,
)
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.frame_utils import FrameUtils

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(rel_or_abs: str) -> str:
    """Resolve a configuration path relative to the repository root if not already absolute."""
    p = Path(rel_or_abs)
    if p.is_Absolute() if hasattr(p, "is_Absolute") else p.is_absolute():
        return str(p)
    candidate = REPO_ROOT / p
    if candidate.exists():
        return str(candidate)
    # Check inside CuRobo robot configs
    curobo_robot_dir = REPO_ROOT / "curobo" / "src" / "curobo" / "content" / "configs" / "robot"
    if (curobo_robot_dir / p).exists():
        return str(curobo_robot_dir / p)
    return str(rel_or_abs)


def calculate_robot_sphere_count(robot_cfg: dict) -> Tuple[int, int]:
    """
    Calculate (base_sphere_count, extra_sphere_count) from a CuRobo robot configuration dict.
    """
    from curobo.util_file import load_yaml

    robots_spheres_dir = REPO_ROOT / "curobo" / "src" / "curobo" / "content" / "configs" / "robot"
    collision_spheres = robot_cfg["kinematics"]["collision_spheres"]
    if isinstance(collision_spheres, str):
        spheres_path = robots_spheres_dir / collision_spheres
        collision_spheres_dict = load_yaml(str(spheres_path))["collision_spheres"]
    else:
        collision_spheres_dict = collision_spheres

    sphere_count = sum(
        len(spheres) for spheres in collision_spheres_dict.values() if isinstance(spheres, list)
    )
    extra_spheres = robot_cfg["kinematics"].get("extra_collision_spheres", {})
    extra_sphere_count = sum(extra_spheres.values())
    return sphere_count, extra_sphere_count


class MRStormAgent:
    """
    Single decentralized MR-STORM arm controller.

    Can be instantiated either:
    1. Inside a multi-arm single-process fleet via `MRStormCoordinator`, OR
    2. Standalone inside an independent ROS 2 node / hardware control process
       (using `MRStormAgent.create_distributed(...)`) where each arm runs in its
       own process/machine and exchanges `BroadcastPlanPacket` messages.
    """

    def __init__(
        self,
        robot_id: int,
        base_pose: List[float],
        robot_cfg: dict,
        particle_file_path: str,
        world_cfg: Optional[object] = None,
        solver_config_overrides: Optional[dict] = None,
        peer_robot_ids: Optional[List[int]] = None,
    ):
        from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
        from curobo.geom.types import WorldConfig
        from curobo.rollout.cost.custom.arm_base.dynamic_obs_cost import DynamicObsCost
        from curobo.rollout.rollout_base import Goal
        from curobo.types.base import TensorDeviceType
        from curobo.types.math import Pose
        from curobo.types.robot import RobotConfig
        from curobo.types.state import FilterCoeff, JointState
        from curobo.util_file import load_yaml
        from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
        from projects_root.utils.transforms import transform_poses_batched_optimized_for_spheres

        self._Pose = Pose
        self._JointState = JointState
        self._FilterCoeff = FilterCoeff
        self._DynamicObsCost = DynamicObsCost
        self._transform_spheres = transform_poses_batched_optimized_for_spheres

        self.robot_id = robot_id
        self.base_pose = list(base_pose)
        self.robot_cfg = deepcopy(robot_cfg)
        self.particle_file_path = resolve_repo_path(particle_file_path)
        self.particle_cfg_dict = load_yaml(self.particle_file_path)
        self.peer_robot_ids = peer_robot_ids or []
        self.tensor_args = TensorDeviceType()

        if world_cfg is None:
            world_cfg = WorldConfig()

        solver_cfg_dict = {"override_particle_file": self.particle_file_path}
        if solver_config_overrides:
            solver_cfg_dict.update(solver_config_overrides)

        _solver_cfg = MpcSolverConfig.load_from_robot_config(
            self.robot_cfg,
            world_cfg,
            **solver_cfg_dict,
        )
        self.solver = MpcSolver(_solver_cfg)
        self.joint_names: List[str] = list(self.solver.rollout_fn.joint_names)
        self.n_dofs: int = len(self.joint_names)
        self.ee_link_name: str = self.robot_cfg["kinematics"]["ee_link"]

        # Forward kinematics model for fast horizon sphere rollout
        robot_config_obj = RobotConfig.from_dict(self.robot_cfg, self.tensor_args)
        self.crm = CudaRobotModel(robot_config_obj.kinematics)

        base_spheres, extra_spheres = calculate_robot_sphere_count(self.robot_cfg)
        self.n_col_spheres_valid = base_spheres
        self.n_col_spheres_total = base_spheres + extra_spheres
        self.horizon: int = int(self.particle_cfg_dict["model"]["horizon"])

        # Initialize state at retract configuration
        retract_cfg = self.solver.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
        self.current_state = JointState.from_position(retract_cfg, joint_names=self.joint_names)
        kin_state = self.solver.rollout_fn.compute_kinematics(self.current_state)

        initial_goal_R = Pose(position=kin_state.ee_pos_seq, quaternion=kin_state.ee_quat_seq)
        p_W, q_W = FrameUtils.F_to_world(
            np.asarray(self.base_pose[:3], dtype=np.float32),
            np.asarray(self.base_pose[3:], dtype=np.float32),
            initial_goal_R.position.flatten().cpu().numpy(),
            initial_goal_R.quaternion.flatten().cpu().numpy(),
        )
        self.goal_pose_W = (p_W, q_W)

        goal_R = Goal(
            current_state=self.current_state,
            goal_state=JointState.from_position(retract_cfg, joint_names=self.joint_names),
            goal_pose=initial_goal_R,
        )
        self.solver_goal_buf_R = self.solver.setup_solve_single(goal_R, 1)
        self.solver.update_goal(self.solver_goal_buf_R)
        self.solver.step(self.current_state, max_attempts=2)

        self.is_cpred_initialized = False
        self.last_pos_error: float = 0.0
        self.last_rot_error: float = 0.0

    @classmethod
    def create_distributed(
        cls,
        arm_cfg: ArmConfig,
        all_arm_cfgs: List[ArmConfig],
        world_cfg: Optional[object] = None,
    ) -> "MRStormAgent":
        """
        Factory for creating a standalone decentralized MR-STORM agent in its own
        process or ROS 2 node while pre-allocating collision buffers for peer arms.
        """
        from curobo.rollout.cost.custom.arm_base.dynamic_obs_cost import DynamicObsCost
        from curobo.util_file import load_yaml
        from projects_root.projects.dynamic_obs.dynamic_obs_predictor.runtime_topics import (
            get_topics,
            init_runtime_topics,
        )

        n_arms = len(all_arm_cfgs)
        init_runtime_topics(n_envs=1, robots_per_env=n_arms)
        env_topics = get_topics().get_default_env()

        robot_cfgs = []
        robot_paths = []
        particle_paths = []
        particle_cfgs = []
        sphere_splits = []
        sphere_totals = []

        for cfg in all_arm_cfgs:
            r_path = resolve_repo_path(cfg.robot_yml)
            p_path = resolve_repo_path(cfg.particle_yml)
            r_dict = load_yaml(r_path)["robot_cfg"]
            if cfg.retract_cfg is not None:
                r_dict["kinematics"]["cspace"]["retract_config"] = cfg.retract_cfg
            p_dict = load_yaml(p_path)
            b_cnt, e_cnt = calculate_robot_sphere_count(r_dict)

            robot_paths.append(r_path)
            particle_paths.append(p_path)
            robot_cfgs.append(r_dict)
            particle_cfgs.append(p_dict)
            sphere_splits.append((b_cnt, e_cnt))
            sphere_totals.append(b_cnt + e_cnt)

        idx = arm_cfg.robot_id
        peers = (
            arm_cfg.override_col_With
            if arm_cfg.override_col_With is not None
            else [j for j in range(n_arms) if j != idx]
        )
        n_obs_spheres = sum(sphere_totals[j] for j in peers)

        # Populate context for this agent so DynamicObsCost initializes cleanly
        DynamicObsCost._assignment_index = 0
        ctx = env_topics[idx]
        ctx["env_id"] = 0
        ctx["robot_id"] = idx
        ctx["robot_pose"] = list(arm_cfg.base_pose)
        ctx["n_obstacle_spheres"] = n_obs_spheres
        ctx["n_own_spheres"] = sphere_totals[idx]
        ctx["horizon"] = particle_cfgs[idx]["model"]["horizon"]
        ctx["n_rollouts"] = particle_cfgs[idx]["mppi"]["num_particles"]
        ctx["col_pred_with"] = peers
        ctx["mpc_config_paths"] = particle_paths
        ctx["robot_config_paths"] = robot_paths
        ctx["robot_sphere_counts"] = sphere_splits

        return cls(
            robot_id=idx,
            base_pose=arm_cfg.base_pose,
            robot_cfg=robot_cfgs[idx],
            particle_file_path=particle_paths[idx],
            world_cfg=world_cfg,
            peer_robot_ids=peers,
        )

    def get_col_pred(self):
        """Return the underlying `DynamicObsCollPredictor` instance if enabled."""
        custom_costs = (
            self.solver.solver.optimizers[0].rollout_fn._custom_arm_base_costs
        )
        for instance in custom_costs.values():
            if isinstance(instance, self._DynamicObsCost):
                return instance.col_pred
        return None

    def set_goal_world(self, position_W: Union[List[float], np.ndarray], quaternion_W: Union[List[float], np.ndarray]):
        """Set the end-effector target pose in the world frame W (position [x,y,z], quaternion [qw,qx,qy,qz])."""
        p_W = np.asarray(position_W, dtype=np.float32).reshape(3)
        q_W = np.asarray(quaternion_W, dtype=np.float32).reshape(4)
        self.goal_pose_W = (p_W, q_W)

        p_R, q_R = FrameUtils.world_to_F(
            np.asarray(self.base_pose[:3], dtype=np.float32),
            np.asarray(self.base_pose[3:], dtype=np.float32),
            p_W,
            q_W,
        )
        goal_R = self._Pose(
            position=self.solver.tensor_args.to_device(torch.from_numpy(p_R)),
            quaternion=self.solver.tensor_args.to_device(torch.from_numpy(q_R)),
        )
        self.solver_goal_buf_R.goal_pose.copy_(goal_R)
        self.solver.update_goal(self.solver_goal_buf_R)

    def compute_ee_error(self, joint_positions: np.ndarray) -> Tuple[float, float]:
        """Compute current (position_error_m, rotation_error_deg) relative to `goal_pose_W`."""
        q_tensor = self.solver.tensor_args.to_device(
            torch.from_numpy(np.asarray(joint_positions[: self.n_dofs], dtype=np.float32)).unsqueeze(0)
        )
        p_ee_R, q_ee_R, *_ = self.crm.forward(q_tensor)
        p_ee_W, q_ee_W = FrameUtils.F_to_world(
            np.asarray(self.base_pose[:3], dtype=np.float32),
            np.asarray(self.base_pose[3:], dtype=np.float32),
            p_ee_R.squeeze(0).cpu().numpy(),
            q_ee_R.squeeze(0).cpu().numpy(),
        )
        goal_p_W, goal_q_W = self.goal_pose_W
        pos_err = float(np.linalg.norm(p_ee_W - goal_p_W))
        dot = float(np.clip(np.abs(np.dot(q_ee_W, goal_q_W)), 0.0, 1.0))
        rot_err_deg = float(np.degrees(2.0 * np.arccos(dot)))
        self.last_pos_error = pos_err
        self.last_rot_error = rot_err_deg
        return pos_err, rot_err_deg

    def build_broadcast_plan(
        self,
        joint_state: RobotJointState,
        naive_static_broadcast: bool = False,
    ) -> BroadcastPlanPacket:
        """
        Integrate the MPPI mean acceleration policy over the horizon H and compute
        the predicted collision spheres in the world frame W to broadcast to peers.
        """
        pi_mpc_means = self.solver.solver.optimizers[0].mean_action.squeeze(0)  # (H, n_dofs)
        H, n_dofs = pi_mpc_means.shape

        state_filter = self.solver.solver.safety_rollout.dynamics_model.state_filter
        filter_coeff = state_filter.filter_coeff
        dt_planning = state_filter.dt

        pos_seq = torch.zeros_like(pi_mpc_means)
        js = self._JointState(
            torch.from_numpy(joint_state.positions[:n_dofs]),
            torch.from_numpy(joint_state.velocities[:n_dofs]),
            torch.zeros(n_dofs),
            self.joint_names,
            torch.zeros(n_dofs),
            self.solver.tensor_args,
        )
        js_prev = None

        for h in range(H):
            action = pi_mpc_means[h]
            if js_prev is not None:
                js.position[:] = filter_coeff.position * js.position + (1.0 - filter_coeff.position) * js_prev.position
                js.velocity[:] = filter_coeff.velocity * js.velocity + (1.0 - filter_coeff.velocity) * js_prev.velocity
                js.acceleration[:] = (
                    filter_coeff.acceleration * js.acceleration
                    + (1.0 - filter_coeff.acceleration) * js_prev.acceleration
                )
            next_js = js.clone()
            next_js.acceleration[:] = action
            next_js.velocity[:] = js.velocity + next_js.acceleration * dt_planning
            next_js.position[:] = js.position + next_js.velocity * dt_planning
            pos_seq[h] = next_js.position.squeeze()
            js = next_js
            js_prev = js

        _, _, _, _, _, _, prad_spheres = self.crm.forward(self.solver.tensor_args.to_device(pos_seq))
        p_spheres = prad_spheres[:, : self.n_col_spheres_valid, :3]
        r_spheres = prad_spheres[0, : self.n_col_spheres_valid, 3]

        q_ident = torch.zeros(p_spheres.shape[:-1] + (4,), device=p_spheres.device, dtype=p_spheres.dtype)
        q_ident[..., 0] = 1.0
        X_world = self._transform_spheres(torch.cat([p_spheres, q_ident], dim=-1), self.base_pose)
        p_world = X_world[..., :3]

        if naive_static_broadcast:
            p_world = p_world[0:1].expand_as(p_world)

        pos_err, rot_err = self.compute_ee_error(joint_state.positions)

        # Initialize own sphere radii on first broadcast
        col_pred = self.get_col_pred()
        if col_pred is not None and not self.is_cpred_initialized:
            col_pred.set_own_rads(r_spheres)
            self.is_cpred_initialized = True

        return BroadcastPlanPacket(
            robot_id=self.robot_id,
            timestamp=time.time(),
            horizon=H,
            n_spheres=self.n_col_spheres_valid,
            sphere_positions=p_world.detach().cpu().numpy(),
            sphere_radii=r_spheres.detach().cpu().numpy(),
            pos_error=pos_err,
            rot_error=rot_err,
            base_pose=self.base_pose,
        )

    def update_from_peer_plans(self, peer_packets: Dict[int, BroadcastPlanPacket]):
        """
        Update the agent's `DynamicObsCollPredictor` with the latest broadcast
        packets received from peer arms (via shared memory, ROS 2, or ZMQ).
        """
        col_pred = self.get_col_pred()
        if col_pred is None:
            return

        subto_errors = {}
        for peer_id in self.peer_robot_ids:
            pkt = peer_packets.get(peer_id)
            if pkt is None:
                continue
            rad_t = self.solver.tensor_args.to_device(torch.from_numpy(pkt.sphere_radii))
            pos_t = self.solver.tensor_args.to_device(torch.from_numpy(pkt.sphere_positions[: self.horizon]))
            col_pred.set_obs_rads_for_robot(peer_id, rad_t)
            col_pred.update_robot_spheres(peer_id, pos_t)
            subto_errors[peer_id] = (pkt.pos_error, pkt.rot_error)

        if col_pred.prior_rule == "pose_wta":
            col_pred.pose_wta_conflict_resolution["own_errors"] = (
                self.last_pos_error,
                self.last_rot_error,
            )
            col_pred.pose_wta_conflict_resolution["subto_errors"] = subto_errors

    def step(self, joint_state: RobotJointState) -> JointCommand:
        """
        Run one GPU MPPI optimization step given the current `RobotJointState`
        and return the resulting `JointCommand`.
        """
        t0 = time.perf_counter()
        n = self.n_dofs
        cu_js = self._JointState(
            position=self.solver.tensor_args.to_device(
                torch.from_numpy(joint_state.positions[:n]).unsqueeze(0)
            ),
            velocity=self.solver.tensor_args.to_device(
                torch.from_numpy(joint_state.velocities[:n]).unsqueeze(0)
            ),
            acceleration=self.solver.tensor_args.to_device(
                torch.from_numpy(joint_state.accelerations[:n]).unsqueeze(0)
            ),
            joint_names=self.joint_names,
            tensor_args=self.solver.tensor_args,
        )
        self.current_state.copy_(cu_js)

        mpc_result = self.solver.step(self.current_state, max_attempts=2)
        js_action = mpc_result.js_action
        dt_step = float(self.solver.solver.safety_rollout.dynamics_model.state_filter.dt)
        pos_err, rot_err = self.compute_ee_error(joint_state.positions)
        solve_ms = (time.perf_counter() - t0) * 1000.0

        return JointCommand(
            positions=js_action.position.view(-1).detach().cpu().numpy(),
            velocities=(
                js_action.velocity.view(-1).detach().cpu().numpy()
                if js_action.velocity is not None
                else np.zeros(n, dtype=np.float32)
            ),
            accelerations=(
                js_action.acceleration.view(-1).detach().cpu().numpy()
                if js_action.acceleration is not None
                else np.zeros(n, dtype=np.float32)
            ),
            joint_names=list(js_action.joint_names or self.joint_names),
            dt=dt_step,
            ee_position_error=pos_err,
            ee_rotation_error=rot_err,
            solve_time_ms=solve_ms,
        )
