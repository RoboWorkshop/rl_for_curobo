"""
Loyal extraction of the original MR-STORM classes and functions from
`projects_root/experiments/core_api/benchmark_sim.py`.

Preserves the exact algorithm, class hierarchy, method signatures, and data structures:
- `Stopwatch`
- `Plan`
- `PlanPubSub`
- `PoseUtils`
- `get_per_axis_euler_error`
- `publish_robot_context`, `publish_to_context`, `calculate_robot_sphere_count`
- `recursive_fill_from_default`
- `CuPlanner`
- `MpcPlanner`
- `CumotionPlanner`
- `CuAgent`

No top-level `omni.isaac.kit.SimulationApp` import is required, allowing these exact
classes to run with any simulator (Isaac Sim, Gazebo, MuJoCo, Kinematic) or real hardware.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from copy import copy, deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
from queue import Empty, Queue
import random
from threading import Event, Lock, Thread
from time import sleep, time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.geom.types import WorldConfig
from curobo.opt.particle.parallel_mppi import ParallelMPPI
from curobo.rollout.arm_reacher import ArmReacher
from curobo.rollout.cost.custom.arm_base.dynamic_obs_cost import DynamicObsCost
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import FilterCoeff, JointState
from curobo.types.tensor import T_DOF
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.wrap.wrap_mpc import WrapMpc

from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import (
    DynamicObsCollPredictor,
)
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.frame_utils import FrameUtils
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.runtime_topics import (
    get_topics,
    init_runtime_topics,
)
from projects_root.utils.transforms import (
    transform_poses_batched,
    transform_poses_batched_optimized_for_spheres,
)
from projects_root.utils.world_model_wrapper import WorldModelWrapper

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(rel_or_abs: str) -> str:
    """Resolve a relative config path against the repository root if needed."""
    p = Path(rel_or_abs)
    if p.is_absolute():
        return str(p)
    candidate = REPO_ROOT / p
    if candidate.exists():
        return str(candidate)
    curobo_robot_dir = REPO_ROOT / "curobo" / "src" / "curobo" / "content" / "configs" / "robot"
    if (curobo_robot_dir / p).exists():
        return str(curobo_robot_dir / p)
    return str(rel_or_abs)


@dataclass
class JointsStateWrap:
    """
    Lightweight wrapper matching `omni.isaac.core.utils.types.JointsState`
    (`.positions` and `.velocities` numpy arrays), as specified in the docstring
    of `MpcPlanner.get_estimated_plan`:
    '(for real robot, wrap the joint states of the robot with this isaac_JointsState class)'.
    """

    positions: np.ndarray
    velocities: np.ndarray
    efforts: Optional[np.ndarray] = None


def _get_scipy_rotation():
    from scipy.spatial.transform import Rotation as Rotation

    return Rotation


def get_per_axis_euler_error(q1: list[float], q2: list[float]) -> np.ndarray:
    """
    Calculate the rotation error between two quaternions (each wxyz).
    Exact implementation from `benchmark_sim.py`.
    """
    euler1 = _get_scipy_rotation().from_quat(q1).as_euler("xyz", degrees=True)
    euler2 = _get_scipy_rotation().from_quat(q2).as_euler("xyz", degrees=True)
    euler_error = euler2 - euler1
    # Normalize angle to [-180, 180]
    euler_error = (euler_error + 180) % 360 - 180
    return euler_error


class PoseUtils:
    """Exact `PoseUtils` from `benchmark_sim.py`."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            self._local_rng = random.Random(seed)
        else:
            self._local_rng = random.Random()

    def sample_pos_in_box(self, box_center: list[float], box_dim: float) -> list[float]:
        ans = []
        for dim in range(3):
            delta = box_dim / 2
            dim_translation = self._local_rng.uniform(-delta, delta)
            ans.append(box_center[dim] + dim_translation)
        return ans

    def sample_pos_in_sphere(self, sphere_center: list[float], radius: float) -> list[float]:
        ans = []
        x_delta = self._local_rng.uniform(-radius, radius)
        y_delta = self._local_rng.uniform(-radius, radius)
        z_delta_abs = np.sqrt(radius**2 - x_delta**2 + y_delta**2)
        z_delta = self._local_rng.choice([-z_delta_abs, z_delta_abs])
        ans.append(sphere_center[0] + x_delta)
        ans.append(sphere_center[1] + y_delta)
        ans.append(sphere_center[2] + z_delta)
        return ans

    @staticmethod
    def rotate_quat(
        q_in: Union[np.ndarray, list[float]],
        euler_deg: Union[np.ndarray, list[float]],
        q_in_wxyz: bool = True,
        q_out_wxyz: bool = True,
    ) -> list[float]:
        if q_in_wxyz:
            q_in_scipy = [q_in[1], q_in[2], q_in[3], q_in[0]]
        else:
            q_in_scipy = q_in

        r_in = _get_scipy_rotation().from_quat(q_in_scipy)
        r_delta = _get_scipy_rotation().from_euler("xyz", euler_deg, degrees=True)
        r_new = r_delta * r_in

        q_new = r_new.as_quat()  # [x, y, z, w]
        if q_out_wxyz:
            q_out = [float(q_new[3]), float(q_new[0]), float(q_new[1]), float(q_new[2])]
        else:
            q_out = [float(x) for x in q_new]
        return q_out


class Stopwatch:
    """Exact `Stopwatch` from `benchmark_sim.py`."""

    def __init__(self):
        self.total = 0.0
        self.start_time = 0.0
        self.end_time = 0.0

    def on(self):
        self.start_time = time()

    def off(self):
        self.end_time = time()
        new = self.end_time - self.start_time
        self.total += new
        return new

    def reset(self):
        self.total = 0.0
        self.start_time = 0.0
        self.end_time = 0.0


class Plan:
    """Exact `Plan` from `benchmark_sim.py`."""

    def __init__(self, cmd_idx=0, cmd_plan: Optional[JointState] = None):
        self.cmd_idx = cmd_idx
        self.cmd_plan = cmd_plan

    def _is_finished(self):
        exhausted_plan = self.cmd_idx >= len(self.cmd_plan.position) - 1
        if exhausted_plan:
            print(f"DEBUG exhausted_plan: {exhausted_plan} after {self.cmd_idx} steps")
        return exhausted_plan

    def consume_action(self):
        if self.cmd_plan is None or self._is_finished():
            self.cmd_idx = 0
            self.cmd_plan = None
            return None
        else:
            cmd = self.cmd_plan[self.cmd_idx]
            self.cmd_idx += 1
            return cmd


class PlanPubSub:
    """Exact `PlanPubSub` from `benchmark_sim.py`."""

    def __init__(
        self,
        pub_cfg: dict,
        sub_cfg: dict,
        valid_spheres: int,
        total_spheres: int,
    ):
        self.pub_cfg = pub_cfg
        self.sub_cfg = sub_cfg
        self.valid_spheres = valid_spheres
        self.total_spheres = total_spheres

    def should_pub_now(self, t: int) -> bool:
        def bernoulli():
            return random.random() <= self.pub_cfg["pr"]

        if self.pub_cfg["is_dt_in_sec"]:
            if not hasattr(self, "_last_pub_time"):
                self._last_pub_time = time()
            if time() - self._last_pub_time >= self.pub_cfg["dt"] and bernoulli():
                self._last_pub_time = time()
                return True
            return False
        else:
            return t % self.pub_cfg["dt"] == 0 and bernoulli()


class CuPlanner:
    """Exact `CuPlanner` from `benchmark_sim.py`."""

    def __init__(
        self,
        base_pose: list[float],
        solver: Union[MotionGen, MpcSolver],
        solver_config: Union[MotionGenConfig, MpcSolverConfig],
        ordered_j_names: list[str],
        robot_cfg: dict,
    ):
        self.base_pose = base_pose  # robot base pose in world frame
        self.solver = solver
        self.solver_config = solver_config
        self.ordered_j_names = ordered_j_names
        self.is_cpred_initialized = False
        # setup all "goal-constrained" links (links that we can set goals for): ee (musts) + (optional) extra links:
        self.ee_link_name: str = self.solver.kinematics.ee_link
        self.constrained_links_names: list[str] = copy(self.solver.kinematics.link_names)
        if self.ee_link_name in self.constrained_links_names:
            self.constrained_links_names.remove(self.ee_link_name)

        # buffer for the state of the current planning goals (for ee link + extra constrained links)
        self.plan_goals: dict[str, Pose] = {}

        self.crm = CudaRobotModel(CudaRobotModelConfig.from_data_dict(robot_cfg))

    def _set_goals_to_retract_state(self):
        """
        init the plan goals to be as the retract state
        """
        retract_kinematics_state = self.solver.kinematics.get_state(
            self.solver.get_retract_config().view(1, -1)
        )
        links_retract_poses = retract_kinematics_state.link_pose
        ee_retract_pose = retract_kinematics_state.ee_pose

        initial_goals_R = {self.ee_link_name: ee_retract_pose}
        for link_name in self.constrained_links_names:
            initial_goals_R[link_name] = links_retract_poses[link_name]

        initial_goals_W = self.goals_dict_R_to_W(self.base_pose, initial_goals_R)
        self.plan_goals = initial_goals_W

    def yield_action(self, **kwargs) -> Optional[JointState]:
        pass

    def _outdated_plan_goals(self, goals: dict[str, Pose]):
        """
        check if the current plan goals are outdated
        """
        for link_name, goal in goals.items():
            updated_goal_pos = goal.position.flatten().cpu().numpy()
            updated_goal_quat = goal.quaternion.flatten().cpu().numpy()

            current_solver_goal_pos = self.plan_goals[link_name].position.flatten().cpu().numpy()
            current_solver_goal_quat = self.plan_goals[link_name].quaternion.flatten().cpu().numpy()

            pos_changed = np.linalg.norm(current_solver_goal_pos - updated_goal_pos) > 1e-9
            rot_changed = (
                np.linalg.norm(
                    get_per_axis_euler_error(
                        list(current_solver_goal_quat), list(updated_goal_quat)
                    )
                )
                > 1e-9
            )

            if pos_changed or rot_changed:
                return True

        return False

    def convert_action_to_embodiment(
        self,
        full_js_action: JointState,
        ctrl_js_names: list[str],
        order_finder: Optional[Callable[[str], int]] = None,
    ) -> dict[str, Any]:
        """
        Simulator- and hardware-agnostic conversion of CuRobo JointState action
        to target embodiment joint ordering.
        """
        action_idx_list = []
        common_js_names = []
        for idx, x in enumerate(ctrl_js_names):
            if x in full_js_action.joint_names:
                action_idx_list.append(order_finder(x) if order_finder is not None else idx)
                common_js_names.append(x)

        full_ordered_js_action = full_js_action.get_ordered_joint_state(common_js_names)
        self._cmd_state_full = full_ordered_js_action

        pos = full_ordered_js_action.position.view(-1).detach().cpu().numpy()
        vel = (
            full_ordered_js_action.velocity.view(-1).detach().cpu().numpy()
            if full_ordered_js_action.velocity is not None
            else None
        )
        acc = (
            full_ordered_js_action.acceleration.view(-1).detach().cpu().numpy()
            if full_ordered_js_action.acceleration is not None
            else None
        )
        return {
            "positions": pos,
            "velocities": vel,
            "accelerations": acc,
            "joint_indices": action_idx_list,
            "joint_names": common_js_names,
            "ordered_js_action": full_ordered_js_action,
        }

    def convert_action_to_isaac(
        self,
        full_js_action: JointState,
        sim_js_names: list[str],
        order_finder: Callable,
    ):
        """
        A utility function to convert curobo action to isaac sim action (ArticulationAction).
        """
        from omni.isaac.core.utils.types import ArticulationAction

        converted = self.convert_action_to_embodiment(full_js_action, sim_js_names, order_finder)
        articulation_action = ArticulationAction(
            converted["positions"],
            joint_indices=converted["joint_indices"],
        )
        return articulation_action

    def goals_dict_W_to_R(
        self, robot_pose: list[float], goals_W: dict[str, Pose]
    ) -> dict[str, Pose]:
        """
        convert the goals to the robot's base frame
        """
        F = robot_pose
        goals_R = {}
        for link_name, goal in goals_W.items():
            p_goal_R, q_goal_R = FrameUtils.world_to_F(
                np.array(F[:3]),
                np.array(F[3:]),
                goal.position.flatten().cpu().numpy(),
                goal.quaternion.flatten().cpu().numpy(),
            )
            goals_R[link_name] = Pose(
                position=self.solver.tensor_args.to_device(torch.from_numpy(p_goal_R)),
                quaternion=self.solver.tensor_args.to_device(torch.from_numpy(q_goal_R)),
            )
        return goals_R

    def goals_dict_R_to_W(
        self, robot_pose: list[float], goals_R: dict[str, Pose]
    ) -> dict[str, Pose]:
        """
        convert the goals to the world frame
        """
        F = robot_pose
        goals_W = {}
        for link_name, goal in goals_R.items():
            p_goal_W, q_goal_W = FrameUtils.F_to_world(
                np.array(F[:3]),
                np.array(F[3:]),
                np.array(goal.position.flatten().cpu().numpy()),
                np.array(goal.quaternion.flatten().cpu().numpy()),
            )
            goals_W[link_name] = Pose(
                position=self.solver.tensor_args.to_device(torch.from_numpy(p_goal_W)),
                quaternion=self.solver.tensor_args.to_device(torch.from_numpy(q_goal_W)),
            )
        return goals_W

    @abstractmethod
    def get_estimated_plan(
        self,
        dof_names: list[str],
        n_col_spheres_valid: int,
        joints_state: Any,
        include_task_space: bool = True,
        n_steps: int = -1,
        valid_spheres_only: bool = True,
        naive: bool = False,
    ):
        pass

    def get_col_pred(self, **kwargs) -> Optional[DynamicObsCollPredictor]:
        return None

    @abstractmethod
    def update_col_pred(self, plans_board, idx, col_pred_with, goal_errors=None):
        pass

    def get_col_pred_debug(
        self,
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        return None

    def get_spheres_from_solkin(self, to_world_frame: bool = True) -> torch.Tensor:
        """
        return spheres of robot- positions are in robot frame
        """
        spheres_pr_robot_frame = deepcopy(self.solver.kinematics.robot_spheres)
        if spheres_pr_robot_frame is None:
            return torch.tensor([])
        if to_world_frame:
            robot_base_pos = self.solver.tensor_args.to_device(torch.tensor(self.base_pose[:3]))
            spheres_pr_robot_frame[:, :3] += robot_base_pos
        return spheres_pr_robot_frame

    def get_robot_link_meshes_from_solkin(self):
        return self.solver.kinematics.get_robot_link_meshes()

    def attach_external_obj_from_robot(self, joint_state, external_robot):
        return self.solver.kinematics.attach_external_objects_to_robot(joint_state, external_robot)

    def get_state_in_task_space(self, js: JointState, frame="W"):
        """
        js: JointState object
        frame: 'W' or 'R'
        robot_base_pose: list of 7 elements [x,y,z,qw, qx,qy,qz]
        """
        crm = self.crm
        robot_base_pose = self.base_pose

        js_tensor_2d = js.position  # [1,DOF]

        # all the poses from next call are in robot frame:
        p_ee, q_ee, _, _, p_links, q_links, prad = crm.forward(js_tensor_2d)

        q_spheres = torch.empty(prad.shape[:-1] + torch.Size([4]), device=prad.device)
        q_spheres[..., :] = torch.tensor(
            [1, 0, 0, 0], device=prad.device, dtype=prad.dtype
        )  # [1,0,0,0] is the identity quaternion

        d = {
            "ee": {"p": p_ee, "q": q_ee},
            "links": {"p": p_links, "q": q_links},
            "spheres": {"p": prad[:, :, :3], "q": q_spheres, "r": prad[:, :, 3]},
        }

        # We first convert the poses to world frame
        if frame == "W":
            for key in d.keys():
                X_world = transform_poses_batched_optimized_for_spheres(
                    torch.cat([d[key]["p"], d[key]["q"]], dim=-1), robot_base_pose
                )
                d[key]["p"] = X_world[..., :3]
                d[key]["q"] = X_world[..., 3:]

        return d

    def kill_cost_plots(self):
        pass


class MpcPlanner(CuPlanner):
    """Exact `MpcPlanner` from `benchmark_sim.py`."""

    def __init__(
        self,
        base_pose: list[float],
        solver_config_dict: dict,
        robot_cfg: dict,
        world_cfg: WorldConfig,
        particle_file_path: str,
    ):
        self.solver: MpcSolver
        _solver_cfg = MpcSolverConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            **solver_config_dict,
        )
        solver = MpcSolver(_solver_cfg)
        ordered_j_names = solver.rollout_fn.joint_names
        self.cmd_state_full = None
        super().__init__(base_pose, solver, _solver_cfg, ordered_j_names, robot_cfg)

        retract_cfg = self.solver.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
        joint_names = self.solver.rollout_fn.joint_names
        state = self.solver.rollout_fn.compute_kinematics(
            JointState.from_position(retract_cfg, joint_names=joint_names)
        )
        self.current_state = JointState.from_position(retract_cfg, joint_names=joint_names)

        initial_goals_R = {
            self.ee_link_name: Pose(position=state.ee_pos_seq, quaternion=state.ee_quat_seq)
        }
        for link_name in self.constrained_links_names:
            initial_goals_R[link_name] = state.link_poses[link_name]

        initial_goals_W = self.goals_dict_R_to_W(self.base_pose, initial_goals_R)
        self.plan_goals = initial_goals_W

        goal_R = Goal(
            current_state=self.current_state,
            goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
            goal_pose=initial_goals_R[self.ee_link_name],
            links_goal_pose={
                link_name: initial_goals_R[link_name] for link_name in self.constrained_links_names
            },
        )

        self.solver_goal_buf_R = self.solver.setup_solve_single(goal_R, 1)
        self.solver.update_goal(self.solver_goal_buf_R)
        mpc_result = self.solver.step(self.current_state, max_attempts=2)

        self.particle_cfg_dict = load_yaml(resolve_repo_path(particle_file_path))
        self.particle_file_path = particle_file_path
        self.solver_cfg_dict = solver_config_dict
        self.dyn_obs_cost_in_cfg = (
            "cost" in self.particle_cfg_dict
            and "custom" in self.particle_cfg_dict["cost"]
            and "arm_base" in self.particle_cfg_dict["cost"]["custom"]
            and "dynamic_obs_cost" in self.particle_cfg_dict["cost"]["custom"]["arm_base"]
        )

    def yield_action(self, goals: dict[str, Pose]):
        # save the new goals for tracking in future if goals changed
        self.plan_goals = goals

        # update the solver goal buffer with the new goals (in robot frame)
        goals_R = self.goals_dict_W_to_R(self.base_pose, goals)
        self.solver_goal_buf_R.goal_pose.copy_(goals_R[self.ee_link_name])  # ee link goal
        for link_name in self.constrained_links_names:  # extra links goals
            if link_name in goals_R:
                self.solver_goal_buf_R.links_goal_pose[link_name] = goals_R[link_name]
        self.solver.update_goal(self.solver_goal_buf_R)

        mpc_result = self.solver.step(self.current_state, max_attempts=2)
        action = mpc_result.js_action
        self.cmd_state_full = action
        return mpc_result.js_action

    def update_state(self, cu_js: JointState):
        if self.cmd_state_full is None:
            self.current_state.copy_(cu_js)
        else:
            current_state_partial = self.cmd_state_full.get_ordered_joint_state(
                self.solver.rollout_fn.joint_names
            )
            self.current_state.copy_(current_state_partial)
            self.current_state.joint_names = current_state_partial.joint_names

        self.current_state.copy_(cu_js)

    def get_policy_means(self) -> torch.Tensor:
        """Returning the mean values of the mpc policy ([Horizon x N actuated dofs] torch tensor)."""
        return self.solver.solver.optimizers[0].mean_action.squeeze(0)

    def get_estimated_plan(
        self,
        dof_names: list[str],
        n_col_spheres_valid: int,
        joints_state: Any,
        include_task_space: bool = True,
        n_steps: int = -1,
        valid_spheres_only: bool = True,
        naive: bool = False,
    ) -> Optional[dict]:
        """
        Exact `MpcPlanner.get_estimated_plan` from `benchmark_sim.py`.
        """

        def _broadcast_first_step_over_horizon(plan: torch.Tensor) -> torch.Tensor:
            old_shape = plan.shape
            first_step = plan[0]
            return first_step.expand(old_shape)

        def map_nested_tensors(d: dict, fn: Callable[[torch.Tensor], torch.Tensor]) -> dict:
            out = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    out[k] = map_nested_tensors(v, fn)
                elif isinstance(v, torch.Tensor):
                    out[k] = fn(v)
                else:
                    raise TypeError(f"Unsupported value type at key '{k}': {type(v)}")
            return out

        pi_mpc_means = self.get_policy_means()  # (H x num of joints) accelerations

        plan = {
            "joint_space": {
                "acc": pi_mpc_means,
                "vel": torch.zeros(pi_mpc_means.shape),
                "pos": torch.zeros(pi_mpc_means.shape),
            }
        }

        _wrap_mpc = self.solver.solver
        _arm_reacher = _wrap_mpc.safety_rollout
        _kinematics_model = _arm_reacher.dynamics_model
        _state_filter = _kinematics_model.state_filter
        filter_coefficients_solver = _state_filter.filter_coeff
        control_dt_solver = _state_filter.dt

        apply_js_filter = True
        custom_filter = False
        if apply_js_filter:
            if custom_filter:
                filter_coeff = FilterCoeff(0.01, 0.01, 0.0, 0.0)
            else:
                filter_coeff = filter_coefficients_solver
        n_dofs = pi_mpc_means.shape[1]
        js_state = JointState(
            torch.from_numpy(np.asarray(joints_state.positions[:n_dofs], dtype=np.float32)),
            torch.from_numpy(np.asarray(joints_state.velocities[:n_dofs], dtype=np.float32)),
            torch.zeros(n_dofs),
            dof_names,
            torch.zeros(n_dofs),
            self.solver.tensor_args,
        )
        js_state_prev = None

        for h, action in enumerate(pi_mpc_means):
            if apply_js_filter:
                js_state = self._filter_joint_state(js_state_prev, js_state, filter_coeff)
            next_js_state = self._integrate_acc(action, js_state, control_dt_solver)
            plan["joint_space"]["vel"][h] = next_js_state.velocity.squeeze()
            plan["joint_space"]["pos"][h] = next_js_state.position.squeeze()
            js_state = next_js_state
            js_state_prev = js_state

        if include_task_space:
            p_eeplan, q_eeplan, _, _, p_linksplan, q_linksplan, prad_spheresPlan = (
                self.crm.forward(self.solver.tensor_args.to_device(plan["joint_space"]["pos"]))
            )
            task_space_plan = {
                "ee": {"p": p_eeplan, "q": q_eeplan},
                "links": {"p": p_linksplan, "q": q_linksplan},
                "spheres": {
                    "p": prad_spheresPlan[:, :, :3],
                    "r": prad_spheresPlan[:, :, 3],
                },
            }
            plan["task_space"] = task_space_plan

            if valid_spheres_only:
                plan["task_space"]["spheres"]["p"] = plan["task_space"]["spheres"]["p"][
                    :, :n_col_spheres_valid
                ]
                plan["task_space"]["spheres"]["r"] = plan["task_space"]["spheres"]["r"][
                    :, :n_col_spheres_valid
                ]

            # express in world frame:
            for key in plan["task_space"].keys():
                if (
                    isinstance(plan["task_space"][key], dict)
                    and "p" in plan["task_space"][key].keys()
                ):
                    self_transform = self.base_pose
                    pKey = plan["task_space"][key]["p"]
                    if "q" in plan["task_space"][key].keys():
                        qKey = plan["task_space"][key]["q"]
                    else:
                        qKey = torch.empty(
                            pKey.shape[:-1] + torch.Size([4]), device=pKey.device
                        )
                        qKey[..., :] = torch.tensor(
                            [1, 0, 0, 0], device=pKey.device, dtype=pKey.dtype
                        )

                    X_world = transform_poses_batched_optimized_for_spheres(
                        torch.cat([pKey, qKey], dim=-1), self_transform
                    )
                    pKey = X_world[..., :3]
                    qKey = X_world[..., 3:]
                    plan["task_space"][key]["p"] = pKey
                    plan["task_space"][key]["q"] = qKey

            if not n_steps == -1:
                plan = map_nested_tensors(plan, lambda x: x[:n_steps])
        if naive:
            plan = map_nested_tensors(plan, _broadcast_first_step_over_horizon)
        return plan

    def get_col_pred_debug(self):
        col_pred = self.get_col_pred()
        debug = col_pred._debug
        p_obs = debug["p_obs"]
        p_own = debug["p_own"]
        r_obs = debug["r_obs"]
        r_own = debug["r_own"]
        return p_obs, p_own, r_obs, r_own

    def _filter_joint_state(
        self,
        js_state_prev: Union[JointState, None],
        js_state_new: JointState,
        filter_coefficients: FilterCoeff,
    ):
        if js_state_prev is None:
            return js_state_new

        js_state_new.position[:] = (
            filter_coefficients.position * js_state_new.position
            + (1.0 - filter_coefficients.position) * js_state_prev.position
        )
        js_state_new.velocity[:] = (
            filter_coefficients.velocity * js_state_new.velocity
            + (1.0 - filter_coefficients.velocity) * js_state_prev.velocity
        )
        js_state_new.acceleration[:] = (
            filter_coefficients.acceleration * js_state_new.acceleration
            + (1.0 - filter_coefficients.acceleration) * js_state_prev.acceleration
        )
        js_state_new.jerk[:] = (
            filter_coefficients.jerk * js_state_new.jerk
            + (1.0 - filter_coefficients.jerk) * js_state_prev.jerk
        )
        return js_state_new

    def _integrate_acc(
        self, acceleration: T_DOF, cmd_joint_state: JointState, dt_planning: float
    ) -> JointState:
        next_joint_state = cmd_joint_state.clone()
        next_joint_state.acceleration[:] = acceleration
        next_joint_state.velocity[:] = (
            cmd_joint_state.velocity + next_joint_state.acceleration * dt_planning
        )
        next_joint_state.position[:] = (
            cmd_joint_state.position + next_joint_state.velocity * dt_planning
        )
        if cmd_joint_state.jerk is None:
            next_joint_state.jerk = acceleration * 0.0
        return next_joint_state

    def _get_solver(self) -> MpcSolver:
        return self.solver

    def _get_wrap_mpc(self) -> WrapMpc:
        return self._get_solver().solver

    def _get_safety_rollout(self) -> ArmReacher:
        return self._get_wrap_mpc().safety_rollout

    def _get_wrap_mpc_optimizer(self) -> ParallelMPPI:
        return self._get_wrap_mpc().optimizers[0]

    def _get_custom_arm_base_costs(self) -> dict:
        return self._get_wrap_mpc_optimizer().rollout_fn._custom_arm_base_costs

    def _get_custom_arm_reacher_costs(self) -> dict:
        return self._get_wrap_mpc_optimizer().rollout_fn._custom_arm_reacher_costs

    def _get_arm_reacher(self) -> ArmReacher:
        try:
            optimizer = self._get_wrap_mpc_optimizer()
            if optimizer is not None and hasattr(optimizer, "rollout_fn"):
                return optimizer.rollout_fn
            else:
                print("Warning: optimizer is None or doesn't have rollout_fn")
                return None
        except Exception as e:
            print(f"Error getting arm reacher: {e}")
            return None

    def get_col_pred(self) -> Optional[DynamicObsCollPredictor]:
        for instance in self._get_custom_arm_base_costs().values():
            if isinstance(instance, DynamicObsCost):
                return instance.col_pred
        return None

    def kill_cost_plots(self):
        try:
            arm_reacher = self._get_arm_reacher()
            if arm_reacher is not None and hasattr(arm_reacher, "kill_live_plot"):
                arm_reacher.kill_live_plot()
                print("Called arm_reacher.kill_live_plot()")
        except Exception as e:
            print(f"Error in standard kill_live_plot: {e}")

    def update_col_pred(
        self,
        plans_board,
        idx,
        col_pred_with,
        goal_errors,
        plans_lock: Optional[Lock] = None,
    ):
        """Exact `MpcPlanner.update_col_pred` from `benchmark_sim.py`."""

        def _get_rads_own():
            return plans_board[idx]["task_space"]["spheres"]["r"][0]

        def _get_rads_robotj(j):
            return plans_board[j]["task_space"]["spheres"]["r"][0]

        def _get_plan_robotj(j):
            return plans_board[j]["task_space"]["spheres"]

        col_pred = self.get_col_pred()

        if col_pred is None:
            return
        if not self.is_cpred_initialized:
            if plans_lock is not None:
                with plans_lock:
                    if plans_board[idx] is None:
                        return
                    rads_own = _get_rads_own()
            else:
                if plans_board[idx] is None:
                    return
                rads_own = _get_rads_own()
            col_pred.set_own_rads(rads_own)
            self.is_cpred_initialized = True
        else:
            # Update positions for each robot that this robot collides with
            for j in col_pred_with:
                if plans_board[j] is None:
                    continue

                rad_spheres_robotj = _get_rads_robotj(j)
                col_pred.set_obs_rads_for_robot(j, rad_spheres_robotj)
                H = self.particle_cfg_dict["model"]["horizon"]

                plan_robot_j = _get_plan_robotj(j)
                plan_robot_j_horizon = plan_robot_j["p"][:H]
                col_pred.update_robot_spheres(j, plan_robot_j_horizon)

                if (
                    col_pred.prior_rule == "pose_wta"
                    and goal_errors is not None
                    and goal_errors[idx] is not None
                ):
                    subto_to_err = {}
                    for subto in col_pred.col_with_idx_map.keys():
                        if goal_errors[subto] is not None:
                            p_err_subto, q_err_subto = goal_errors[subto]
                            subto_to_err[subto] = (p_err_subto, q_err_subto)

                    col_pred.pose_wta_conflict_resolution["own_errors"] = goal_errors[idx]
                    col_pred.pose_wta_conflict_resolution["subto_errors"] = subto_to_err

    def get_rollouts_in_world_frame(self):
        p_visual_rollouts_robotframe = self.solver.get_visual_rollouts()
        q_visual_rollouts_robotframe = torch.empty(
            p_visual_rollouts_robotframe.shape[:-1] + torch.Size([4]),
            device=p_visual_rollouts_robotframe.device,
        )
        q_visual_rollouts_robotframe[..., :] = torch.tensor(
            [1, 0, 0, 0],
            device=p_visual_rollouts_robotframe.device,
            dtype=p_visual_rollouts_robotframe.dtype,
        )
        visual_rollouts = torch.cat(
            [p_visual_rollouts_robotframe, q_visual_rollouts_robotframe], dim=-1
        )
        visual_rollouts = transform_poses_batched(visual_rollouts, self.base_pose)
        return visual_rollouts


class CumotionPlanner(CuPlanner):
    """Exact `CumotionPlanner` from `benchmark_sim.py`."""

    def __init__(
        self,
        base_pose: list[float],
        motion_gen_config: MotionGenConfig,
        plan_config: MotionGenPlanConfig,
        warmup_config: dict,
        robot_cfg: dict,
    ):
        solver = MotionGen(motion_gen_config)
        solver_config = motion_gen_config
        ordered_j_names = solver.kinematics.joint_names

        super().__init__(base_pose, solver, solver_config, ordered_j_names, robot_cfg)
        self.plan_config = plan_config
        self.warmup_config = warmup_config
        self.solver: MotionGen = self.solver
        print("warming up...")
        self.solver.warmup(**self.warmup_config)
        self.plan = Plan()
        self._set_goals_to_retract_state()

    def _plan_new(
        self,
        cu_js: JointState,
        goals: dict[str, Pose],
    ) -> bool:
        goals_R = self.goals_dict_W_to_R(self.base_pose, goals)
        ee_goal_R = goals_R[self.ee_link_name]
        extra_links_goals_R = {
            link_name: goals_R[link_name] for link_name in self.constrained_links_names
        }
        result = self.solver.plan_single(
            cu_js.unsqueeze(0),
            ee_goal_R,
            self.plan_config.clone(),
            link_poses=extra_links_goals_R,
        )
        succ = result.success.item()
        if succ:
            print("planned successfully, resetting plan...")
            self.plan.cmd_plan = result.get_interpolated_plan()
            self.plan.cmd_idx = 0
            self.plan_goals = deepcopy(goals)
            return True
        else:
            print("WARN: Plan did not converge to a solution: " + str(result.status))
            return False

    def _in_move(self, joint_velocities: np.ndarray):
        return np.max(np.abs(joint_velocities)) > 0.5

    def yield_action(
        self,
        goals: dict[str, Pose],
        cu_js: JointState,
        joint_velocities: np.ndarray,
        stop_when_goal_changed: bool = True,
    ):
        PLAN_NEW = 0
        STOP_IN_PLACE = 1
        CONSUME_FROM_PLAN = 2

        if self._outdated_plan_goals(goals):
            if self._in_move(joint_velocities):
                code = STOP_IN_PLACE if stop_when_goal_changed else CONSUME_FROM_PLAN
                if code == STOP_IN_PLACE:
                    if not hasattr(self, "_js_at_stop_cmd"):
                        self._js_at_stop_cmd = None
                    if self._js_at_stop_cmd is None:
                        self._js_at_stop_cmd = deepcopy(cu_js.position)
            else:
                self._js_at_stop_cmd = None
                code = PLAN_NEW
        else:
            code = CONSUME_FROM_PLAN

        consume = True
        action = None
        if code == PLAN_NEW:
            self._plan_new(cu_js, goals)
        elif code == STOP_IN_PLACE:
            action = JointState(
                position=self._js_at_stop_cmd,
                joint_names=cu_js.joint_names,
            )
            consume = False
        elif code == CONSUME_FROM_PLAN:
            pass
        else:
            raise ValueError(f"Invalid code: {code}")

        if consume:
            action = self.plan.consume_action()

        if action is not None:
            action = deepcopy(self.solver.get_full_js(action))

        return action

    def get_estimated_plan(self, *args, **kwargs):
        return None

    def update_col_pred(self, *args, **kwargs):
        pass


def publish_robot_context(
    robot_idx: int,
    robot_context: dict,
    robot_pose: list,
    n_obstacle_spheres: int,
    robot_sphere_count: int,
    mpc_cfg: dict,
    col_pred_with_robot: List[int],
    mpc_config_paths: List[str],
    robot_config_paths: List[str],
    robot_sphere_counts_split: List[Tuple[int, int]],
):
    """Exact `publish_robot_context` from `benchmark_sim.py`."""
    robot_context["env_id"] = 0
    robot_context["robot_id"] = robot_idx
    robot_context["robot_pose"] = robot_pose
    robot_context["n_obstacle_spheres"] = n_obstacle_spheres

    robot_context["n_own_spheres"] = robot_sphere_count
    robot_context["horizon"] = mpc_cfg["model"]["horizon"]
    robot_context["n_rollouts"] = mpc_cfg["mppi"]["num_particles"]
    robot_context["col_pred_with"] = col_pred_with_robot

    robot_context["mpc_config_paths"] = mpc_config_paths
    robot_context["robot_config_paths"] = robot_config_paths
    robot_context["robot_sphere_counts"] = robot_sphere_counts_split
    robot_context["link_name_to_pose"] = {}
    robot_context["name_link_to_target"] = {}
    robot_context["target_name_to_pose"] = {}


def publish_to_context(robot_context: dict, key: str, value):
    robot_context[key] = value


def calculate_robot_sphere_count(robot_cfg: dict) -> Tuple[int, int]:
    """Exact `calculate_robot_sphere_count` from `benchmark_sim.py`."""
    robots_collision_spheres_configs_parent_dir = str(
        REPO_ROOT / "curobo" / "src" / "curobo" / "content" / "configs" / "robot"
    )

    collision_spheres = robot_cfg["kinematics"]["collision_spheres"]

    if isinstance(collision_spheres, str):
        collision_spheres_cfg = load_yaml(
            os.path.join(robots_collision_spheres_configs_parent_dir, collision_spheres)
        )
        collision_spheres_dict = collision_spheres_cfg["collision_spheres"]
    else:
        collision_spheres_dict = collision_spheres

    sphere_count = 0
    for link_name, spheres in collision_spheres_dict.items():
        if isinstance(spheres, list):
            sphere_count += len(spheres)

    extra_spheres = robot_cfg["kinematics"].get("extra_collision_spheres", {})
    extra_sphere_count = 0
    for obj_name, count in extra_spheres.items():
        extra_sphere_count += count

    return sphere_count, extra_sphere_count


def recursive_fill_from_default(a_cfg: dict, default_cfg: dict, use_deepcopy: bool = False):
    """Exact `recursive_fill_from_default` from `benchmark_sim.py`."""
    for key in default_cfg:
        if key not in a_cfg:
            a_cfg[key] = deepcopy(default_cfg[key]) if use_deepcopy else default_cfg[key]
        elif isinstance(a_cfg[key], dict) and isinstance(default_cfg[key], dict):
            recursive_fill_from_default(a_cfg[key], default_cfg[key], use_deepcopy=use_deepcopy)
    return a_cfg


def parse_viz_color(color: str) -> list[float]:
    if color == "green":
        return [0, 1, 0]
    elif color == "red":
        return [1, 0, 0]
    elif color == "blue":
        return [0, 0, 1]
    elif color == "yellow":
        return [1, 1, 0]
    elif color == "purple":
        return [1, 0, 1]
    elif color == "orange":
        return [1, 0.5, 0]
    else:
        return [1, 0.5, 0]


class CuAgent:
    """
    Exact `CuAgent` from `benchmark_sim.py`, supporting both Isaac Sim (`sim_robot`)
    and simulator/hardware-agnostic environments.
    """

    def __init__(
        self,
        idx: int,
        tensor_args: TensorDeviceType,
        planner: CuPlanner,
        cu_world_wrapper_cfg: dict,
        robot_cfg_path: str,
        robot_cfg: dict,
        sim_robot: Optional[Any] = None,
        plan_pub_sub: Optional[PlanPubSub] = None,
        viz_color: str = "orange",
        stat_man_cfg: dict = {},
        world: Optional[Any] = None,
    ):
        self.idx = idx
        self.step_count = 0
        self.planning_stopwatch = Stopwatch()
        self.tensor_args = tensor_args
        self.planner = planner
        self.base_pose = planner.base_pose
        self.sim_robot = sim_robot
        self.robot_cfg_path = robot_cfg_path
        self.robot_cfg = robot_cfg
        self.plan_pub_sub = plan_pub_sub
        self.viz_color = parse_viz_color(viz_color)
        self.stat_man = None

        _solver_wm = self.planner.solver.world_coll_checker.world_model
        assert isinstance(_solver_wm, WorldConfig)
        self.cu_world_wrapper = WorldModelWrapper(
            world_config=_solver_wm,
            X_robot_W=np.array(self.base_pose),
            verbosity=cu_world_wrapper_cfg.get("verbosity", 0),
        )
        self.cu_world_wrapper_update_policy = {
            "never_add": list(cu_world_wrapper_cfg.get("never_add", [])),
            "never_update": list(cu_world_wrapper_cfg.get("never_update", [])),
        }

        # async worker plumbing (queue-based, lock-free)
        self._worker_thread: Optional[Thread] = None
        self._worker_stop_event = Event()
        self._state_queue: "Queue[tuple[JointState, dict[str, Pose], Optional[np.ndarray]]]" = (
            Queue(maxsize=1)
        )
        self._action_queue: "Queue[Optional[JointState]]" = Queue(maxsize=1)

    def start_worker(self):
        if self._worker_thread is not None:
            return
        self._worker_stop_event.clear()
        self._worker_thread = Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop_worker(self):
        if self._worker_thread is None:
            return
        self._worker_stop_event.set()
        try:
            self._worker_thread.join(timeout=1.0)
        except Exception:
            pass
        self._worker_thread = None

    def post_frame_state(
        self,
        cu_js: JointState,
        goals: dict[str, Pose],
        joint_velocities: Optional[np.ndarray] = None,
    ):
        while not self._state_queue.empty():
            try:
                self._state_queue.get_nowait()
            except Empty:
                break
        try:
            self._state_queue.put_nowait((cu_js, goals, joint_velocities))
        except Exception:
            pass

    def try_get_action(self) -> Optional[JointState]:
        try:
            return self._action_queue.get_nowait()
        except Empty:
            return None

    def _worker_loop(self):
        while not self._worker_stop_event.is_set():
            try:
                packet = self._state_queue.get(timeout=0.001)
            except Empty:
                continue
            try:
                cu_js, goals, js_vel = packet
                if isinstance(self.planner, MpcPlanner):
                    self.planner.update_state(cu_js)
                if isinstance(self.planner, CumotionPlanner):
                    action = self.planner.yield_action(goals, cu_js, js_vel)
                elif isinstance(self.planner, MpcPlanner):
                    action = self.planner.yield_action(goals)
                else:
                    action = None
                while not self._action_queue.empty():
                    try:
                        self._action_queue.get_nowait()
                    except Empty:
                        break
                self._action_queue.put_nowait(action)
            except Exception:
                pass

    def reset_col_model_from_isaac_sim(
        self, usd_help: Any, robot_prim_path: str, ignore_substrings: List[str]
    ):
        isaac_cu_world_R = usd_help.get_obstacles_from_stage(
            only_paths=["/World"],
            reference_prim_path=robot_prim_path,
            ignore_substring=ignore_substrings,
        )
        return self.reset_col_model(isaac_cu_world_R)

    def update_col_model_from_isaac_sim(
        self,
        robot_prim_path: str,
        usd_help: Any,
        ignore_list: List[str] = ["/World/defaultGroundPlane"],
        paths_to_search_obs_under: List[str] = ["/World"],
        sim_lock: Optional[Lock] = None,
    ):
        from projects_root.utils.usd_pose_helper import get_stage_poses, list_relevant_prims

        def _get_pose_dict_from_sim():
            return get_stage_poses(
                usd_helper=usd_help,
                only_paths=paths_to_search_obs_under,
                reference_prim_path="/World",
                ignore_substring=ignore_list,
            )

        def _get_world_cfg_from_stage():
            return usd_help.get_obstacles_from_stage(
                only_paths=list(new_paths),
                reference_prim_path=robot_prim_path,
                ignore_substring=ignore_list,
            )

        if sim_lock is not None:
            with sim_lock:
                pose_dict = _get_pose_dict_from_sim()
        else:
            pose_dict = _get_pose_dict_from_sim()

        self.cu_world_wrapper.update_from_pose_dict(pose_dict)
        current_paths = set(
            list_relevant_prims(usd_help, paths_to_search_obs_under, ignore_list)
        )
        new_paths = current_paths - self.cu_world_wrapper.get_known_prims()

        if new_paths:
            if sim_lock is not None:
                with sim_lock:
                    new_world_cfg = _get_world_cfg_from_stage()
            else:
                new_world_cfg = _get_world_cfg_from_stage()

            new_world_cfg = new_world_cfg.get_collision_check_world()
            if new_world_cfg.objects:
                self.cu_world_wrapper.add_new_obstacles_from_cu_world(
                    cu_world_R=new_world_cfg,
                    silent=False,
                )
                for obj in new_world_cfg.objects:
                    self.cu_world_wrapper.add_prim_to_known(obj.name)

    def update_col_model_from_pose_dict(
        self, pose_dict_W: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ):
        """
        Simulator-agnostic obstacle pose update (for Gazebo, MuJoCo, motion capture, or ROS TF).
        `pose_dict_W` maps obstacle name -> (position_xyz_W, quaternion_wxyz_W).
        """
        if pose_dict_W:
            self.cu_world_wrapper.update_from_pose_dict(pose_dict_W)

    def reset_col_model(self, cu_world_R: WorldConfig):
        """
        Exact `reset_col_model` from `benchmark_sim.py`.
        """
        cu_col_world_R = self.cu_world_wrapper.initialize_from_cu_world(cu_world_R)
        assert self.planner.solver.world_coll_checker is not None
        self.planner.solver.world_coll_checker.load_collision_model(cu_col_world_R)
        self.cu_world_wrapper.set_collision_checker(self.planner.solver.world_coll_checker)
        self.cu_world_wrapper.set_known_prims()

    def get_spheres_from_solkin(self, to_world_frame: bool = True) -> torch.Tensor:
        return self.planner.get_spheres_from_solkin(to_world_frame)

    def update_col_pred(
        self, plans_board, mean_goal_err=None, plans_lock: Optional[Lock] = None
    ):
        sub_to = (
            self.plan_pub_sub.sub_cfg["to"]
            if (self.plan_pub_sub is not None and isinstance(self.planner, MpcPlanner))
            else []
        )
        self.planner.update_col_pred(
            plans_board, self.idx, sub_to, mean_goal_err, plans_lock=plans_lock
        )

    def get_sphere_tensor_W(self, cu_js: JointState) -> torch.Tensor:
        task_space_state_W = self.planner.get_state_in_task_space(cu_js, frame="W")
        spheres_W = task_space_state_W["spheres"]
        p_W, r_W = spheres_W["p"], spheres_W["r"]
        sphere_tensor_W = torch.cat((p_W.squeeze(0), r_W.T), dim=1)
        return sphere_tensor_W

    def split_sphere_tensor_W_into_arms(
        self, sphere_tensor_W: torch.Tensor, n_arms: int
    ) -> list[torch.Tensor]:
        ans = []
        spheres_per_arm = sphere_tensor_W.shape[0] // n_arms
        for i in range(n_arms):
            ans.append(sphere_tensor_W[i * spheres_per_arm : (i + 1) * spheres_per_arm])
        return ans

    def is_plan_subscriber(self) -> bool:
        return self.plan_pub_sub is not None and self.plan_pub_sub.sub_cfg["is_on"]

    def is_full_plan_publisher(self) -> bool:
        return self.plan_pub_sub is not None and self.plan_pub_sub.pub_cfg["is_on"]

    @staticmethod
    def check_collisions_between_agents(
        agents_spheres: list[torch.Tensor],
    ) -> list[list[list[tuple[int, int]]]]:
        ans = [[[] for _ in range(len(agents_spheres))] for _ in range(len(agents_spheres))]
        for i in range(len(agents_spheres)):
            if len(agents_spheres[i]) == 0:
                continue
            for j in range(i + 1, len(agents_spheres)):
                if len(agents_spheres[j]) == 0:
                    continue
                ans[i][j] = CuAgent.agent_to_agent_colcheck(
                    agents_spheres[i], agents_spheres[j]
                )
        return ans

    @staticmethod
    def agent_to_agent_colcheck(ai: torch.Tensor, aj: torch.Tensor):
        if len(ai) == 0 or len(aj) == 0:
            return []
        sphere_tensor_i = ai
        sphere_tensor_j = aj
        sphere_tensor_i_expanded = sphere_tensor_i.unsqueeze(0).expand(
            sphere_tensor_j.shape[0], -1, -1
        )
        sphere_tensor_j_expanded = sphere_tensor_j.unsqueeze(1).expand(
            -1, sphere_tensor_i.shape[0], -1
        )
        sphere_centers_dist = torch.norm(
            sphere_tensor_i_expanded[:, :, :3] - sphere_tensor_j_expanded[:, :, :3], dim=2
        )
        sphere_radii_sum = sphere_tensor_i_expanded[:, :, 3] + sphere_tensor_j_expanded[:, :, 3]
        in_collision = sphere_centers_dist <= sphere_radii_sum
        collision_indices_agent_i = torch.where(in_collision)[0]
        collision_indices_agent_j = torch.where(in_collision)[1]
        return [
            (k, l)
            for k, l in zip(
                collision_indices_agent_i.tolist(), collision_indices_agent_j.tolist()
            )
        ]

    def check_col_with_others(self, agents_spheres: list[torch.Tensor]):
        own_spheres = agents_spheres[self.idx]
        ans = []
        for j in range(len(agents_spheres)):
            if j == self.idx:
                ans.append([])
            else:
                cols_with_j = CuAgent.agent_to_agent_colcheck(own_spheres, agents_spheres[j])
                ans.append(cols_with_j)
        return ans


def build_cu_agents_from_meta_cfg(
    meta_cfg: dict,
    tensor_args: Optional[TensorDeviceType] = None,
    active_agent_indices: Optional[List[int]] = None,
    world_cfg: Optional[WorldConfig] = None,
) -> Tuple[List[CuAgent], Dict[str, Any]]:
    """
    Build `CuAgent` instances from a `meta_cfg` dictionary using the exact
    initialization sequence of `benchmark_sim.py` (lines 3351-3507).

    Args:
        meta_cfg: Parsed meta configuration dict (e.g., from `meta_cfg_arms.yml`)
        tensor_args: Optional `TensorDeviceType`
        active_agent_indices: If None, builds all agents `0..N-1` (for sync or single-process async).
                              If a subset like `[robot_id]` is given (for multi-process ROS async),
                              initializes `runtime_topics` for `robot_id` with full fleet awareness
                              and instantiates only the requested `CuAgent`s.
        world_cfg: Optional initial `WorldConfig` for static obstacles.

    Returns:
        (cu_agents, init_info) where `cu_agents` is the list of `CuAgent`s (or dict/indexed list)
        and `init_info` contains `base_pose`, `col_pred_with`, `sphere_counts_total`, etc.
    """
    if tensor_args is None:
        tensor_args = TensorDeviceType()

    meta_cfg = deepcopy(meta_cfg)
    agent_cfgs = meta_cfg["cu_agents"]
    n_agents = len(agent_cfgs)

    if active_agent_indices is None:
        active_agent_indices = list(range(n_agents))

    # Reset DynamicObsCost assignment counter and initialize runtime topics
    DynamicObsCost._assignment_index = 0
    init_runtime_topics(n_envs=1, robots_per_env=n_agents)
    runtime_topics = get_topics()
    env_topics: List[dict] = (
        runtime_topics.get_default_env() if runtime_topics is not None else []
    )

    robot_cfgs_paths = ["" for _ in range(n_agents)]
    robot_cfgs = [{} for _ in range(n_agents)]
    solver_cfgs = [{} for _ in range(n_agents)]
    planner_type = ["" for _ in range(n_agents)]
    mpc_particle_file_paths = ["" for _ in range(n_agents)]
    mpc_particle_cfgs = [{} for _ in range(n_agents)]
    cumotion_plan_cfgs = [{} for _ in range(n_agents)]
    cumotion_warmup_cfgs = [{} for _ in range(n_agents)]
    sphere_counts_splits = [(-1, -1) for _ in range(n_agents)]
    sphere_counts_total = [-1 for _ in range(n_agents)]
    col_pred_with = [[] for _ in range(n_agents)]
    base_pose = [[] for _ in range(n_agents)]
    pub_sub_cfgs = [{} for _ in range(n_agents)]

    for a_idx, a_cfg in enumerate(agent_cfgs):
        if "default" in meta_cfg:
            recursive_fill_from_default(a_cfg, meta_cfg["default"], use_deepcopy=True)

        robot_cfgs_paths[a_idx] = resolve_repo_path(a_cfg["robot"])
        robot_cfgs[a_idx] = load_yaml(robot_cfgs_paths[a_idx])["robot_cfg"]

        if "retract_cfg" in a_cfg and a_cfg["retract_cfg"] is not None:
            robot_cfgs[a_idx]["kinematics"]["cspace"]["retract_config"] = a_cfg["retract_cfg"]

        base_pose[a_idx] = list(a_cfg["base_pose"])
        if len(base_pose[a_idx]) == 6:
            base_pose[a_idx][3:] = PoseUtils.rotate_quat(
                [1, 0, 0, 0], base_pose[a_idx][3:], q_in_wxyz=True, q_out_wxyz=True
            )
        elif len(base_pose[a_idx]) != 7:
            raise ValueError(f"Invalid base pose length: {len(base_pose[a_idx])}")

        pub_sub_cfgs[a_idx]["sub"] = deepcopy(a_cfg["plan_pub_sub"]["sub"])
        pub_sub_cfgs[a_idx]["pub"] = deepcopy(a_cfg["plan_pub_sub"]["pub"])
        pub_sub_cfgs[a_idx]["is_on"] = a_cfg["plan_pub_sub"]["is_on"]

        if pub_sub_cfgs[a_idx]["sub"]["to"] == "all":
            subto = [i for i in range(n_agents) if i != a_idx]
        else:
            subto = list(a_cfg["plan_pub_sub"]["sub"]["to"])
        pub_sub_cfgs[a_idx]["sub"]["to"] = subto
        col_pred_with[a_idx] = subto

        planner_type[a_idx] = a_cfg["planner"]
        if planner_type[a_idx] == "mpc":
            solver_cfgs[a_idx] = deepcopy(a_cfg["mpc"]["mpc_solver_cfg"])
            raw_particle_path = solver_cfgs[a_idx]["override_particle_file"]
            resolved_particle_path = resolve_repo_path(raw_particle_path)
            solver_cfgs[a_idx]["override_particle_file"] = resolved_particle_path
            mpc_particle_file_paths[a_idx] = resolved_particle_path
            mpc_particle_cfgs[a_idx] = load_yaml(resolved_particle_path)
        else:
            solver_cfgs[a_idx] = deepcopy(a_cfg["cumotion"]["motion_gen_cfg"])
            cumotion_plan_cfgs[a_idx] = deepcopy(a_cfg["cumotion"]["motion_gen_plan_cfg"])
            cumotion_warmup_cfgs[a_idx] = deepcopy(a_cfg["cumotion"]["warmup_cfg"])

        sphere_counts_splits[a_idx] = calculate_robot_sphere_count(robot_cfgs[a_idx])
        sphere_counts_total[a_idx] = (
            sphere_counts_splits[a_idx][0] + sphere_counts_splits[a_idx][1]
        )

    # Publish robot context for active agents (so DynamicObsCost._assignment_index maps 1-to-1)
    for a_idx in active_agent_indices:
        a_cfg = agent_cfgs[a_idx]
        if a_cfg["planner"] == "mpc":
            n_obstacle_spheres = sum(
                sphere_counts_total[other_idx] for other_idx in col_pred_with[a_idx]
            )
            publish_robot_context(
                a_idx,
                env_topics[a_idx],
                base_pose[a_idx],
                n_obstacle_spheres,
                sphere_counts_total[a_idx],
                mpc_particle_cfgs[a_idx],
                col_pred_with[a_idx],
                mpc_particle_file_paths,
                robot_cfgs_paths,
                sphere_counts_splits,
            )

    cu_agents: List[CuAgent] = []
    for a_idx in active_agent_indices:
        a_cfg = agent_cfgs[a_idx]
        viz_color = a_cfg.get("viz_color", "orange")
        agent_world_cfg = deepcopy(world_cfg) if world_cfg is not None else WorldConfig()

        if planner_type[a_idx] == "cumotion":
            _motion_gen_config = MotionGenConfig.load_from_robot_config(
                robot_cfgs[a_idx],
                agent_world_cfg,
                tensor_args,
                **solver_cfgs[a_idx],
            )
            _plan_config = MotionGenPlanConfig(**cumotion_plan_cfgs[a_idx])
            _warmup_config = dict(cumotion_warmup_cfgs[a_idx])
            planner = CumotionPlanner(
                base_pose[a_idx],
                _motion_gen_config,
                _plan_config,
                _warmup_config,
                robot_cfgs[a_idx],
            )
        elif planner_type[a_idx] == "mpc":
            planner = MpcPlanner(
                base_pose[a_idx],
                solver_cfgs[a_idx],
                robot_cfgs[a_idx],
                agent_world_cfg,
                mpc_particle_file_paths[a_idx],
            )
        else:
            raise ValueError(f"Invalid planner type: {planner_type[a_idx]}")

        plan_pub_sub = (
            None
            if (not pub_sub_cfgs[a_idx]["is_on"])
            else PlanPubSub(
                pub_sub_cfgs[a_idx]["pub"],
                pub_sub_cfgs[a_idx]["sub"],
                sphere_counts_splits[a_idx][0],
                sphere_counts_total[a_idx],
            )
        )
        a = CuAgent(
            a_idx,
            tensor_args,
            planner,
            cu_world_wrapper_cfg=a_cfg.get(
                "cu_world_wrapper", {"never_add": [], "never_update": [], "verbosity": 0}
            ),
            robot_cfg_path=robot_cfgs_paths[a_idx],
            robot_cfg=robot_cfgs[a_idx],
            sim_robot=None,
            plan_pub_sub=plan_pub_sub,
            viz_color=viz_color,
        )
        cu_agents.append(a)

    init_info = {
        "n_agents": n_agents,
        "base_pose": base_pose,
        "col_pred_with": col_pred_with,
        "sphere_counts_splits": sphere_counts_splits,
        "sphere_counts_total": sphere_counts_total,
        "robot_cfgs": robot_cfgs,
        "robot_cfgs_paths": robot_cfgs_paths,
        "mpc_particle_file_paths": mpc_particle_file_paths,
    }
    return cu_agents, init_info
