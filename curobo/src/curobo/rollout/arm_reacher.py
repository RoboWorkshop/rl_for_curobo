#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Standard Library
from dataclasses import dataclass
import datetime
import os
from typing import Any, Dict, List, Optional, Union

# Third Party
from curobo.rollout.cost.custom.custom_cost import CustomCost
import torch
import torch.autograd.profiler as profiler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
# CuRobo
from curobo.geom.sdf.world import WorldCollision
from curobo.rollout.cost.cost_base import CostConfig
from curobo.rollout.cost.dist_cost import DistCost, DistCostConfig
from curobo.rollout.cost.pose_cost import PoseCost, PoseCostConfig, PoseCostMetric
from curobo.rollout.cost.pose_cost_multi_arm import PoseCostMultiArm, PoseCostMultiArmConfig
from curobo.rollout.cost.straight_line_cost import StraightLineCost
from curobo.rollout.cost.zero_cost import ZeroCost
from curobo.rollout.dynamics_model.kinematic_model import KinematicModelState
from curobo.rollout.rollout_base import Goal, RolloutMetrics
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.types.tensor import T_BValue_float, T_BValue_int
from curobo.util.helpers import list_idx_if_not_none
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util.tensor_util import cat_max
from curobo.util.torch_utils import get_torch_jit_decorator

from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import DynamicObsCollPredictor
from curobo.rollout.cost.pose_cost_multi_arm import PoseCostMultiArm

# Local Folder
from .arm_base import ArmBase, ArmBaseConfig, ArmCostConfig


@dataclass
class ArmReacherMetrics(RolloutMetrics):
    cspace_error: Optional[T_BValue_float] = None
    position_error: Optional[T_BValue_float] = None
    rotation_error: Optional[T_BValue_float] = None
    pose_error: Optional[T_BValue_float] = None
    goalset_index: Optional[T_BValue_int] = None
    null_space_error: Optional[T_BValue_float] = None

    def __getitem__(self, idx):
        d_list = [
            self.cost,
            self.constraint,
            self.feasible,
            self.state,
            self.cspace_error,
            self.position_error,
            self.rotation_error,
            self.pose_error,
            self.goalset_index,
            self.null_space_error,
        ]
        idx_vals = list_idx_if_not_none(d_list, idx)
        return ArmReacherMetrics(*idx_vals)

    def clone(self, clone_state=False):
        if clone_state:
            raise NotImplementedError()
        return ArmReacherMetrics(
            cost=None if self.cost is None else self.cost.clone(),
            constraint=None if self.constraint is None else self.constraint.clone(),
            feasible=None if self.feasible is None else self.feasible.clone(),
            state=None if self.state is None else self.state,
            cspace_error=None if self.cspace_error is None else self.cspace_error.clone(),
            position_error=None if self.position_error is None else self.position_error.clone(),
            rotation_error=None if self.rotation_error is None else self.rotation_error.clone(),
            pose_error=None if self.pose_error is None else self.pose_error.clone(),
            goalset_index=None if self.goalset_index is None else self.goalset_index.clone(),
            null_space_error=(
                None if self.null_space_error is None else self.null_space_error.clone()
            ),
        )


@dataclass
class ArmReacherCostConfig(ArmCostConfig):
    pose_cfg: Optional[PoseCostConfig] = None
    cspace_cfg: Optional[DistCostConfig] = None
    straight_line_cfg: Optional[CostConfig] = None
    zero_acc_cfg: Optional[CostConfig] = None
    zero_vel_cfg: Optional[CostConfig] = None
    zero_jerk_cfg: Optional[CostConfig] = None
    link_pose_cfg: Optional[PoseCostConfig] = None
    # custom_cfg is inherited from ArmCostConfig

    @staticmethod
    def _get_base_keys():
        base_k = ArmCostConfig._get_base_keys()
        # add new cost terms:
        new_k = {
            "pose_cfg": PoseCostConfig,  # Revert to single config
            "cspace_cfg": DistCostConfig,
            "straight_line_cfg": CostConfig,
            "zero_acc_cfg": CostConfig,
            "zero_vel_cfg": CostConfig,
            "zero_jerk_cfg": CostConfig,
            "link_pose_cfg": PoseCostConfig,
        }
        new_k.update(base_k)
        return new_k

    @staticmethod
    def from_dict(
        data_dict: Dict,
        robot_cfg: RobotConfig,
        world_coll_checker: Optional[WorldCollision] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        enable_auto_discovery: bool = False,
        _num_particles_rollout_full: int = -1,
    ):
        k_list = ArmReacherCostConfig._get_base_keys()
        
        # Handle multi-arm pose configuration conditionally
        if "pose_cfg" in data_dict and isinstance(data_dict["pose_cfg"], dict):
            pose_cfg = data_dict["pose_cfg"]
            if "num_arms" in pose_cfg and pose_cfg["num_arms"] > 1:
                # Use PoseCostMultiArmConfig for multi-arm setup
                k_list = k_list.copy()  # Make a copy to avoid modifying the original
                k_list["pose_cfg"] = PoseCostMultiArmConfig
        
        data = ArmCostConfig._get_formatted_dict(
            data_dict,
            k_list,
            robot_cfg,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
            _num_particles_rollout_full=_num_particles_rollout_full,
        )
        
        # Handle custom costs with auto-discovery (inherited from ArmCostConfig)
        custom_dict = data_dict.get("custom", {})
        
        
        data["custom_cfg"] = ArmCostConfig._parse_custom_costs(
            custom_dict, 
            tensor_args, 
            enable_auto_discovery=enable_auto_discovery,
            _num_particles_rollout_full=_num_particles_rollout_full,
        )
        
        # CustomCost.parse_cfg_from_file(custom_dict) 
        
        
        return ArmReacherCostConfig(**data)


@dataclass
class ArmReacherConfig(ArmBaseConfig):
    cost_cfg: Optional[ArmReacherCostConfig] = None
    constraint_cfg: Optional[ArmReacherCostConfig] = None
    convergence_cfg: Optional[ArmReacherCostConfig] = None

    @staticmethod
    def cost_from_dict(
        cost_data_dict: Dict,
        robot_cfg: RobotConfig,
        world_coll_checker: Optional[WorldCollision] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        enable_auto_discovery: bool = False,
        _num_particles_rollout_full: int = -1,
    ):
        return ArmReacherCostConfig.from_dict(
            cost_data_dict,
            robot_cfg,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
            enable_auto_discovery=enable_auto_discovery,
            _num_particles_rollout_full=_num_particles_rollout_full,
        )

    @classmethod
    def from_dict(
        cls,
        robot_cfg: Union[Dict, RobotConfig],
        model_data_dict: Dict,
        cost_data_dict: Dict,
        constraint_data_dict: Dict,
        convergence_data_dict: Dict,
        world_coll_checker_dict: Optional[Dict] = None,
        world_model_dict: Optional[Dict] = None,
        world_coll_checker: Optional[WorldCollision] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        enable_auto_discovery: bool = False,
        _num_particles_rollout_full: int = -1,
    ):
        """Create ArmReacherConfig from dictionary, properly handling _num_particles_rollout_full parameter."""
        
        # Call parent from_dict method with all parameters
        base_config = super().from_dict(
            robot_cfg=robot_cfg,
            model_data_dict=model_data_dict,
            cost_data_dict=cost_data_dict,
            constraint_data_dict=constraint_data_dict,
            convergence_data_dict=convergence_data_dict,
            world_coll_checker_dict=world_coll_checker_dict,
            world_model_dict=world_model_dict,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
            enable_auto_discovery=enable_auto_discovery,
            _num_particles_rollout_full=_num_particles_rollout_full,
        )
        
        # Override cost configurations with ArmReacher-specific ones
        cost = cls.cost_from_dict(
            cost_data_dict,
            base_config.model_cfg.robot_config,
            world_coll_checker=base_config.world_coll_checker,
            tensor_args=tensor_args,
            enable_auto_discovery=enable_auto_discovery,
            _num_particles_rollout_full=_num_particles_rollout_full,
        )
        constraint = cls.cost_from_dict(
            constraint_data_dict,
            base_config.model_cfg.robot_config,
            world_coll_checker=base_config.world_coll_checker,
            tensor_args=tensor_args,
            enable_auto_discovery=enable_auto_discovery,
            _num_particles_rollout_full=_num_particles_rollout_full,
        )
        convergence = cls.cost_from_dict(
            convergence_data_dict,
            base_config.model_cfg.robot_config,
            world_coll_checker=base_config.world_coll_checker,
            tensor_args=tensor_args,
            enable_auto_discovery=enable_auto_discovery,
            _num_particles_rollout_full=_num_particles_rollout_full,
        )
        
        # Create ArmReacherConfig with the proper cost configurations
        result = cls(
            model_cfg=base_config.model_cfg,
            cost_cfg=cost,
            constraint_cfg=constraint,
            convergence_cfg=convergence,
            world_coll_checker=base_config.world_coll_checker,
            tensor_args=base_config.tensor_args,
            sum_horizon=base_config.sum_horizon,
            sampler_seed=base_config.sampler_seed,
            _num_particles_rollout_full=_num_particles_rollout_full,
        )
        
        return result


@get_torch_jit_decorator()
def _compute_g_dist_jit(rot_err_norm, goal_dist):
    # goal_cost = goal_cost.view(cost.shape)
    # rot_err_norm = rot_err_norm.view(cost.shape)
    # goal_dist = goal_dist.view(cost.shape)
    g_dist = goal_dist.unsqueeze(-1) + 10.0 * rot_err_norm.unsqueeze(-1)
    return g_dist


class ArmReacher(ArmBase, ArmReacherConfig):
    """
    .. inheritance-diagram:: curobo.rollout.arm_reacher.ArmReacher
    """

    @profiler.record_function("arm_reacher/init")
    def __init__(self, config: Optional[ArmReacherConfig] = None):
        if config is not None:
            ArmReacherConfig.__init__(self, **vars(config))
        ArmBase.__init__(self)

        # self.goal_state = None
        # self.goal_ee_pos = None
        # self.goal_ee_rot = None
        # self.goal_ee_quat = None
        self._compute_g_dist = False
        self._n_goalset = 1

        if self.cost_cfg.cspace_cfg is not None:
            self.cost_cfg.cspace_cfg.dof = self.d_action
            # self.cost_cfg.cspace_cfg.update_vec_weight(self.dynamics_model.cspace_distance_weight)
            self.dist_cost = DistCost(self.cost_cfg.cspace_cfg)
        if self.cost_cfg.pose_cfg is not None:
            self.cost_cfg.pose_cfg.waypoint_horizon = self.horizon
            
            # Check if this is a multi-arm setup
            num_arms = getattr(self.cost_cfg.pose_cfg, 'num_arms', 1)
            if num_arms > 1:
                # Convert PoseCostConfig to PoseCostMultiArmConfig
                multi_arm_config = PoseCostMultiArmConfig(**vars(self.cost_cfg.pose_cfg))
                multi_arm_config.num_arms = num_arms
                self.goal_cost = PoseCostMultiArm(multi_arm_config)
            else:
                self.goal_cost = PoseCost(self.cost_cfg.pose_cfg)
                
            if self.cost_cfg.link_pose_cfg is None:
                log_info(
                    "Deprecated: Add link_pose_cfg to your rollout config. Using pose_cfg instead."
                )
                self.cost_cfg.link_pose_cfg = self.cost_cfg.pose_cfg
        self._link_pose_costs = {}

        if self.cost_cfg.link_pose_cfg is not None:
            # For link pose costs, always use regular PoseCostConfig even if main pose cost is multi-arm
            link_pose_config = self.cost_cfg.link_pose_cfg
            if hasattr(link_pose_config, 'num_arms'):
                # Convert multi-arm config to regular config for individual link poses
                link_pose_config = PoseCostConfig(
                    cost_type=link_pose_config.cost_type,
                    use_metric=link_pose_config.use_metric,
                    project_distance=link_pose_config.project_distance,
                    run_vec_weight=getattr(link_pose_config, 'run_vec_weight', None),
                    use_projected_distance=link_pose_config.use_projected_distance,
                    offset_waypoint=getattr(link_pose_config, 'offset_waypoint', [0, 0, 0, 0, 0, 0]),
                    offset_tstep_fraction=link_pose_config.offset_tstep_fraction,
                    waypoint_horizon=link_pose_config.waypoint_horizon,
                    weight=link_pose_config.weight,
                    vec_weight=link_pose_config.vec_weight,
                    vec_convergence=link_pose_config.vec_convergence,
                    run_weight=link_pose_config.run_weight,
                    return_loss=link_pose_config.return_loss,
                    terminal=link_pose_config.terminal,
                    tensor_args=link_pose_config.tensor_args
                )
            
            for i in self.kinematics.link_names:
                if i != self.kinematics.ee_link:
                    self._link_pose_costs[i] = PoseCost(link_pose_config)
        if self.cost_cfg.straight_line_cfg is not None:
            self.straight_line_cost = StraightLineCost(self.cost_cfg.straight_line_cfg)
        if self.cost_cfg.zero_vel_cfg is not None:
            self.zero_vel_cost = ZeroCost(self.cost_cfg.zero_vel_cfg)
            self._max_vel = self.state_bounds["velocity"][1]
            if self.zero_vel_cost.hinge_value is not None:
                self._compute_g_dist = True
        if self.cost_cfg.zero_acc_cfg is not None:
            self.zero_acc_cost = ZeroCost(self.cost_cfg.zero_acc_cfg)
            self._max_vel = self.state_bounds["velocity"][1]
            if self.zero_acc_cost.hinge_value is not None:
                self._compute_g_dist = True

        if self.cost_cfg.zero_jerk_cfg is not None:
            self.zero_jerk_cost = ZeroCost(self.cost_cfg.zero_jerk_cfg)
            self._max_vel = self.state_bounds["velocity"][1]
            if self.zero_jerk_cost.hinge_value is not None:
                self._compute_g_dist = True

        self.z_tensor = torch.tensor(
            0, device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        self._link_pose_convergence = {}

        if self.convergence_cfg.pose_cfg is not None:
            self.pose_convergence = PoseCost(self.convergence_cfg.pose_cfg)
            if self.convergence_cfg.link_pose_cfg is None:
                log_warn(
                    "Deprecated: Add link_pose_cfg to your rollout config. Using pose_cfg instead."
                )
                self.convergence_cfg.link_pose_cfg = self.convergence_cfg.pose_cfg

        if self.convergence_cfg.link_pose_cfg is not None:
            # For convergence link pose costs, always use regular PoseCostConfig even if main pose cost is multi-arm
            conv_link_pose_config = self.convergence_cfg.link_pose_cfg
            if hasattr(conv_link_pose_config, 'num_arms'):
                # Convert multi-arm config to regular config for individual link poses
                conv_link_pose_config = PoseCostConfig(
                    cost_type=conv_link_pose_config.cost_type,
                    use_metric=conv_link_pose_config.use_metric,
                    project_distance=conv_link_pose_config.project_distance,
                    run_vec_weight=getattr(conv_link_pose_config, 'run_vec_weight', None),
                    use_projected_distance=conv_link_pose_config.use_projected_distance,
                    offset_waypoint=getattr(conv_link_pose_config, 'offset_waypoint', [0, 0, 0, 0, 0, 0]),
                    offset_tstep_fraction=conv_link_pose_config.offset_tstep_fraction,
                    waypoint_horizon=conv_link_pose_config.waypoint_horizon,
                    weight=conv_link_pose_config.weight,
                    vec_weight=conv_link_pose_config.vec_weight,
                    vec_convergence=conv_link_pose_config.vec_convergence,
                    run_weight=conv_link_pose_config.run_weight,
                    return_loss=conv_link_pose_config.return_loss,
                    terminal=conv_link_pose_config.terminal,
                    tensor_args=conv_link_pose_config.tensor_args
                )
                
            for i in self.kinematics.link_names:
                if i != self.kinematics.ee_link:
                    self._link_pose_convergence[i] = PoseCost(conv_link_pose_config)
        if self.convergence_cfg.cspace_cfg is not None:
            self.convergence_cfg.cspace_cfg.dof = self.d_action
            self.cspace_convergence = DistCost(self.convergence_cfg.cspace_cfg)

        # check if g_dist is required in any of the cost terms:
        self.update_params(Goal(current_state=self._start_state))

        # Initialize custom costs for arm_reacher
        self._custom_arm_reacher_costs = {}
        if (hasattr(self.cost_cfg, 'custom_cfg') and 
            self.cost_cfg.custom_cfg is not None and 
            "arm_reacher" in self.cost_cfg.custom_cfg):
            modules_path_prefix = CustomCost.get_modules_path_prefix() + 'arm_reacher' + '.'
            
            # # go over all the custom cost terms
            # for cost_module_name, cost_info in self.cost_cfg.custom_cfg["arm_reacher"].items():
            #     module_path = modules_path_prefix + cost_module_name
            #     # cost_class = cost_info["cost_class"]
            #     # cost_config = cost_info["cost_config"]
            #     cost_class = 'Cost'
            #     cost_config = 'Cfg'
            #     cost_config._horizon_rollout_full = self.horizon
            #     cost_config._num_particles_rollout_full = self._num_particles_rollout_full
            #     cost_instance = cost_class(cost_config)
            #     self._custom_arm_reacher_costs[cost_module_name] = cost_instance
                
    def cost_fn(self, state: KinematicModelState, action_batch=None):
        """
        Compute cost given that state dictionary and actions


        :class:`curobo.rollout.cost.PoseCost`
        :class:`curobo.rollout.cost.DistCost`

        """
        state_batch = state.state_seq
        with profiler.record_function("cost/base"):
            # get ArmBase cost
            cost_list = super(ArmReacher, self).cost_fn(state, action_batch, return_list=True)
            
        ee_pos_batch, ee_quat_batch = state.ee_pos_seq, state.ee_quat_seq
        g_dist = None
        with profiler.record_function("cost/pose"):
            if (
                self._goal_buffer.goal_pose.position is not None
                and self.cost_cfg.pose_cfg is not None
                and self.goal_cost.enabled
            ):
                # Check if this is a multi-arm pose cost that needs special handling
                is_multi_arm_pose_cost = isinstance(self.goal_cost, PoseCostMultiArm)
                
                # Format end-effector data for multi-arm pose cost if needed
                if is_multi_arm_pose_cost:
                    # For multi-arm pose cost, construct 4D tensors from link poses
                    num_arms = getattr(self.goal_cost, 'num_arms', 2)  # Default to 2 for dual-arm
                    multi_arm_ee_pos, multi_arm_ee_quat = self._format_multi_arm_ee_data(state, num_arms)
                    ee_pos_for_cost = multi_arm_ee_pos
                    ee_quat_for_cost = multi_arm_ee_quat
                    
                    # Debug: Track both arms' poses over batch x horizon (first few times)
                    if not hasattr(self, '_debug_ee_poses_count'):
                        self._debug_ee_poses_count = 0
                    self._debug_ee_poses_count += 1
                    
                    if self._debug_ee_poses_count <= 3:  # Debug first 3 cost function calls
                        print(f"\n=== EE Poses Debug #{self._debug_ee_poses_count} ===")
                        batch_size, horizon, num_arms_tensor, _ = multi_arm_ee_pos.shape
                        print(f"Multi-arm EE data shape: pos={multi_arm_ee_pos.shape}, quat={multi_arm_ee_quat.shape}")
                        
                        for arm_idx in range(min(num_arms_tensor, 2)):  # Only debug first 2 arms
                            print(f"\n--- Arm {arm_idx} EE Poses Over Horizon ---")
                            for t in range(min(horizon, 5)):  # First 5 time steps
                                arm_pos = multi_arm_ee_pos[0, t, arm_idx, :]  # [3]
                                arm_quat = multi_arm_ee_quat[0, t, arm_idx, :]  # [4]
                                print(f"  t={t}: pos={arm_pos.cpu().numpy()}, quat={arm_quat.cpu().numpy()}")
                        
                        # Also print goal targets for comparison
                        if hasattr(self._goal_buffer, 'goal_pose') and self._goal_buffer.goal_pose.position is not None:
                            goal_pos = self._goal_buffer.goal_pose.position
                            goal_quat = self._goal_buffer.goal_pose.quaternion
                            print(f"\n--- Goal Targets ---")
                            if goal_pos.dim() == 2 and goal_pos.shape[0] >= 2:  # Multi-arm goals
                                for arm_idx in range(min(goal_pos.shape[0], 2)):
                                    print(f"  Arm {arm_idx} goal: pos={goal_pos[arm_idx, :].cpu().numpy()}, quat={goal_quat[arm_idx, :].cpu().numpy()}")
                            else:
                                print(f"  Single goal: pos={goal_pos.cpu().numpy()}, quat={goal_quat.cpu().numpy()}")
                        
                        print("=== End EE Poses Debug ===\n")
                else:
                    # For single-arm pose cost, use the regular ee_pos_batch and ee_quat_batch
                    ee_pos_for_cost = ee_pos_batch
                    ee_quat_for_cost = ee_quat_batch
                
                if self._compute_g_dist:
                    goal_cost, rot_err_norm, goal_dist = self.goal_cost.forward_out_distance(
                        ee_pos_for_cost,
                        ee_quat_for_cost,
                        self._goal_buffer,
                    )

                    g_dist = _compute_g_dist_jit(rot_err_norm, goal_dist)
                else:
                    goal_cost = self.goal_cost.forward(
                        ee_pos_for_cost, ee_quat_for_cost, self._goal_buffer
                    )
                cost_list.append(goal_cost)
        with profiler.record_function("cost/link_poses"):
            if self._goal_buffer.links_goal_pose is not None and self.cost_cfg.pose_cfg is not None:
                link_poses = state.link_pose

                for k in self._goal_buffer.links_goal_pose.keys():
                    if k != self.kinematics.ee_link:
                        current_fn = self._link_pose_costs[k]
                        if current_fn.enabled:
                            # get link pose
                            current_pose = link_poses[k].contiguous()
                            current_pos = current_pose.position
                            current_quat = current_pose.quaternion

                            c = current_fn.forward(current_pos, current_quat, self._goal_buffer, k)
                            cost_list.append(c)

        if (
            self._goal_buffer.goal_state is not None
            and self.cost_cfg.cspace_cfg is not None
            and self.dist_cost.enabled
        ):

            joint_cost = self.dist_cost.forward_target_idx(
                self._goal_buffer.goal_state.position,
                state_batch.position,
                self._goal_buffer.batch_goal_state_idx,
            )
            cost_list.append(joint_cost)
        if self.cost_cfg.straight_line_cfg is not None and self.straight_line_cost.enabled:
            st_cost = self.straight_line_cost.forward(ee_pos_batch)
            cost_list.append(st_cost)

        if (
            self.cost_cfg.zero_acc_cfg is not None
            and self.zero_acc_cost.enabled
            # and g_dist is not None
        ):
            z_acc = self.zero_acc_cost.forward(
                state_batch.acceleration,
                g_dist,
            )

            cost_list.append(z_acc)
        if self.cost_cfg.zero_jerk_cfg is not None and self.zero_jerk_cost.enabled:
            z_jerk = self.zero_jerk_cost.forward(
                state_batch.jerk,
                g_dist,
            )
            cost_list.append(z_jerk)

        if self.cost_cfg.zero_vel_cfg is not None and self.zero_vel_cost.enabled:
            z_vel = self.zero_vel_cost.forward(
                state_batch.velocity,
                g_dist,
            )
            cost_list.append(z_vel)
        
        # Execute custom arm_reacher costs
        if hasattr(self, '_custom_arm_reacher_costs'):
            for cost_name, cost_instance in self._custom_arm_reacher_costs.items():
                if cost_instance.enabled:
                    with profiler.record_function(f"cost/custom_arm_reacher/{cost_name}"):
                        try:
                            custom_cost = cost_instance.forward(state)
                            cost_list.append(custom_cost)
                        except Exception as e:
                            log_error(f"Error computing custom arm_reacher cost {cost_name}: {e}")
        
        # Add live plotting support for ArmReacher - plot all costs in one comprehensive view
        if getattr(self, '_enable_live_plotting', False):
            self._update_live_plot_reacher(cost_list)
            
        with profiler.record_function("cat_sum"):
            if self.sum_horizon:
                cost = cat_sum_horizon_reacher(cost_list)
            else:
                cost = cat_sum_reacher(cost_list)

        return cost

    def convergence_fn(
        self, state: KinematicModelState, out_metrics: Optional[ArmReacherMetrics] = None
    ) -> ArmReacherMetrics:
        if out_metrics is None:
            out_metrics = ArmReacherMetrics()
        if not isinstance(out_metrics, ArmReacherMetrics):
            out_metrics = ArmReacherMetrics(**vars(out_metrics))
        base_metrics = super(ArmReacher, self).convergence_fn(state, out_metrics)
        
        # Copy base metrics to our typed metrics
        if base_metrics != out_metrics:
            out_metrics.cost = base_metrics.cost
            out_metrics.constraint = base_metrics.constraint
            out_metrics.feasible = base_metrics.feasible
            out_metrics.state = base_metrics.state

        # compute error with pose?
        if (
            self._goal_buffer.goal_pose.position is not None
            and self.convergence_cfg.pose_cfg is not None
        ):
            (
                out_metrics.pose_error,
                out_metrics.rotation_error,
                out_metrics.position_error,
            ) = self.pose_convergence.forward_out_distance(
                state.ee_pos_seq, state.ee_quat_seq, self._goal_buffer
            )
            out_metrics.goalset_index = self.pose_convergence.goalset_index_buffer  # .clone()
        if (
            self._goal_buffer.links_goal_pose is not None
            and self.convergence_cfg.pose_cfg is not None
        ):
            pose_error = [out_metrics.pose_error] if out_metrics.pose_error is not None else []
            position_error = [out_metrics.position_error] if out_metrics.position_error is not None else []
            quat_error = [out_metrics.rotation_error] if out_metrics.rotation_error is not None else []
            link_poses = state.link_pose

            for k in self._goal_buffer.links_goal_pose.keys():
                if k != self.kinematics.ee_link:
                    current_fn = self._link_pose_convergence[k]
                    if current_fn.enabled:
                        # get link pose
                        current_pos = link_poses[k].position.contiguous()
                        current_quat = link_poses[k].quaternion.contiguous()

                        pose_err, pos_err, quat_err = current_fn.forward_out_distance(
                            current_pos, current_quat, self._goal_buffer, k
                        )
                        pose_error.append(pose_err)
                        position_error.append(pos_err)
                        quat_error.append(quat_err)
            if pose_error:
                out_metrics.pose_error = cat_max(pose_error)
                out_metrics.rotation_error = cat_max(quat_error)
                out_metrics.position_error = cat_max(position_error)

        if (
            self._goal_buffer.goal_state is not None
            and self.convergence_cfg.cspace_cfg is not None
            and self.cspace_convergence.enabled
        ):
            _, out_metrics.cspace_error = self.cspace_convergence.forward_target_idx(
                self._goal_buffer.goal_state.position,
                state.state_seq.position,
                self._goal_buffer.batch_goal_state_idx,
                True,
            )

        if (
            self.convergence_cfg.null_space_cfg is not None
            and self.null_convergence.enabled
            and self._goal_buffer.batch_retract_state_idx is not None
        ):
            out_metrics.null_space_error = self.null_convergence.forward_target_idx(
                self._goal_buffer.retract_state,
                state.state_seq.position,
                self._goal_buffer.batch_retract_state_idx,
            )

        return out_metrics

    def update_params(
        self,
        goal: Goal,
    ):
        """
        Update params for the cost terms and dynamics model.

        """

        super(ArmReacher, self).update_params(goal)
        if goal.batch_pose_idx is not None:
            self._goal_idx_update = False
        if goal.goal_pose.position is not None:
            self.enable_cspace_cost(False)
        return True

    def enable_pose_cost(self, enable: bool = True):
        if enable:
            self.goal_cost.enable_cost()
        else:
            self.goal_cost.disable_cost()

    def enable_cspace_cost(self, enable: bool = True):
        if enable:
            self.dist_cost.enable_cost()
            self.cspace_convergence.enable_cost()
        else:
            self.dist_cost.disable_cost()
            self.cspace_convergence.disable_cost()

    def _format_multi_arm_ee_data(self, state: KinematicModelState, num_arms: int):
        """Format end-effector data for multi-arm pose cost.
        
        Constructs 4D tensors [batch, horizon, num_arms, 3/4] from available link poses.
        For dual-arm setup, typically looks for 'left_panda_hand' and 'right_panda_hand'.
        If a link is missing, fills with zeros or uses available data.
        
        Args:
            state: Kinematic model state containing link poses
            num_arms: Number of arms expected by the multi-arm pose cost
            
        Returns:
            Tuple of (ee_pos_4d, ee_quat_4d) tensors in 4D format
        """
        # Get batch and horizon dimensions from state
        batch_size = state.ee_pos_seq.shape[0] if state.ee_pos_seq is not None else 1
        horizon = state.ee_pos_seq.shape[1] if state.ee_pos_seq is not None else 1
        
        # Initialize 4D tensors for multi-arm end-effector data
        device = self.tensor_args.device
        dtype = self.tensor_args.dtype
        
        ee_pos_4d = torch.zeros((batch_size, horizon, num_arms, 3), device=device, dtype=dtype)
        ee_quat_4d = torch.zeros((batch_size, horizon, num_arms, 4), device=device, dtype=dtype)
        
        # Default quaternion (identity: w=1, x=0, y=0, z=0)
        ee_quat_4d[:, :, :, 0] = 1.0  # Set w component to 1
        
        # Map arm indices to link names - adjust these based on your robot configuration
        arm_link_mapping = []
        if num_arms == 2:
            # Dual-arm setup
            arm_link_mapping = ['left_panda_hand', 'right_panda_hand']
        else:
            # Generic multi-arm setup
            arm_link_mapping = [f'arm_{i}_ee' for i in range(num_arms)]
        
        # Debug: Print available link poses (only first few times)
        if not hasattr(self, '_debug_link_count'):
            self._debug_link_count = 0
        self._debug_link_count += 1
        
        if self._debug_link_count <= 3:  # Print first 3 times
            link_poses = state.link_pose
            if link_poses is not None:
                print(f"\n=== Multi-arm Debug {self._debug_link_count} ===")
                print(f"Available link poses: {list(link_poses.keys())}")
                print(f"Expected arm links: {arm_link_mapping}")
                print(f"Primary EE data shape: pos={state.ee_pos_seq.shape if state.ee_pos_seq is not None else None}, quat={state.ee_quat_seq.shape if state.ee_quat_seq is not None else None}")
        
        # Fill in data for available links
        link_poses = state.link_pose
        if link_poses is not None:
            for arm_idx, link_name in enumerate(arm_link_mapping):
                if link_name in link_poses:
                    # Extract pose data for this arm
                    link_pose = link_poses[link_name]
                    ee_pos_4d[:, :, arm_idx, :] = link_pose.position
                    ee_quat_4d[:, :, arm_idx, :] = link_pose.quaternion
                    
                    if self._debug_link_count <= 3:
                        print(f"Arm {arm_idx} ({link_name}): Found link pose data")
                        print(f"  Position shape: {link_pose.position.shape}")
                        print(f"  Position sample: {link_pose.position[0, 0, :] if link_pose.position.numel() > 3 else link_pose.position}")
                else:
                    # Link not found - try alternative approaches
                    if arm_idx == 0 and state.ee_pos_seq is not None and state.ee_quat_seq is not None:
                        # Use primary end-effector data for first arm
                        ee_pos_4d[:, :, arm_idx, :] = state.ee_pos_seq
                        ee_quat_4d[:, :, arm_idx, :] = state.ee_quat_seq
                        
                        if self._debug_link_count <= 3:
                            print(f"Arm {arm_idx} ({link_name}): Using primary EE data as fallback")
                    elif hasattr(self, 'kinematics') and hasattr(self.kinematics, 'get_state'):
                        # Try to compute the missing end-effector pose using kinematics
                        try:
                            # Get current joint positions
                            current_joints = state.state_seq.position
                            if current_joints is not None:
                                # Use kinematics to compute pose for the missing link
                                kin_state = self.kinematics.get_state(current_joints)
                                if hasattr(kin_state, 'link_pose') and link_name in kin_state.link_pose:
                                    computed_pose = kin_state.link_pose[link_name]
                                    ee_pos_4d[:, :, arm_idx, :] = computed_pose.position
                                    ee_quat_4d[:, :, arm_idx, :] = computed_pose.quaternion
                                    
                                    if self._debug_link_count <= 3:
                                        print(f"Arm {arm_idx} ({link_name}): Computed pose using kinematics")
                                        print(f"  Computed position sample: {computed_pose.position[0, 0, :] if computed_pose.position.numel() > 3 else computed_pose.position}")
                                else:
                                    if self._debug_link_count <= 3:
                                        print(f"Arm {arm_idx} ({link_name}): Kinematics computation failed - using zeros/identity")
                            else:
                                if self._debug_link_count <= 3:
                                    print(f"Arm {arm_idx} ({link_name}): No joint positions available - using zeros/identity")
                        except Exception as e:
                            if self._debug_link_count <= 3:
                                print(f"Arm {arm_idx} ({link_name}): Kinematics computation error: {e} - using zeros/identity")
                    else:
                        if self._debug_link_count <= 3:
                            print(f"Arm {arm_idx} ({link_name}): Missing - using zeros/identity")
                    # If all fallbacks fail, leave as zeros/identity quaternion (already initialized)
        
        if self._debug_link_count <= 3:
            print(f"Final 4D tensor shapes: pos={ee_pos_4d.shape}, quat={ee_quat_4d.shape}")
            print("=== End Multi-arm Debug ===\n")
        
        return ee_pos_4d, ee_quat_4d

    def get_pose_costs(
        self,
        include_link_pose: bool = False,
        include_convergence: bool = True,
        only_convergence: bool = False,
    ):
        if only_convergence:
            return [self.pose_convergence]
        pose_costs = [self.goal_cost]
        if include_convergence:
            pose_costs += [self.pose_convergence]
        if include_link_pose:
            log_error("Not implemented yet")
        return pose_costs

    def update_pose_cost_metric(
        self,
        metric: PoseCostMetric,
    ):
        pose_costs = self.get_pose_costs(
            include_link_pose=metric.include_link_pose, include_convergence=False
        )
        for p in pose_costs:
            p.update_metric(metric, update_offset_waypoint=True)

        pose_costs = self.get_pose_costs(only_convergence=True)
        for p in pose_costs:
            p.update_metric(metric, update_offset_waypoint=False)

    # def set_dynamic_obs_coll_predictor(self, predictor: DynamicObsCollPredictor):
    #     self._dynamic_obs_coll_predictor = predictor
    
    # def get_dynamic_obs_coll_predictor(self) -> Optional[DynamicObsCollPredictor]:
    #     return self._dynamic_obs_coll_predictor

    def _update_live_plot_reacher(self, cost_list):
        """Update live plot of cost values in real-time with comprehensive ArmReacher labeling"""
        
        
        # Initialize plotting components if not already done
        if not hasattr(self, '_plot_initialized'):
            self._plot_initialized = True
            self._cost_histories = {}  # Dictionary to store history for each cost component
            self._cost_lines = {}  # Dictionary to store plot lines for each cost component
            self._plot_counter = 0  # Counter for plotting frequency
            self._plot_every_k = 5  # Plot every 5 iterations to save resources
            
            # Set up the figure and axis
            plt.ion()  # Turn on interactive mode
            self._fig, self._ax = plt.subplots(1, 1, figsize=(16, 10))
            self._fig.suptitle('Real-time Cost Monitoring - ArmReacher All Components')
            
            self._ax.set_title('All Cost Components Over Time (Base + Reacher + Custom)')
            self._ax.set_xlabel('Iteration')
            self._ax.set_ylabel('Cost Value')
            self._ax.grid(True)
            
            plt.tight_layout()
            plt.show(block=False)
        
        # Increment counter and check if we should plot this iteration
        self._plot_counter += 1
        if self._plot_counter % self._plot_every_k != 0:
            return  # Skip this iteration
        
        
        # Dynamic cost labeling based on what's actually enabled
        cost_labels_dynamic = []
        
        # Check which base costs are enabled and create appropriate labels
        base_cost_count = 0
        if hasattr(self, 'bound_cost') and self.bound_cost.enabled:
            cost_labels_dynamic.append('Bound Cost')
            base_cost_count += 1
        if hasattr(self, 'stop_cost') and self.stop_cost.enabled:
            cost_labels_dynamic.append('Stop Cost')
            base_cost_count += 1
        if hasattr(self, 'robot_self_collision_cost') and self.robot_self_collision_cost.enabled:
            cost_labels_dynamic.append('Self Collision')
            base_cost_count += 1
        if hasattr(self, 'primitive_collision_cost') and self.primitive_collision_cost.enabled:
            cost_labels_dynamic.append('Primitive Collision')
            base_cost_count += 1
        
        # Add custom arm_base costs with their class names
        if hasattr(self, '_custom_arm_base_costs'):
            for cost_name, cost_instance in self._custom_arm_base_costs.items():
                if cost_instance.enabled:
                    # Extract class name from the cost instance
                    class_name = cost_instance.__class__.__name__
                    cost_labels_dynamic.append(f'Custom Base: {class_name}')
                    base_cost_count += 1
        
        # # Check for dynamic obstacles (added after custom arm_base costs)
        # if hasattr(self, 'dynamic_obs_cost') and self._dynamic_obs_coll_predictor is not None:
        #     cost_labels_dynamic.append('Dynamic Obstacles')
        #     base_cost_count += 1
        
        if hasattr(self, 'manipulability_cost') and self.manipulability_cost.enabled:
            cost_labels_dynamic.append('Manipulability')
            base_cost_count += 1
            
        # Add ArmReacher specific costs
        if hasattr(self, 'goal_cost') and self.goal_cost.enabled:
            cost_labels_dynamic.append('Goal/Pose Cost')
        if hasattr(self, '_link_pose_costs'):
            for link_name in self._link_pose_costs:
                if self._link_pose_costs[link_name].enabled:
                    cost_labels_dynamic.append(f'Link Pose ({link_name})')
        if hasattr(self, 'dist_cost') and self.dist_cost.enabled:
            cost_labels_dynamic.append('CSpace Cost')
        if hasattr(self, 'straight_line_cost') and self.straight_line_cost.enabled:
            cost_labels_dynamic.append('Straight Line Cost')
        if hasattr(self, 'zero_acc_cost') and self.zero_acc_cost.enabled:
            cost_labels_dynamic.append('Zero Acceleration')
        if hasattr(self, 'zero_jerk_cost') and self.zero_jerk_cost.enabled:
            cost_labels_dynamic.append('Zero Jerk')
        if hasattr(self, 'zero_vel_cost') and self.zero_vel_cost.enabled:
            cost_labels_dynamic.append('Zero Velocity')
        
        # Add custom arm_reacher costs with their class names
        if hasattr(self, '_custom_arm_reacher_costs'):
            for cost_name, cost_instance in self._custom_arm_reacher_costs.items():
                if cost_instance.enabled:
                    # Extract class name from the cost instance
                    class_name = cost_instance.__class__.__name__
                    cost_labels_dynamic.append(f'Custom Reacher: {class_name}')
        
        # Colors for plotting
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black', 'darkred', 'darkgreen', 'darkblue']
        
        # Process each cost component
        active_costs = []
        for i, cost_tensor in enumerate(cost_list):
            if cost_tensor is not None:
                # Get cost label
                if i < len(cost_labels_dynamic):
                    label = cost_labels_dynamic[i]
                else:
                    label = f'Unknown_Cost_{i}'
                
                # Calculate mean of this cost component
                cost_mean = torch.mean(cost_tensor).cpu().numpy().item()
                
                # Track all costs, even very small ones, but highlight significant ones
                active_costs.append((label, cost_mean, i))
                    
                # Initialize history for this component if not exists
                if label not in self._cost_histories:
                    self._cost_histories[label] = deque(maxlen=200)  # Keep last 200 plot points
                    color = colors[i % len(colors)]
                    
                    # Special styling for different cost types
                    if 'Goal' in label or 'Pose' in label:
                        linewidth = 3
                        marker = 'o'
                        markersize = 4
                    elif 'Custom' in label:
                        linewidth = 2.5
                        marker = 's'  # Square markers for custom costs
                        markersize = 4
                    else:
                        linewidth = 2
                        marker = 'o'
                        markersize = 3
                    
                    self._cost_lines[label], = self._ax.plot([], [], color=color, label=label, 
                                                           linewidth=linewidth, marker=marker, markersize=markersize)
                
                # Add current value to history
                self._cost_histories[label].append(cost_mean)
        
        # Print active costs for debugging (first few times)
        # if self._plot_counter <= self._plot_every_k * 2:  # First 2 plot updates
        #     print(f"Active cost components: {[(label, f'{val:.6f}') for label, val, _ in active_costs]}")
        #     custom_costs = [label for label, _, _ in active_costs if 'Custom' in label]
        #     if custom_costs:
        #         print(f"Custom costs detected: {custom_costs}")
            # Only show custom costs debug info, not all goal/pose costs
        
        # Update all plot lines
        for label, history in self._cost_histories.items():
            if len(history) > 0:
                x_data = list(range(len(history)))
                y_data = list(history)
                self._cost_lines[label].set_data(x_data, y_data)
        
        # Update plot limits and legend
        if self._cost_histories:
            # Get all x and y data for proper scaling
            all_x_data = []
            all_y_data = []
            for history in self._cost_histories.values():
                if len(history) > 0:
                    all_x_data.extend(range(len(history)))
                    all_y_data.extend(history)
            
            if all_x_data and all_y_data:
                self._ax.set_xlim(0, max(all_x_data) + 1)
                y_min, y_max = min(all_y_data), max(all_y_data)
                y_range = y_max - y_min
                if y_range > 0:
                    self._ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
                else:
                    self._ax.set_ylim(y_min - 0.1, y_max + 0.1)
        
        # Update legend (only when new components are added)
        if not hasattr(self, '_legend_updated') or len(self._cost_lines) != getattr(self, '_last_legend_count', 0):
            self._ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
            self._legend_updated = True
            self._last_legend_count = len(self._cost_lines)
        
        # Refresh the plot
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        
        # Optional: Save periodic snapshots (less frequent)
        if self._save_plots and self._plot_counter % (self._plot_every_k * 20) == 0:  # Every 100 actual iterations
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if not hasattr(self, '_cost_plots_dir'):
                self._cost_plots_dir = os.path.join(os.getcwd(),'tmp_artifacts', 'cost_plots', timestamp)
                os.makedirs(self._cost_plots_dir, exist_ok=False)
            self._fig.savefig(os.path.join(self._cost_plots_dir, f'costs_iter_{self._plot_counter}.png'), dpi=150, bbox_inches='tight')
            print(f"Saved plot snapshot at iteration {self._plot_counter}")

    def set_plot_frequency(self, k: int):
        """Set how often to update the live plot (every k iterations)
        
        Args:
            k (int): Update plot every k iterations
        """
        if hasattr(self, '_plot_every_k'):
            self._plot_every_k = k
            print(f"Plot frequency set to every {k} iterations")
        else:
            print("Live plotting not initialized yet. This will take effect when plotting starts.")


@get_torch_jit_decorator()
def cat_sum_reacher(tensor_list: List[torch.Tensor]):
    cat_tensor = torch.sum(torch.stack(tensor_list, dim=0), dim=0)
    return cat_tensor


@get_torch_jit_decorator()
def cat_sum_horizon_reacher(tensor_list: List[torch.Tensor]):
    cat_tensor = torch.sum(torch.stack(tensor_list, dim=0), dim=(0, -1))
    return cat_tensor