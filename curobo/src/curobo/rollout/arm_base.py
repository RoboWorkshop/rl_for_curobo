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
from abc import abstractmethod
from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Union
import datetime
from curobo.rollout.cost.custom.custom_cost import CustomCost
import matplotlib.pyplot as plt

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.geom.sdf.utils import create_collision_checker
from curobo.geom.sdf.world import WorldCollision, WorldCollisionConfig
from curobo.geom.types import WorldConfig
from curobo.rollout.cost.bound_cost import BoundCost, BoundCostConfig
from curobo.rollout.cost.cost_base import CostBase, CostConfig
from curobo.rollout.cost.dist_cost import DistCost, DistCostConfig
from curobo.rollout.cost.manipulability_cost import ManipulabilityCost, ManipulabilityCostConfig
from curobo.rollout.cost.primitive_collision_cost import (
    PrimitiveCollisionCost,
    PrimitiveCollisionCostConfig,
)
from curobo.rollout.cost.self_collision_cost import SelfCollisionCost, SelfCollisionCostConfig
from curobo.rollout.cost.stop_cost import StopCost, StopCostConfig
from curobo.rollout.dynamics_model.kinematic_model import (
    KinematicModel,
    KinematicModelConfig,
    KinematicModelState,
)
from curobo.rollout.rollout_base import Goal, RolloutBase, RolloutConfig, RolloutMetrics, Trajectory
from curobo.types.base import TensorDeviceType
from curobo.types.robot import CSpaceConfig, RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util.tensor_util import cat_sum, cat_sum_horizon

from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import DynamicObsCollPredictor

import importlib

@dataclass
class ArmCostConfig:
    bound_cfg: Optional[BoundCostConfig] = None
    null_space_cfg: Optional[DistCostConfig] = None
    manipulability_cfg: Optional[ManipulabilityCostConfig] = None
    stop_cfg: Optional[StopCostConfig] = None
    self_collision_cfg: Optional[SelfCollisionCostConfig] = None
    primitive_collision_cfg: Optional[PrimitiveCollisionCostConfig] = None
    custom_cfg: Optional[Dict] = None  # Add custom cost terms configuration

    @staticmethod
    def _get_base_keys():
        """
        Returns:
            Dictionary of base (cost terms of base calss ArmBase) cost keys
        """
        k_list = {
            "null_space_cfg": DistCostConfig,
            "manipulability_cfg": ManipulabilityCostConfig,
            "stop_cfg": StopCostConfig,
            "self_collision_cfg": SelfCollisionCostConfig,
            "bound_cfg": BoundCostConfig,
        }
        prefix_module = CustomCost.get_modules_path_prefix() + 'base' + '.' 
        modules = importlib.import_module(prefix_module)
        for module_name in modules: 
            import module.Cost as Cost
            import module.CostConfig as CostConfig
            k_list[module_name] = Cost
            k_list[module_name + 'Config'] = CostConfig

    @staticmethod
    def _load_custom_cost_class(module_path: str, class_name: str):
        """Dynamically load a custom cost class."""
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            log_error(f"Failed to load custom cost class {class_name} from {module_path}: {e}")
            return None

    @staticmethod
    def _discover_custom_costs_in_directory(directory_type: str, tensor_args: TensorDeviceType) -> Dict:
        """
        Automatically discover custom cost files in arm_base or arm_reacher directories.
        
        Args:
            directory_type: Either "arm_base" or "arm_reacher"
            tensor_args: Tensor device configuration
            
        Returns:
            Dictionary of discovered custom costs
        """
        import os
        import importlib.util
        import inspect
        from curobo.rollout.cost.cost_base import CostBase, CostConfig
        
        discovered_costs = {}
        
        # Get the path to the custom cost directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        custom_dir = os.path.join(current_dir, "cost", "custom", directory_type)
        custom_dir = os.path.normpath(custom_dir)
        
        if not os.path.exists(custom_dir):
            log_info(f"Custom cost directory not found: {custom_dir}")
            return discovered_costs
            
        # Scan for Python files (excluding __init__.py)
        for filename in os.listdir(custom_dir):
            if filename.endswith('.py') and filename != '__init__.py':
                module_name = filename[:-3]  # Remove .py extension
                file_path = os.path.join(custom_dir, filename)
                
                try:
                    # Load the module
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec is None or spec.loader is None:
                        continue
                        
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find cost classes in the module
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, CostBase) and 
                            obj is not CostBase):
                            
                            # Look for corresponding config class
                            config_class_name = name + "Config"
                            config_class = getattr(module, config_class_name, None)
                            
                            if config_class and issubclass(config_class, CostConfig):
                                # Create default configuration
                                try:
                                    default_config = config_class(
                                        weight=1.0,  # Default weight
                                        terminal=False,  # Default terminal setting
                                        tensor_args=tensor_args
                                    )
                                    
                                    # Store the discovered cost
                                    cost_key = f"auto_{module_name}_{name}"
                                    discovered_costs[cost_key] = {
                                        "cost_class": obj,
                                        "cost_config": default_config,
                                        "source_file": filename,
                                        "auto_discovered": True
                                    }
                                    
                                    log_info(f"Auto-discovered custom cost: {cost_key} from {filename}")
                                    
                                except Exception as e:
                                    log_warn(f"Failed to create default config for {name}: {e}")
                                    
                except Exception as e:
                    log_warn(f"Failed to load custom cost module {filename}: {e}")
                    
        return discovered_costs

    @staticmethod
    def _parse_custom_costs(custom_dict: Dict, _num_particles_rollout_full: int = -1) -> Dict:
        """Parse custom cost configurations for arm_base or arm_reacher."""
        custom_costs = {}        
        # # Check if there are any explicitly configured custom costs
        # has_explicit_arm_base_costs = "arm_base" in custom_dict and custom_dict["arm_base"]
        # has_explicit_arm_reacher_costs = "arm_reacher" in custom_dict and custom_dict["arm_reacher"]
        
        for type_of_custom_cost in ['base', 'reacher']:
            type_of_custom_cost_key = f"arm_{type_of_custom_cost}"
            has_explicit_type_of_custom_cost = type_of_custom_cost_key in custom_dict and custom_dict[type_of_custom_cost_key]

            if has_explicit_type_of_custom_cost:
                custom_costs[type_of_custom_cost_key] = {}
                for cost_name, cost_config in custom_dict["arm_base"].items():
                    if isinstance(cost_config, dict):
                        # Extract class information (use get() instead of pop() to avoid modifying original dict)
                        # module_path = cost_config.get("module_path", None)
                        module_path = CustomCost.get_modules_path_prefix() + type_of_custom_cost + '.' + cost_name
                        # class_name = cost_config.get("class_name", None)
                        # config_class_name = cost_config.get("config_class_name", None)
                         
                        
                        if module_path and class_name:
                            # Create a copy of the config dict without the class info fields
                            config_params = {k: v for k, v in cost_config.items() 
                                        if k not in ['module_path', 'class_name', 'config_class_name']}
                            
                            # Add the _num_particles_rollout_full parameter to config_params
                            config_params['_num_particles_rollout_full'] = _num_particles_rollout_full
                            config_params['_horizon_rollout_full'] = -1  # Default value, should be set elsewhere
                            
                            
                            # Load the custom cost class
                            cost_class = ArmCostConfig._load_custom_cost_class(module_path, class_name)
                            if cost_class:
                                # Load config class if specified
                                if config_class_name:
                                    config_class = ArmCostConfig._load_custom_cost_class(module_path, config_class_name)
                                    if config_class:
                                        cost_cfg = config_class(**config_params, tensor_args=tensor_args)
                                    else:
                                        # Fallback to basic CostConfig
                                        cost_cfg = CostConfig(**config_params, tensor_args=tensor_args)
                                else:
                                    # Use basic CostConfig
                                    cost_cfg = CostConfig(**config_params, tensor_args=tensor_args)
                                
                                custom_costs[type_of_custom_cost_key][cost_name] = {
                                    "cost_class": cost_class,
                                    "cost_config": cost_cfg
                                }
                    


        # Process explicitly configured arm_base custom costs
        if has_explicit_arm_base_costs:
            custom_costs["arm_base"] = {}
            for cost_name, cost_config in custom_dict["arm_base"].items():
                if isinstance(cost_config, dict):
                    # Extract class information (use get() instead of pop() to avoid modifying original dict)
                    module_path = cost_config.get("module_path", None)
                    class_name = cost_config.get("class_name", None)
                    config_class_name = cost_config.get("config_class_name", None)
                    
                    
                    if module_path and class_name:
                        # Create a copy of the config dict without the class info fields
                        config_params = {k: v for k, v in cost_config.items() 
                                       if k not in ['module_path', 'class_name', 'config_class_name']}
                        
                        # Add the _num_particles_rollout_full parameter to config_params
                        config_params['_num_particles_rollout_full'] = _num_particles_rollout_full
                        config_params['_horizon_rollout_full'] = -1  # Default value, should be set elsewhere
                        
                        
                        # Load the custom cost class
                        cost_class = ArmCostConfig._load_custom_cost_class(module_path, class_name)
                        if cost_class:
                            # Load config class if specified
                            if config_class_name:
                                config_class = ArmCostConfig._load_custom_cost_class(module_path, config_class_name)
                                if config_class:
                                    cost_cfg = config_class(**config_params, tensor_args=tensor_args)
                                else:
                                    # Fallback to basic CostConfig
                                    cost_cfg = CostConfig(**config_params, tensor_args=tensor_args)
                            else:
                                # Use basic CostConfig
                                cost_cfg = CostConfig(**config_params, tensor_args=tensor_args)
                            
                            custom_costs["arm_base"][cost_name] = {
                                "cost_class": cost_class,
                                "cost_config": cost_cfg
                            }

        
        # Auto-discover arm_base costs only if enabled AND no explicit arm_base costs are configured
        if enable_auto_discovery and not has_explicit_arm_base_costs:
            if "arm_base" not in custom_costs:
                custom_costs["arm_base"] = {}
            discovered_arm_base = ArmCostConfig._discover_custom_costs_in_directory("arm_base", tensor_args)
            # Only add discovered costs that aren't already explicitly configured
            for cost_name, cost_info in discovered_arm_base.items():
                if cost_name not in custom_costs["arm_base"]:
                    custom_costs["arm_base"][cost_name] = cost_info
        
        # Process explicitly configured arm_reacher custom costs
        if has_explicit_arm_reacher_costs:
            custom_costs["arm_reacher"] = {}
            for cost_name, cost_config in custom_dict["arm_reacher"].items():
                if isinstance(cost_config, dict):
                    # Extract class information (use get() instead of pop() to avoid modifying original dict)
                    module_path = cost_config.get("module_path", None)
                    class_name = cost_config.get("class_name", None)
                    config_class_name = cost_config.get("config_class_name", None)
                    
                    if module_path and class_name:
                        # Create a copy of the config dict without the class info fields
                        config_params = {k: v for k, v in cost_config.items() 
                                       if k not in ['module_path', 'class_name', 'config_class_name']}
                        
                        # Add the _num_particles_rollout_full parameter to config_params
                        config_params['_num_particles_rollout_full'] = _num_particles_rollout_full
                        config_params['_horizon_rollout_full'] = -1  # Default value, should be set elsewhere
                        
                        # Load the custom cost class
                        cost_class = ArmCostConfig._load_custom_cost_class(module_path, class_name)
                        if cost_class:
                            # Load config class if specified
                            if config_class_name:
                                config_class = ArmCostConfig._load_custom_cost_class(module_path, config_class_name)
                                if config_class:
                                    cost_cfg = config_class(**config_params, tensor_args=tensor_args)
                                else:
                                    # Fallback to basic CostConfig
                                    cost_cfg = CostConfig(**config_params, tensor_args=tensor_args)
                            else:
                                # Use basic CostConfig
                                cost_cfg = CostConfig(**config_params, tensor_args=tensor_args)
                            
                            custom_costs["arm_reacher"][cost_name] = {
                                "cost_class": cost_class,
                                "cost_config": cost_cfg
                            }
        
        # Auto-discover arm_reacher costs only if enabled AND no explicit arm_reacher costs are configured
        if enable_auto_discovery and not has_explicit_arm_reacher_costs:
            if "arm_reacher" not in custom_costs:
                custom_costs["arm_reacher"] = {}
            discovered_arm_reacher = ArmCostConfig._discover_custom_costs_in_directory("arm_reacher", tensor_args)
            # Only add discovered costs that aren't already explicitly configured
            for cost_name, cost_info in discovered_arm_reacher.items():
                if cost_name not in custom_costs["arm_reacher"]:
                    custom_costs["arm_reacher"][cost_name] = cost_info
        
        return custom_costs

    @staticmethod
    def from_dict(
        data_dict: Dict,
        robot_config: RobotConfig,
        world_coll_checker: Optional[WorldCollision] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        enable_auto_discovery: bool = False,
        _num_particles_rollout_full: int = -1,
    ):
        k_list = ArmCostConfig._get_base_keys()
        data = ArmCostConfig._get_formatted_dict(
            data_dict,
            k_list,
            robot_config,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
            _num_particles_rollout_full=_num_particles_rollout_full,
        )
        
        # Handle custom costs with auto-discovery
        custom_dict = data_dict.get("custom", {})
        data["custom_cfg"] = ArmCostConfig._parse_custom_costs(
            custom_dict, 
            tensor_args, 
            enable_auto_discovery=enable_auto_discovery,
            _num_particles_rollout_full=_num_particles_rollout_full,
        )
        
        return ArmCostConfig(**data)

    @staticmethod
    def _get_formatted_dict(
        data_dict: Dict,
        cost_key_list: Dict,
        robot_config: RobotConfig,
        world_coll_checker: Optional[WorldCollision] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        _num_particles_rollout_full: int = -1,
    ):
        data = {}
        for k in cost_key_list:
            if k in data_dict:
                data[k] = cost_key_list[k](**data_dict[k], tensor_args=tensor_args)
        if "primitive_collision_cfg" in data_dict and world_coll_checker is not None:
            data["primitive_collision_cfg"] = PrimitiveCollisionCostConfig(
                **data_dict["primitive_collision_cfg"],
                world_coll_checker=world_coll_checker,
                tensor_args=tensor_args
            )

        return data


@dataclass
class ArmBaseConfig(RolloutConfig):
    model_cfg: Optional[KinematicModelConfig] = None
    cost_cfg: Optional[ArmCostConfig] = None
    constraint_cfg: Optional[ArmCostConfig] = None
    convergence_cfg: Optional[ArmCostConfig] = None
    world_coll_checker: Optional[WorldCollision] = None
    _num_particles_rollout_full: int = -1
    

    @staticmethod
    def model_from_dict(
        model_data_dict: Dict,
        robot_cfg: RobotConfig,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        return KinematicModelConfig.from_dict(model_data_dict, robot_cfg, tensor_args=tensor_args)

    @staticmethod
    def cost_from_dict(
        cost_data_dict: Dict,
        robot_cfg: RobotConfig,
        world_coll_checker: Optional[WorldCollision] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        enable_auto_discovery: bool = False,
        _num_particles_rollout_full: int = -1,
    ):
        return ArmCostConfig.from_dict(
            cost_data_dict,
            robot_cfg,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
            enable_auto_discovery=enable_auto_discovery,
            _num_particles_rollout_full=_num_particles_rollout_full,
        )

    @staticmethod
    def world_coll_checker_from_dict(
        world_coll_checker_dict: Optional[Dict] = None,
        world_model_dict: Optional[Union[WorldConfig, Dict]] = None,
        world_coll_checker: Optional[WorldCollision] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        # TODO: Check which type of collision checker and load that.
        if (
            world_coll_checker is None
            and world_model_dict is not None
            and world_coll_checker_dict is not None
        ):
            world_coll_cfg = WorldCollisionConfig.load_from_dict(
                world_coll_checker_dict, world_model_dict, tensor_args
            )

            world_coll_checker = create_collision_checker(world_coll_cfg)
        else:
            log_info("*******USING EXISTING COLLISION CHECKER***********")
        return world_coll_checker

    @classmethod
    @profiler.record_function("arm_base_config/from_dict")
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
        """Create ArmBase class from dictionary

        NOTE: We declare this as a classmethod to allow for derived classes to use it.

        Args:
            robot_cfg (Union[Dict, RobotConfig]): _description_
            model_data_dict (Dict): _description_
            cost_data_dict (Dict): _description_
            constraint_data_dict (Dict): _description_
            convergence_data_dict (Dict): _description_
            world_coll_checker_dict (Optional[Dict], optional): _description_. Defaults to None.
            world_model_dict (Optional[Dict], optional): _description_. Defaults to None.
            world_coll_checker (Optional[WorldCollision], optional): _description_. Defaults to None.
            tensor_args (TensorDeviceType, optional): _description_. Defaults to TensorDeviceType().
            enable_auto_discovery (bool, optional): Enable auto-discovery of custom costs. Defaults to False.
            _num_particles_rollout_full (int, optional): Number of particles for rollout. Defaults to -1.
        Returns:
            _type_: _description_
        """
        if isinstance(robot_cfg, dict):
            robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
        world_coll_checker = cls.world_coll_checker_from_dict(
            world_coll_checker_dict, world_model_dict, world_coll_checker, tensor_args
        )
        model = cls.model_from_dict(model_data_dict, robot_cfg, tensor_args=tensor_args)
        cost = cls.cost_from_dict(
            cost_data_dict,
            robot_cfg,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
            enable_auto_discovery=enable_auto_discovery,
            _num_particles_rollout_full=_num_particles_rollout_full,
        )
        constraint = cls.cost_from_dict(
            constraint_data_dict,
            robot_cfg,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
            enable_auto_discovery=enable_auto_discovery,
            _num_particles_rollout_full=_num_particles_rollout_full,
        )
        convergence = cls.cost_from_dict(
            convergence_data_dict,
            robot_cfg,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
            enable_auto_discovery=enable_auto_discovery,
            _num_particles_rollout_full=_num_particles_rollout_full,
        )
        return cls(
            model_cfg=model,
            cost_cfg=cost,
            constraint_cfg=constraint,
            convergence_cfg=convergence,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
            _num_particles_rollout_full=_num_particles_rollout_full,
        )


class ArmBase(RolloutBase, ArmBaseConfig):
    """
    This rollout function is for reaching a cartesian pose for a robot
    """

    @profiler.record_function("arm_base/init")
    def __init__(self, config: Optional[ArmBaseConfig] = None):
        """Initialize ArmBase."""
        if config is not None:
            ArmBaseConfig.__init__(self, **vars(config))
        RolloutBase.__init__(self)
        # self._num_particles_rollout = -1  # Initialize this to avoid recursion
        self._init_after_config_load()
        # self._dynamic_obs_coll_predictor: Optional[DynamicObsCollPredictor] = None

    # def set_dynamic_obs_coll_predictor(self, predictor: DynamicObsCollPredictor):
    #     self._dynamic_obs_coll_predictor = predictor
    
    # def get_dynamic_obs_coll_predictor(self) -> Optional[DynamicObsCollPredictor]:
    #     return self._dynamic_obs_coll_predictor
    
    @profiler.record_function("arm_base/init_after_config_load")
    def _init_after_config_load(self):
        # self.current_state = None
        # self.retract_state = None
        self._goal_buffer = Goal()
        self._goal_idx_update = True
        # Create the dynamical system used for rollouts
        self.dynamics_model = KinematicModel(self.model_cfg)

        self.n_dofs = self.dynamics_model.n_dofs
        self.traj_dt = self.dynamics_model.traj_dt
        if self.cost_cfg.bound_cfg is not None:
            self.cost_cfg.bound_cfg.set_bounds(
                self.dynamics_model.get_state_bounds(),
                teleport_mode=self.dynamics_model.teleport_mode,
            )
            self.cost_cfg.bound_cfg.cspace_distance_weight = (
                self.dynamics_model.cspace_distance_weight
            )
            self.cost_cfg.bound_cfg.state_finite_difference_mode = (
                self.dynamics_model.state_finite_difference_mode
            )
            self.cost_cfg.bound_cfg.update_vec_weight(self.dynamics_model.null_space_weight)

            if self.cost_cfg.null_space_cfg is not None:
                self.cost_cfg.bound_cfg.null_space_weight = self.cost_cfg.null_space_cfg.weight
                log_warn(
                    "null space cost is deprecated, use null_space_weight in bound cost instead"
                )
            self.cost_cfg.bound_cfg.dof = self.n_dofs
            self.bound_cost = BoundCost(self.cost_cfg.bound_cfg)

        if self.cost_cfg.manipulability_cfg is not None:
            self.manipulability_cost = ManipulabilityCost(self.cost_cfg.manipulability_cfg)

        if self.cost_cfg.stop_cfg is not None:
            self.cost_cfg.stop_cfg.horizon = self.dynamics_model.horizon
            self.cost_cfg.stop_cfg.dt_traj_params = self.dynamics_model.dt_traj_params
            self.stop_cost = StopCost(self.cost_cfg.stop_cfg)
        self._goal_buffer.retract_state = self.retract_state
        if self.cost_cfg.primitive_collision_cfg is not None:
            self.primitive_collision_cost = PrimitiveCollisionCost(
                self.cost_cfg.primitive_collision_cfg
            )
            if self.dynamics_model.robot_model.total_spheres == 0:
                self.primitive_collision_cost.disable_cost()

        if self.cost_cfg.self_collision_cfg is not None:
            self.cost_cfg.self_collision_cfg.self_collision_kin_config = (
                self.dynamics_model.robot_model.get_self_collision_config()
            )
            self.robot_self_collision_cost = SelfCollisionCost(self.cost_cfg.self_collision_cfg)
            if self.dynamics_model.robot_model.total_spheres == 0:
                self.robot_self_collision_cost.disable_cost()

        # setup constraint terms:
        if self.constraint_cfg.primitive_collision_cfg is not None:
            self.primitive_collision_constraint = PrimitiveCollisionCost(
                self.constraint_cfg.primitive_collision_cfg
            )
            if self.dynamics_model.robot_model.total_spheres == 0:
                self.primitive_collision_constraint.disable_cost()

        if self.constraint_cfg.self_collision_cfg is not None:
            self.constraint_cfg.self_collision_cfg.self_collision_kin_config = (
                self.dynamics_model.robot_model.get_self_collision_config()
            )
            self.robot_self_collision_constraint = SelfCollisionCost(
                self.constraint_cfg.self_collision_cfg
            )

            if self.dynamics_model.robot_model.total_spheres == 0:
                self.robot_self_collision_constraint.disable_cost()

        self.constraint_cfg.bound_cfg.set_bounds(
            self.dynamics_model.get_state_bounds(), teleport_mode=self.dynamics_model.teleport_mode
        )
        self.constraint_cfg.bound_cfg.cspace_distance_weight = (
            self.dynamics_model.cspace_distance_weight
        )
        self.cost_cfg.bound_cfg.state_finite_difference_mode = (
            self.dynamics_model.state_finite_difference_mode
        )
        self.cost_cfg.bound_cfg.dof = self.n_dofs
        self.constraint_cfg.bound_cfg.dof = self.n_dofs
        self.bound_constraint = BoundCost(self.constraint_cfg.bound_cfg)

        if self.convergence_cfg.null_space_cfg is not None:
            self.convergence_cfg.null_space_cfg.dof = self.n_dofs
            self.null_convergence = DistCost(self.convergence_cfg.null_space_cfg)

        # Initialize custom costs for arm_base
        self._custom_arm_base_costs = {}
        if self.cost_cfg.custom_cfg is not None and "arm_base" in self.cost_cfg.custom_cfg:
            for cost_name, cost_info in self.cost_cfg.custom_cfg["arm_base"].items():
                cost_class = cost_info["cost_class"]
                cost_config = cost_info["cost_config"]
                
                # Pass rollout parameters to custom cost config if they have safe defaults
                # Note: num_particles is not available during initialization, will be set later by optimizer
                cost_config._horizon_rollout_full = self.horizon
                cost_config._num_particles_rollout_full = self._num_particles_rollout_full
                # Instantiate the custom cost
                custom_cost_instance = cost_class(cost_config)
                self._custom_arm_base_costs[cost_name] = custom_cost_instance
                log_info(f"Successfully loaded custom arm_base cost: {cost_name}")
                    
                
        # set start state:
        start_state = torch.randn(
            (1, self.dynamics_model.d_state), **(self.tensor_args.as_torch_dict())
        )
        self._start_state = JointState(
            position=start_state[:, : self.dynamics_model.d_dof],
            velocity=start_state[:, : self.dynamics_model.d_dof],
            acceleration=start_state[:, : self.dynamics_model.d_dof],
        )
        self.update_cost_dt(self.dynamics_model.dt_traj_params.base_dt)
        return RolloutBase._init_after_config_load(self)

    def cost_fn(self, state: KinematicModelState, action_batch=None, return_list=False):
        # ee_pos_batch, ee_rot_batch = state_dict["ee_pos_seq"], state_dict["ee_rot_seq"]
        state_batch = state.state_seq
        cost_list = []

        # compute state bound  cost:
        if self.bound_cost.enabled:
            with profiler.record_function("cost/bound"):
                c = self.bound_cost.forward(
                    state_batch,
                    self._goal_buffer.retract_state,
                    self._goal_buffer.batch_retract_state_idx,
                )
                cost_list.append(c)
        if self.cost_cfg.manipulability_cfg is not None and self.manipulability_cost.enabled:
            raise NotImplementedError("Manipulability Cost is not implemented")
        if self.cost_cfg.stop_cfg is not None and self.stop_cost.enabled:
            st_cost = self.stop_cost.forward(state_batch.velocity)
            cost_list.append(st_cost)
        if self.cost_cfg.self_collision_cfg is not None and self.robot_self_collision_cost.enabled:
            with profiler.record_function("cost/self_collision"):
                coll_cost = self.robot_self_collision_cost.forward(state.robot_spheres)
                # cost += coll_cost
                cost_list.append(coll_cost)
        if (
            self.cost_cfg.primitive_collision_cfg is not None
            and self.primitive_collision_cost.enabled
        ):
            with profiler.record_function("cost/collision"):
                coll_cost = self.primitive_collision_cost.forward(
                    state.robot_spheres,
                    env_query_idx=self._goal_buffer.batch_world_idx,
                )
                cost_list.append(coll_cost)
        
        # Execute custom arm_base costs
        if hasattr(self, '_custom_arm_base_costs'):
            for cost_name, cost_instance in self._custom_arm_base_costs.items():
                if cost_instance.enabled:
                    with profiler.record_function(f"cost/custom_arm_base/{cost_name}"):
                        try:
                            custom_cost = cost_instance.forward(state)
                            cost_list.append(custom_cost)
                        except Exception as e:
                            log_error(f"Error computing custom arm_base cost {cost_name}: {e}")
        
        # Dynamic obstacle predictive collision checking.
        # dynamic_obs_col_checker = self.get_dynamic_obs_coll_predictor() # If not used, should be None.
        # if dynamic_obs_col_checker is not None:
        #     # dynamic_coll_cost = dynamic_obs_col_checker.cost_fn(state.robot_spheres)
        #     # cost_list.append(dynamic_coll_cost) 


        # Note: Live plotting is handled by child classes (e.g., ArmReacher) to avoid duplicate plots
       
        if return_list:
            return cost_list
        if self.sum_horizon:
            cost = cat_sum_horizon(cost_list)
        else:
            cost = cat_sum(cost_list)
        return cost

    def constraint_fn(
        self,
        state: KinematicModelState,
        out_metrics: Optional[RolloutMetrics] = None,
        use_batch_env: bool = True,
    ) -> RolloutMetrics:
        # setup constraint terms:

        constraint = self.bound_constraint.forward(state.state_seq)

        constraint_list = [constraint]
        if (
            self.constraint_cfg.primitive_collision_cfg is not None
            and self.primitive_collision_constraint.enabled
        ):
            if use_batch_env and self._goal_buffer.batch_world_idx is not None:
                coll_constraint = self.primitive_collision_constraint.forward(
                    state.robot_spheres,
                    env_query_idx=self._goal_buffer.batch_world_idx,
                )
            else:
                coll_constraint = self.primitive_collision_constraint.forward(
                    state.robot_spheres, env_query_idx=None
                )

            constraint_list.append(coll_constraint)
        if (
            self.constraint_cfg.self_collision_cfg is not None
            and self.robot_self_collision_constraint.enabled
        ):
            self_constraint = self.robot_self_collision_constraint.forward(state.robot_spheres)
            constraint_list.append(self_constraint)
        
        # if (dynamic_obs_col_checker := self.get_dynamic_obs_coll_predictor()) is not None: # new
        #     constraint_list.append(dynamic_obs_col_checker.cost_fn(state.robot_spheres))
        
        constraint = cat_sum(constraint_list)

        feasible = constraint == 0.0

        if out_metrics is None:
            out_metrics = RolloutMetrics()
        out_metrics.feasible = feasible
        out_metrics.constraint = constraint
        return out_metrics

    def get_metrics(self, state: Union[JointState, KinematicModelState]):
        """Compute metrics given state

        Args:
            state (Union[JointState, URDFModelState]): _description_

        Returns:
            _type_: _description_

        """
        if self.cuda_graph_instance:
            log_error("Cuda graph is using this instance, please break the graph before using this")
        if isinstance(state, JointState):
            state = self._get_augmented_state(state)
        out_metrics = self.constraint_fn(state)
        out_metrics.state = state
        out_metrics = self.convergence_fn(state, out_metrics)
        out_metrics.cost = self.cost_fn(state)
        return out_metrics

    def get_metrics_cuda_graph(self, state: JointState):
        """Use a CUDA Graph to compute metrics

        Args:
            state: _description_

        Raises:
            ValueError: _description_

        Returns:
            _description_
        """
        if not self._metrics_cuda_graph_init:
            # create new cuda graph for metrics:
            self._cu_metrics_state_in = state.detach().clone()
            s = torch.cuda.Stream(device=self.tensor_args.device)
            s.wait_stream(torch.cuda.current_stream(device=self.tensor_args.device))
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._cu_out_metrics = self.get_metrics(self._cu_metrics_state_in)
            torch.cuda.current_stream(device=self.tensor_args.device).wait_stream(s)
            self.cu_metrics_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cu_metrics_graph, stream=s):
                self._cu_out_metrics = self.get_metrics(self._cu_metrics_state_in)
            self._metrics_cuda_graph_init = True
            self._cuda_graph_valid = True
        if not self.cuda_graph_instance:
            log_error("cuda graph is invalid")
        if self._cu_metrics_state_in.position.shape != state.position.shape:
            log_error("cuda graph changed")
        self._cu_metrics_state_in.copy_(state)
        self.cu_metrics_graph.replay()
        out_metrics = self._cu_out_metrics
        return out_metrics.clone()

    @abstractmethod
    def convergence_fn(
        self, state: KinematicModelState, out_metrics: Optional[RolloutMetrics] = None
    ):
        if out_metrics is None:
            out_metrics = RolloutMetrics()
        return out_metrics

    def _get_augmented_state(self, state: JointState) -> KinematicModelState:
        aug_state = self.compute_kinematics(state)
        if len(aug_state.state_seq.position.shape) == 2:
            aug_state.state_seq = aug_state.state_seq.unsqueeze(1)
            aug_state.ee_pos_seq = aug_state.ee_pos_seq.unsqueeze(1)
            aug_state.ee_quat_seq = aug_state.ee_quat_seq.unsqueeze(1)
            if aug_state.lin_jac_seq is not None:
                aug_state.lin_jac_seq = aug_state.lin_jac_seq.unsqueeze(1)
            if aug_state.ang_jac_seq is not None:
                aug_state.ang_jac_seq = aug_state.ang_jac_seq.unsqueeze(1)
            aug_state.robot_spheres = aug_state.robot_spheres.unsqueeze(1)
            aug_state.link_pos_seq = aug_state.link_pos_seq.unsqueeze(1)
            aug_state.link_quat_seq = aug_state.link_quat_seq.unsqueeze(1)
        return aug_state

    def compute_kinematics(self, state: JointState) -> KinematicModelState:
        # assume input is joint state?
        h = 0
        current_state = state  # .detach().clone()
        if len(current_state.position.shape) == 1:
            current_state = current_state.unsqueeze(0)

        q = current_state.position
        if len(q.shape) == 3:
            b, h, _ = q.shape
            q = q.view(b * h, -1)

        (
            ee_pos_seq,
            ee_rot_seq,
            lin_jac_seq,
            ang_jac_seq,
            link_pos_seq,
            link_rot_seq,
            link_spheres,
        ) = self.dynamics_model.robot_model.forward(q)

        if h != 0:
            ee_pos_seq = ee_pos_seq.view(b, h, 3)
            ee_rot_seq = ee_rot_seq.view(b, h, 4)
            if lin_jac_seq is not None:
                lin_jac_seq = lin_jac_seq.view(b, h, 3, self.n_dofs)
            if ang_jac_seq is not None:
                ang_jac_seq = ang_jac_seq.view(b, h, 3, self.n_dofs)
            link_spheres = link_spheres.view(b, h, link_spheres.shape[-2], link_spheres.shape[-1])
            link_pos_seq = link_pos_seq.view(b, h, -1, 3)
            link_rot_seq = link_rot_seq.view(b, h, -1, 4)

        state = KinematicModelState(
            current_state,
            ee_pos_seq,
            ee_rot_seq,
            link_spheres,
            link_pos_seq,
            link_rot_seq,
            lin_jac_seq,
            ang_jac_seq,
            link_names=self.kinematics.link_names,
        )
        return state

    def rollout_constraint(
        self, act_seq: torch.Tensor, use_batch_env: bool = True
    ) -> RolloutMetrics:
        if self.cuda_graph_instance:
            log_error("Cuda graph is using this instance, please break the graph before using this")
        state = self.dynamics_model.forward(self.start_state, act_seq)
        metrics = self.constraint_fn(state, use_batch_env=use_batch_env)
        return metrics

    def rollout_constraint_cuda_graph(self, act_seq: torch.Tensor, use_batch_env: bool = True):
        # TODO: move this to RolloutBase
        if not self._rollout_constraint_cuda_graph_init:
            # create new cuda graph for metrics:
            self._cu_rollout_constraint_act_in = act_seq.clone()
            s = torch.cuda.Stream(device=self.tensor_args.device)
            s.wait_stream(torch.cuda.current_stream(device=self.tensor_args.device))
            with torch.cuda.stream(s):
                for _ in range(3):
                    state = self.dynamics_model.forward(self.start_state, act_seq)
                    self._cu_rollout_constraint_out_metrics = self.constraint_fn(
                        state, use_batch_env=use_batch_env
                    )
            torch.cuda.current_stream(device=self.tensor_args.device).wait_stream(s)
            self.cu_rollout_constraint_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cu_rollout_constraint_graph, stream=s):
                state = self.dynamics_model.forward(self.start_state, act_seq)
                self._cu_rollout_constraint_out_metrics = self.constraint_fn(
                    state, use_batch_env=use_batch_env
                )
            self._rollout_constraint_cuda_graph_init = True
            self._cuda_graph_valid = True
        if not self.cuda_graph_instance:
            log_error("cuda graph is invalid")
        self._cu_rollout_constraint_act_in.copy_(act_seq)
        self.cu_rollout_constraint_graph.replay()
        out_metrics = self._cu_rollout_constraint_out_metrics
        return out_metrics.clone()

    def rollout_fn(self, act_seq) -> Trajectory:
        """
        Return sequence of costs and states encountered
        by simulating a batch of action sequences

        Parameters
        ----------
        action_seq: torch.Tensor [num_particles, horizon, d_act]
        """

        # print(act_seq.shape, self._goal_buffer.batch_current_state_idx)
        if self.start_state is None:
            raise ValueError("start_state is not set in rollout")
        with profiler.record_function("robot_model/rollout"):
            state = self.dynamics_model.forward(
                self.start_state, act_seq, self._goal_buffer.batch_current_state_idx
            )

        with profiler.record_function("cost/all"):
            cost_seq = self.cost_fn(state, act_seq)

        sim_trajs = Trajectory(actions=act_seq, costs=cost_seq, state=state)

        return sim_trajs

    def update_params(self, goal: Goal):
        """
        Updates the goal targets for the cost functions.

        """
        with profiler.record_function("arm_base/update_params"):
            self._goal_buffer.copy_(goal, update_idx_buffers=self._goal_idx_update)

            if goal.current_state is not None:
                if self.start_state is None:
                    self.start_state = goal.current_state.clone()
                else:
                    self.start_state = self.start_state.copy_(goal.current_state)
            self.batch_size = goal.batch
        return True

    def get_ee_pose(self, current_state):
        current_state = current_state.to(**self.tensor_args)

        (ee_pos_batch, ee_quat_batch) = self.dynamics_model.robot_model.forward(
            current_state[:, : self.dynamics_model.n_dofs]
        )[0:2]

        state = KinematicModelState(current_state, ee_pos_batch, ee_quat_batch)
        return state

    def current_cost(self, current_state: JointState, no_coll=False, return_state=True, **kwargs):
        state = self._get_augmented_state(current_state)

        if "horizon_cost" not in kwargs:
            kwargs["horizon_cost"] = False

        cost = self.cost_fn(state, None, no_coll=no_coll, **kwargs)

        if return_state:
            return cost, state
        else:
            return cost

    def filter_robot_state(self, current_state: JointState) -> JointState:
        return self.dynamics_model.filter_robot_state(current_state)

    def get_robot_command(
        self,
        current_state: JointState,
        act_seq: torch.Tensor,
        shift_steps: int = 1,
        state_idx: Optional[torch.Tensor] = None,
    ) -> JointState:
        return self.dynamics_model.get_robot_command(
            current_state,
            act_seq,
            shift_steps=shift_steps,
            state_idx=state_idx,
        )

    def reset(self):
        self.dynamics_model.state_filter.reset()
        super().reset()

    @property
    def d_action(self):
        return self.dynamics_model.d_action

    @property
    def action_bound_lows(self):
        return self.dynamics_model.action_bound_lows

    @property
    def action_bound_highs(self):
        return self.dynamics_model.action_bound_highs

    @property
    def state_bounds(self) -> Dict[str, List[float]]:
        return self.dynamics_model.get_state_bounds()

    @property
    def dt(self):
        return self.dynamics_model.dt

    @property
    def horizon(self):
        return self.dynamics_model.horizon

    @property
    def action_horizon(self):
        return self.dynamics_model.action_horizon

    def get_init_action_seq(self) -> torch.Tensor:
        act_seq = self.dynamics_model.init_action_mean.unsqueeze(0).repeat(self.batch_size, 1, 1)
        return act_seq

    def reset_shape(self):
        self._goal_idx_update = True
        super().reset_shape()

    def reset_cuda_graph(self):
        super().reset_cuda_graph()

    def get_action_from_state(self, state: JointState):
        return self.dynamics_model.get_action_from_state(state)

    def get_state_from_action(
        self,
        start_state: JointState,
        act_seq: torch.Tensor,
        state_idx: Optional[torch.Tensor] = None,
    ):
        return self.dynamics_model.get_state_from_action(start_state, act_seq, state_idx)

    @property
    def kinematics(self):
        return self.dynamics_model.robot_model

    @property
    def cspace_config(self) -> CSpaceConfig:
        return self.dynamics_model.robot_model.kinematics_config.cspace

    # @property
    # def num_particles_rollout(self):
    #     return self._num_particles_rollout
    
    # def set_num_particles_rollout(self, num_particles: int):
    #     """Set the number of particles for rollout (called by optimizer)."""
    #     was_uninitialized = self._num_particles_rollout <= 0
    #     self._num_particles_rollout = num_particles
        
    #     # Update custom costs if they exist and need this parameter
    #     if hasattr(self, '_custom_arm_base_costs'):
    #         for cost_name, cost_instance in self._custom_arm_base_costs.items():
    #             if hasattr(cost_instance, 'num_particles_rollout'):
    #                 cost_instance.num_particles_rollout = num_particles
    #                 log_info(f"Updated num_particles_rollout={num_particles} for custom arm_base cost: {cost_name}")
                    
    #                 # Trigger initialization if the cost has a _initialize_predictor method
    #                 if hasattr(cost_instance, '_initialize_predictor'):
    #                     cost_instance._initialize_predictor()

    def get_full_dof_from_solution(self, q_js: JointState) -> JointState:
        """This function will all the dof that are locked during optimization.


        Args:
            q_sol: _description_

        Returns:
            _description_
        """
        if self.kinematics.lock_jointstate is None:
            return q_js
        all_joint_names = self.kinematics.all_articulated_joint_names
        lock_joint_state = self.kinematics.lock_jointstate

        new_js = q_js.get_augmented_joint_state(all_joint_names, lock_joint_state)
        return new_js

    @property
    def joint_names(self) -> List[str]:
        return self.kinematics.joint_names

    @property
    def retract_state(self):
        return self.dynamics_model.retract_config

    def update_traj_dt(
        self,
        dt: Union[float, torch.Tensor],
        base_dt: Optional[float] = None,
        max_dt: Optional[float] = None,
        base_ratio: Optional[float] = None,
    ):
        self.dynamics_model.update_traj_dt(dt, base_dt, max_dt, base_ratio)
        self.update_cost_dt(dt)

    def update_cost_dt(self, dt: float):
        # scale any temporal costs by dt:
        self.bound_cost.update_dt(dt)
        if self.cost_cfg.primitive_collision_cfg is not None:
            self.primitive_collision_cost.update_dt(dt)

    def _update_live_plot(self, cost_list):
        """Update live plot of cost values in real-time"""
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from collections import deque
        import numpy as np
        
        # Initialize plotting components if not already done
        if not hasattr(self, '_plot_initialized'):
            self._plot_initialized = True
            self._cost_histories = {}  # Dictionary to store history for each cost component
            self._cost_lines = {}  # Dictionary to store plot lines for each cost component
            self._plot_counter = 0  # Counter for plotting frequency
            self._plot_every_k = 5  # Plot every 5 iterations to save resources
            
            # Set up the figure and axis
            plt.ion()  # Turn on interactive mode
            self._fig, self._ax = plt.subplots(1, 1, figsize=(12, 8))
            self._fig.suptitle('Real-time Cost Monitoring - ArmBase Components')
            
            self._ax.set_title('Individual Cost Components Over Time (Base + Custom)')
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
        if hasattr(self, 'bound_cost') and self.bound_cost.enabled:
            cost_labels_dynamic.append('Bound Cost')
        if hasattr(self, 'stop_cost') and self.stop_cost.enabled:
            cost_labels_dynamic.append('Stop Cost')
        if hasattr(self, 'robot_self_collision_cost') and self.robot_self_collision_cost.enabled:
            cost_labels_dynamic.append('Self Collision')
        if hasattr(self, 'primitive_collision_cost') and self.primitive_collision_cost.enabled:
            cost_labels_dynamic.append('Primitive Collision')
        
        # Add custom arm_base costs with their class names
        if hasattr(self, '_custom_arm_base_costs'):
            for cost_name, cost_instance in self._custom_arm_base_costs.items():
                if cost_instance.enabled:
                    # Extract class name from the cost instance
                    class_name = cost_instance.__class__.__name__
                    cost_labels_dynamic.append(f'Custom: {class_name}')
        
        # Check for dynamic obstacles and manipulability (added after custom costs)
        # if hasattr(self, '_dynamic_obs_coll_predictor') and self._dynamic_obs_coll_predictor is not None:
        #     cost_labels_dynamic.append('Dynamic Obstacles')
        if hasattr(self, 'manipulability_cost') and self.manipulability_cost.enabled:
            cost_labels_dynamic.append('Manipulability')
        
        # Colors for plotting
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # Process each cost component
        active_costs = []
        for i, cost_tensor in enumerate(cost_list):
            if cost_tensor is not None:
                # Get cost label (use index if we don't have enough labels)
                if i < len(cost_labels_dynamic):
                    label = cost_labels_dynamic[i]
                else:
                    label = f'Unknown_Cost_{i}'
                
                # Calculate mean of this cost component
                cost_mean = torch.mean(cost_tensor).cpu().numpy().item()
                active_costs.append((label, cost_mean, i))
                
                # Initialize history for this component if not exists
                if label not in self._cost_histories:
                    self._cost_histories[label] = deque(maxlen=100)  # Keep last 100 iterations
                    color = colors[i % len(colors)]
                    
                    # Special styling for custom costs
                    if 'Custom' in label:
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
        if self._plot_counter <= self._plot_every_k * 2:  # First 2 plot updates
            print(f"Active ArmBase cost components: {[(label, f'{val:.6f}') for label, val, _ in active_costs]}")
            custom_costs = [label for label, _, _ in active_costs if 'Custom' in label]
            if custom_costs:
                print(f"Custom arm_base costs detected: {custom_costs}")
        
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
        
        # Update legend (only once or when new components are added)
        if not hasattr(self, '_legend_updated') or len(self._cost_lines) != getattr(self, '_last_legend_count', 0):
            self._ax.legend(loc='upper right')
            self._legend_updated = True
            self._last_legend_count = len(self._cost_lines)
        
        # Refresh the plot
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        
        # Optional: Save periodic snapshots
        total_iterations = max(len(h) for h in self._cost_histories.values()) if self._cost_histories else 0
        if hasattr(self, '_save_plots') and self._save_plots and total_iterations > 0 and total_iterations % 50 == 0:  # Every 50 iterations
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if not hasattr(self, '_cost_plots_dir'):
                self._cost_plots_dir = os.path.join(os.getcwd(), 'tmp_artifacts', 'cost_plots_base', timestamp)
                os.makedirs(self._cost_plots_dir, exist_ok=True)
            self._fig.savefig(os.path.join(self._cost_plots_dir, f'base_costs_iter_{total_iterations}.png'), dpi=150, bbox_inches='tight')
            print(f"Saved ArmBase plot snapshot at iteration {total_iterations}")

    def enable_live_plotting(self, enable: bool = True, save_plots: bool = False):
        """Enable or disable live plotting of cost values
        
        Args:
            enable (bool): Whether to enable live plotting. Defaults to True.
        """
        self._enable_live_plotting = enable # live plotting of cost values if True
        self._save_plots = save_plots # save plots to file if True
        if not enable and hasattr(self, '_fig'):
            # Close the figure if disabling
            plt.close(self._fig)
            self._plot_initialized = False

    def save_cost_plot(self):
        """Save the current cost plot to file
        
        Args:
            filename (str, optional): Filename to save. If None, uses timestamp.
        """
        if hasattr(self, '_fig') and self._save_plots:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if not hasattr(self, '_cost_plots_dir'):
                self._cost_plots_dir = os.path.join(os.getcwd(), 'cost_plots', timestamp)
                os.makedirs(self._cost_plots_dir, exist_ok=False)
            filename = f'cost_plot_{timestamp}.png'
            self._fig.savefig(os.path.join(self._cost_plots_dir, filename), dpi=150, bbox_inches='tight')
            print(f"Cost plot saved to {os.path.join(self._cost_plots_dir, filename)}")
        else:
            print("No plot available to save. Enable live plotting first.")

    def get_custom_cost(self, cost_identifier: str):
        """Get a custom cost by name or class name.
        
        Args:
            cost_identifier: Can be the exact cost key, class name, or simplified name
            
        Returns:
            The custom cost instance if found, None otherwise
        """
        if not hasattr(self, '_custom_arm_base_costs'):
            return None
            
        # First try exact match
        if cost_identifier in self._custom_arm_base_costs:
            return self._custom_arm_base_costs[cost_identifier]
        
        # Try to find by class name or simplified name
        for cost_key, cost_instance in self._custom_arm_base_costs.items():
            class_name = cost_instance.__class__.__name__
            
            # Match by class name
            if class_name == cost_identifier:
                return cost_instance
            
            # Match by simplified name (e.g., "dynamic_obs_cost" matches "DynamicObsCost")
            simplified_class_name = ''.join(['_' + c.lower() if c.isupper() else c for c in class_name]).lstrip('_')
            if simplified_class_name == cost_identifier:
                return cost_instance
            
            # Match by partial name in the key
            if cost_identifier in cost_key:
                return cost_instance
                
        return None

    def list_custom_costs(self):
        """List all available custom costs with their keys and class names."""
        if not hasattr(self, '_custom_arm_base_costs'):
            return {}
            
        cost_info = {}
        for cost_key, cost_instance in self._custom_arm_base_costs.items():
            class_name = cost_instance.__class__.__name__
            simplified_name = ''.join(['_' + c.lower() if c.isupper() else c for c in class_name]).lstrip('_')
            cost_info[cost_key] = {
                'class_name': class_name,
                'simplified_name': simplified_name,
                'enabled': cost_instance.enabled if hasattr(cost_instance, 'enabled') else 'unknown'
            }
        return cost_info
