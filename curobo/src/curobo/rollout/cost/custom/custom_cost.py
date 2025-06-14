
"""
Custom Cost Template

This template shows how to create custom cost terms for CuRobo.
Copy this file and modify it to create your own custom cost terms.
"""

# Standard Library
from dataclasses import dataclass
import importlib

# Third Party
import numpy as np
import torch

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.rollout.cost.cost_base import CostBase, CostConfig
from curobo.rollout.dynamics_model.kinematic_model import KinematicModelState
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import DynamicObsCollPredictor

@dataclass
class CustomCostConfig(CostConfig):
    name: str = ''
    n_particles: int = -1
    horizon: int = -1
    cost_family: str = ''
    p_R: tuple = ()
    tensor_args: TensorDeviceType = TensorDeviceType()

    def __post_init__(self):
        # ADD YOUR CUSTOM INITIALIZATION HERE
        # your code...
        # Call parent post_init to handle tensor_args setup
        super().__post_init__()

class CustomCost(CostBase, CustomCostConfig):
    @classmethod
    def get_modules_path_prefix(cls):
        return 'curobo.rollout.cost.custom.'
    
    def import_classes_from_modules(cls):
        module = importlib.import_module(cls.get_modules_path_prefix()[:-1])
        for class_name in dir(module):
            if class_name.endswith('Config'):
                setattr(cls, class_name, getattr(module, class_name))
    
    @classmethod
    def parse_cfg_from_file(cls, config: Dict) -> CustomCostConfig:
        return CustomCostConfig(**config)

    def __init__(self):
        CustomCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)

        # self.name = self.name
        # self.n_particles = self.n_particles
        # self.horizon = self.horizon
        # self.p_R = self.p_R
        # self.config = config
        


        
