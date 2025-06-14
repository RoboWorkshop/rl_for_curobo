"""
Custom Cost Template

This template shows how to create custom cost terms for CuRobo.
Copy this file and modify it to create your own custom cost terms.
"""

# Standard Library
from dataclasses import dataclass

# Third Party
from curobo.rollout.cost.cost_base import CostConfig
import numpy as np
import torch

# CuRobo
from curobo.rollout.cost.custom.custom_cost import CustomCost, CustomCostConfig
from curobo.rollout.dynamics_model.kinematic_model import KinematicModelState
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import DynamicObsCollPredictor

@dataclass
class Cfg(CustomCostConfig):
    """Configuration for your custom cost term.
    
    Add any custom parameters here. All CostConfig parameters are available:
    - weight: Cost weight (float)
    - vec_weight: Per-timestep weights (Optional[torch.Tensor])
    - terminal: Apply only to terminal state (bool)
    - run_weight: Weight for non-terminal timesteps (float)
    - return_loss: Return individual losses instead of sum (bool)
    """
    
    # Add your custom parameters here for dynamic obstacle collision checking
    # syntax: "param: type = default_value"
    
    n_own_spheres: int = 61 # number of spheres to check for collision
    n_coll_spheres_valid: int = 61 # number of spheres to check for collision
    manually_express_p_own_in_world_frame: bool = False # whether to manually express the robot's position in the world frame
    safety_margin: float = 0.1 # safety margin in meters for the dynamic obstacle collision checking.
    
    def __post_init__(self):
        # ADD YOUR CUSTOM INITIALIZATION HERE
        # your code...
        # Call parent post_init to handle tensor_args setup
        super().__post_init__()

        
        


class Cost(CustomCost, Cfg):
    """Template for a custom cost term.
    
    This template shows the basic structure for implementing custom costs.
    Replace 'MyCustomCost' with your actual cost name and implement the forward method.
    """
    
    def __init__(self, config: Cfg):
        # Initialize configuration
        Cfg.__init__(self, **vars(config))
        CustomCost.__init__(self)


        assert isinstance(self.weight, torch.Tensor), "weight must be a torch.Tensor"
        weight_arg = self.weight.item() # to float
        # Initialize the dynamic obstacle collision predictor after tensor_args is set up
        self.dynamic_obs_col_pred = DynamicObsCollPredictor(
            self.tensor_args,
            self.horizon, 
            self.n_particles, 
            self.n_own_spheres,
            self.n_coll_spheres_valid,
            weight_arg,
            [],
            self.manually_express_p_own_in_world_frame,
            self.tensor_args.to_device(self.p_R),  # Ensure tensor is on correct device
            self.safety_margin
        )
        
    def forward(self, state: KinematicModelState) -> torch.Tensor:
        """Compute the cost.
        
        This is the main function where you implement your cost logic.
        
        Args:
            state: KinematicModelState containing robot state information.
                  Key attributes:
                  - state.state_seq.position: Joint positions [batch, horizon, n_dofs]
                  - state.state_seq.velocity: Joint velocities [batch, horizon, n_dofs]
                  - state.state_seq.acceleration: Joint accelerations [batch, horizon, n_dofs]
                  - state.ee_pos_seq: End-effector positions [batch, horizon, 3]
                  - state.ee_quat_seq: End-effector quaternions [batch, horizon, 4]
                  - state.robot_spheres: Robot collision spheres [batch, horizon, n_spheres, 4]
                  - state.link_pos_seq: Link positions [batch, horizon, n_links, 3]
                  - state.link_quat_seq: Link quaternions [batch, horizon, n_links, 4]
                      
        Returns:
            torch.Tensor: Cost values of shape [batch_size, horizon]
                        Values should be non-negative.
        """
        # Check if robot_spheres is None first
        if state.robot_spheres is None:
            # Return a default tensor with reasonable dimensions
            dummy_output = torch.zeros(
                1, 1,  # Default batch_size=1, horizon=1
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype
            )
            return dummy_output
        
        # Create dummy output with correct device and dtype using tensor_args
        dummy_output = torch.zeros(
            state.robot_spheres.shape[0], 
            state.robot_spheres.shape[1],
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype
        )
        
        is_mpc_initiation_step = state.robot_spheres.shape[0] != self._num_particles_rollout_full# self.dynamic_obs_col_checker.n_rollouts
        if is_mpc_initiation_step:
            return dummy_output

        # we are in the standard planning step, not the initiation step
        cost = self.dynamic_obs_col_pred.cost_fn(state.robot_spheres)
        
        # Ensure the cost tensor is on the correct device and has the correct dtype
        if not isinstance(cost, torch.Tensor):
            cost = torch.tensor(cost, device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        elif cost.device != self.tensor_args.device or cost.dtype != self.tensor_args.dtype:
            cost = cost.to(device=self.tensor_args.device, dtype=self.tensor_args.dtype)
            
        return cost 
        
        