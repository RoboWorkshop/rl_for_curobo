import threading
from curobo.geom.sdf.world import CollisionQueryBuffer
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from curobo.geom.types import Cuboid, Sphere, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from typing import List
import numpy as np
import copy
import torch
from curobo.geom.sdf.world import WorldCollisionConfig
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.runtime_topics import get_topics
from projects_root.utils.plot_spheres import SphereVisualizer
from projects_root.utils.quaternion import integrate_quat
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.utils import shift_tensor_left, mask_decreasing_values
from concurrent.futures import ProcessPoolExecutor, wait
from projects_root.utils.transforms import transform_poses_batched, transform_robot_positions_to_world, create_optimized_collision_checker_buffers, transform_positions_with_precomputed_matrix
from curobo.geom.transform import batch_transform_points

import time

def f_mask(d_col:torch.Tensor, margin:float) -> torch.Tensor:
    
    if torch.any(d_col < 0):
        print(f'DEBUG: d_col < 0: {d_col}')
        raise ValueError('d_col < 0, d_col should be non-negative') 
    x = d_col 
    m =  -(1 / margin)
    b = 1
    f =   b + m * x
    f_pos = torch.relu(f)
    return f_pos

    

class DynamicObsCollPredictor:
    """
    
    This class is used to predict and compute costs for the collision of the dynamic obstacles in the world.
    You can think of this class as a collection of H collision checkers, one for each time step in the horizon + functionality to compute costs for the MPC (see cost_fn function).
    Each individual collision checker which it contains, is inspired by: https://curobo.org/get_started/2c_world_collision.html
    """
    

    def __init__(self, 
                 a_select_mode:str,
                 tensor_args, 
                 H=30, 
                 n_rollouts=400, 
                 n_own_spheres=65, # total number of spheres of the robot (including ones that we don't want to check for collision)
                 n_obs=65, # total number of spheres of the obstacles in the world (including ones that we don't want to check for collision)
                 cost_weight=100.0, 
                 X = [0,0,0,1,0,0,0],
                 sparse_steps:dict={'use': False, 'ratio': 0.5},
                 sparse_spheres:dict={'exclude_self': [], 'exclude_others': []}, # list of ints, each int is the index of the sphere to exclude from the collision check.
                 col_with_idx_map=None, # mapping from robot IDs to their index ranges in concatenated tensor
                 safety_margin=0.1,
                 cost_type='linear',
                 prior_rule='none',
                 wta_trust=1.0,
                 ):
        """ Initialize H dynamic obstacle collision checker, for each time step in the horizon, 
        as well as setting the cost function parameters for the dynamic obstacle cost function.


        Args:
            tensor_args: pytorch tensor arguments.
            cache (dict): collision checker cache for the pre-defined dynamic primitives.
            H (int, optional): Defaults to 30. The horizon length- number of states in the trajectory during trajectory optimization.
            n_checkers(int, optional): Defaults to H (marked by passing -1). The number of collision checkers to use. If n_checkers is not H, then the collision checkers will be used in a sliding window fashion.
            n_rollouts (int, optional): Defaults to 400. The number of rollouts. TODO: Should be taken from the mpc config.
            cost_weight: weight for the dynamic obstacle cost function (cost term weight). This is a hyper-parameter, unlike the weight_col_check which you should leave as 1. Default value is 100000, as by the original primitive collision cost weight of the mpc.
            col_with_idx_map: Dictionary mapping robot IDs to their index ranges in the concatenated obstacle tensor
            """
        
        self.a_select_mode = a_select_mode # normal/col_free
        self.X = X # the pose (x y z qw qx qy qz) of the robot in the world frame
        self.tensor_args = tensor_args 
        self.H = H
        self.safety_margin = safety_margin
        self.n_rollouts = n_rollouts
        self.n_own_spheres = n_own_spheres
        self.wta_trust = wta_trust
        self.cost_type = cost_type
        assert self.cost_type in ['linear', 'storm_binary'], "Invalid cost type"
        
        # control sparsity over horizon (for efficiency)
        self.sparse_steps = sparse_steps
        self._window_ranges = [] 
        self._debug = {'p_own':np.ndarray([]), 'p_obs':np.ndarray([])}
        
        # pose wta conflict resolution
        self.prior_rule = prior_rule
        assert self.prior_rule in ['none', 'pose', 'random', 'pose_wta'], "Invalid prior rule"
        if self.prior_rule == 'pose_wta': # "winner takes it all" 
            self.pose_wta_conflict_resolution = {} # {robot_id: (p_err, q_err)}


        
        
        if self.sparse_steps['use']:
            assert self.sparse_steps['ratio'] > 0 and self.sparse_steps['ratio'] <= 1, "Error: The ratio must be between 0 and 1"

            # making n non-overlapping windows:            
            # num of windows = num of sampling steps
            self.n_sampling_steps = int(self.H * self.sparse_steps['ratio']) # number of timesteps to sample for collision
            # self.sampling_timesteps = np.linspace(0, self.H-1, self.n_sampling_steps, dtype=int) # evenly spaced timesteps in the horizon
            _sampling_time_steps = []
            _window_size = self.H // self.n_sampling_steps
            _window_start_idx = 0
            for window_idx in range(self.n_sampling_steps):
                if window_idx == self.n_sampling_steps - 1:
                    _window_end_idx = self.H - 1 # last window might  be bigger than the other windows
                else:
                    _window_end_idx = _window_start_idx + _window_size - 1

                _window_mid_idx = (_window_start_idx +_window_end_idx) // 2
                _sampling_time_steps.append(_window_mid_idx)
                self._window_ranges.append([_window_start_idx, _window_end_idx])
                _window_start_idx = _window_end_idx + 1
            self.sampling_timesteps = np.array(_sampling_time_steps)
            
        else:
            self.n_sampling_steps = self.H
            self.sampling_timesteps = np.arange(self.H)
    

        self.cost_weight = cost_weight

        # control sparsity over spheres - filter out spheres which are not needed/less relevant (for efficiency)
        self.sparse_spheres = sparse_spheres # sparse spheres config
        
        # Store collision mapping for robot-specific updates
        self.col_with_idx_map = col_with_idx_map or {}
        
        # Filter out invalid sphere indices that are outside the range of available spheres
        valid_exclude_self = [idx for idx in self.sparse_spheres['exclude_self'] if 0 <= idx < self.n_own_spheres]
        self.valid_own_spheres = np.array(list(set(list(range(self.n_own_spheres))) - set(valid_exclude_self)))
        
        self.n_obs = n_obs # number of valid obstacles (ignoring 4 spheres which are not valid due to negative radius)
        
        # Filter out invalid sphere indices that are outside the range of available obstacle spheres
        valid_exclude_others = [idx for idx in self.sparse_spheres['exclude_others'] if 0 <= idx < self.n_obs]
        self.valid_obs_spheres = np.array(list(set(list(range(self.n_obs))) - set(valid_exclude_others)))
        
        # Pre-compute index tensors for efficient indexing
        self.sampling_timesteps_tensor = torch.tensor(self.sampling_timesteps, device=self.tensor_args.device, dtype=torch.long)
        self.valid_own_spheres_tensor = torch.tensor(self.valid_own_spheres, device=self.tensor_args.device, dtype=torch.long)
        self.valid_obs_spheres_tensor = torch.tensor(self.valid_obs_spheres, device=self.tensor_args.device, dtype=torch.long)
        
        # Compute effective dimensions after filtering
        self.n_valid_own = len(self.valid_own_spheres)
        self.n_valid_obs = len(self.valid_obs_spheres)
        
        # Buffers for obstacles (spheres): position and radius
        self.rad_obs_buf = torch.zeros(self.n_valid_obs, device=self.tensor_args.device) # [n_valid_obs] obstacles radii buffer
        
        # Pre-allocate obstacle position buffer in the final shape to avoid reshaping
        self.p_obs_buf = torch.zeros((self.n_sampling_steps, self.n_valid_obs, 3), device=self.tensor_args.device)
        # Pre-allocate in broadcast-ready shape to eliminate reshaping in cost_fn
        self.p_obs_buf_broadcast = torch.zeros((1, self.n_sampling_steps, 1, self.n_valid_obs, 3), device=self.tensor_args.device)

        # Buffers for own spheres: position and radius  
        self.p_own_buf = torch.empty(n_rollouts, self.n_sampling_steps, self.n_valid_own, 3, device=self.tensor_args.device)

        self.p_own_buf_unfiltered = torch.zeros(n_rollouts, self.H, self.n_own_spheres, 3, device=self.tensor_args.device)
        self.p_spheres_tmp_time_filtered_buf = torch.empty((n_rollouts, self.n_sampling_steps, self.n_own_spheres, 3), device=self.tensor_args.device)
        self.p_own_filtered_buf = torch.empty((n_rollouts, self.n_sampling_steps, self.n_valid_own, 3), device=self.tensor_args.device)
        self.X_own_R = torch.zeros(n_rollouts, self.H, self.n_own_spheres, 7, device=self.tensor_args.device)
        self.X_own_R[:,:,:,3] = 1 # this is making the quat 1,0,0,0, which is the identity quaternion

        # Pre-allocate in broadcast-ready shape to eliminate reshaping in cost_fn
        self.p_own_buf_broadcast = torch.empty(self.n_rollouts, self.n_sampling_steps, self.n_valid_own, 1, 3, device=self.tensor_args.device)
        
        self.rad_own_buf = torch.zeros(self.n_valid_own, device=self.tensor_args.device) # [n_valid_own] Own spheres radii buffer

        # Pre-compute radius sum matrix in final broadcast shape to avoid repeated operations
        # Shape: [1, 1, n_valid_own, n_valid_obs, 1] - ready for broadcasting
        self.pairwise_radsum_broadcast = torch.zeros(1, 1, self.n_valid_own, self.n_valid_obs, 1, device=self.tensor_args.device)
    
        # Main computation buffers - pre-allocated in final shapes
        self.ownobs_diff_vector_buff = torch.zeros(self.n_rollouts, self.n_sampling_steps, self.n_valid_own, self.n_valid_obs, 3, device=self.tensor_args.device)
        self.pairwise_surface_dist_buf = torch.zeros(self.n_rollouts, self.n_sampling_steps, self.n_valid_own, self.n_valid_obs, 1, device=self.tensor_args.device)
        
        # Output buffers
        self.cost_mat_buf = torch.zeros(n_rollouts, H, device=self.tensor_args.device) # [n_rollouts x H] 
        self.tmp_cost_mat_buf_sparse = torch.zeros(n_rollouts, self.n_sampling_steps, device=self.tensor_args.device)
        
        # # Pre-compute projection parameters for sparse-to-full horizon mapping
        # if self.sparse_steps['use']:
        #     self.base_repeat = self.H // self.n_sampling_steps
        #     self.remainder = self.H % self.n_sampling_steps
        #     if self.remainder != 0:
        #         # Pre-compute repeat counts to avoid recreation each time
        #         self.repeat_counts = torch.full((self.n_sampling_steps,), self.base_repeat, dtype=torch.long, device=self.tensor_args.device)
        #         self.repeat_counts[:self.remainder] += 1
        
        # flags
        self.init_rad_buffs = torch.tensor([0], device=self.tensor_args.device) # [1] If 1, the rad_obs_buffs are initialized (obstacles which should be set only once).
    
        self.rotation_matrix = None
        self.world_translation = None
        self.transform_matrix_dirty = True
        self.wall_clock_time = 0.0
        self.wall_clock_steps = 0
        
    def _project_sparse_to_full_horizon(self, sparse_matrix: torch.Tensor, full_matrix: torch.Tensor):
        """
        Project a sparse matrix from (n_rollouts, n_sampling_steps) to (n_rollouts, H).
        Optimized version with pre-computed parameters.
        """
        if self.sparse_steps['use']:
            # project the value at the middle of the window to the whole window
            for window_idx, window_range in enumerate(self._window_ranges):
                win_start, win_end = window_range
                full_matrix[:, win_start:win_end+1] = sparse_matrix[:, window_idx].unsqueeze(1)
            
            # if self.remainder == 0:
            #     # Simple case: all columns repeated equally
            #     full_matrix.copy_(sparse_matrix.repeat_interleave(self.base_repeat, dim=1))
            # else:
            #     # Complex case: use pre-computed repeat counts
            #     full_matrix.copy_(sparse_matrix.repeat_interleave(self.repeat_counts, dim=1))
        else:
            # If not using sparse steps, just copy directly (sparse_matrix is already full horizon)
            full_matrix.copy_(sparse_matrix)
        
    def set_obs_rads(self, rad_obs:torch.Tensor):
        """
        rad_obs: tensor of shape [n_obs]. The radii of the obstacles.
        """
        assert rad_obs.ndim == 1, "Error: The obstacle radii must be a 1D tensor"
        
        # Direct indexing and copy
        self.rad_obs_buf.copy_(rad_obs[self.valid_obs_spheres])

    def set_obs_rads_for_robot(self, robot_id: int, rad_robot: torch.Tensor):
        """
        Set the radii for a specific robot in the concatenated obstacle buffer.
        
        Args:
            robot_id: ID of the robot whose radii to set
            rad_robot: tensor of shape [n_robot_spheres]. The radii for this robot.
        """
        if robot_id not in self.col_with_idx_map:
            print(f"Warning: Robot {robot_id} not found in collision mapping")
            return
        
        robot_map = self.col_with_idx_map[robot_id]
        start_idx = robot_map['start_idx']
        end_idx = robot_map['end_idx']
        valid_indices = robot_map['valid_indices']
        
        # Filter robot radii based on its exclusion list
        filtered_radii = rad_robot[valid_indices]  # [n_valid_robot_spheres]
        
        # Update the corresponding section in the concatenated obstacle radius buffer
        self.rad_obs_buf[start_idx:end_idx] = filtered_radii

    def set_own_rads(self, rad_own:torch.Tensor):
        """
        Set the radii of the own spheres and pre-compute radius sum matrix.
        """
        # Copy own radii
        self.rad_own_buf.copy_(rad_own[self.valid_own_spheres])
        
        # Pre-compute pairwise radius sums in broadcast-ready shape
        # Broadcasting: [n_valid_own, 1] + [1, n_valid_obs] -> [n_valid_own, n_valid_obs]
        rad_sum = self.rad_own_buf.unsqueeze(1) + self.rad_obs_buf.unsqueeze(0)
        # Store in final broadcast shape [1, 1, n_valid_own, n_valid_obs, 1]
        self.pairwise_radsum_broadcast[0, 0, :, :, 0] = rad_sum
        
        self.init_rad_buffs[0] = 1
    
    def update(self, p_obs:torch.Tensor):
        """
        Update the positions of the obstacles (sphere centers).
        note: assuming that the poses are expressed in the world frame [0,0,0,1,0,0,0].
        Args:
            p_obs: tensor of shape [H, n_obs, 3]. The poses of the obstacles.
        """
        # Optimized indexing using pre-computed tensors
        temp_obs = torch.index_select(p_obs, 0, self.sampling_timesteps_tensor)  # Select timesteps
        temp_obs = torch.index_select(temp_obs, 1, self.valid_obs_spheres_tensor)  # Select spheres
        self.p_obs_buf.copy_(temp_obs)
        
        # Update broadcast-ready buffer
        self.p_obs_buf_broadcast[0, :, 0, :, :] = self.p_obs_buf

    def update_world_pose(self, world_pose):
        """Update world pose and pre-compute transformation matrix."""
        self.X = world_pose
        if not hasattr(self, 'X_tensor') or self.X_tensor is None:
            self.X_tensor = torch.tensor(self.X, device=self.tensor_args.device, dtype=torch.float32)
        
        # Pre-compute rotation matrix for ultra-fast transforms
        self.rotation_matrix, self.world_translation = create_optimized_collision_checker_buffers(
            self.n_rollouts, self.H, self.n_own_spheres, self.X_tensor, self.tensor_args.device
        )
        self.transform_matrix_dirty = False

    def cost_fn(self, prad_own_R: torch.Tensor):
        """Ultra-optimized collision cost computation."""
        
        
        cost_fn_start_time = time.time()
        # Update transformation matrix if needed
        if self.transform_matrix_dirty or self.rotation_matrix is None:
            self.update_world_pose(self.X)
        
        # Ensure transformation matrices are available
        if self.rotation_matrix is None or self.world_translation is None:
            raise RuntimeError("Transformation matrices not initialized")
        
        # Extract positions from robot frame poses
        robot_positions = prad_own_R[:, :, :, :3]  # (n_rollouts, H, n_spheres, 3)
        
        # filter out the timesteps and spheres that are not needed *before* transformation to save FLOPs
        torch.index_select(robot_positions, 1, self.sampling_timesteps_tensor, out=self.p_spheres_tmp_time_filtered_buf) # Select timesteps
        torch.index_select(self.p_spheres_tmp_time_filtered_buf, 2, self.valid_own_spheres_tensor, out=self.p_own_filtered_buf)
        
        # In-place ULTRA-FAST transformation directly into the pre-allocated broadcast buffer
        p_own_filtered_flat = self.p_own_filtered_buf.view(-1, 3)
        p_own_buf_broadcast_flat = self.p_own_buf_broadcast.view(-1, 3)
        
        torch.addmm(
            self.world_translation,
            p_own_filtered_flat, 
            self.rotation_matrix.T,
            out=p_own_buf_broadcast_flat
        )
        
        # Fused difference computation: own_pos - obs_pos
        # Broadcasting: [n_rollouts, n_sampling_steps, n_valid_own, 1, 3] - [1, n_sampling_steps, 1, n_valid_obs, 3]
        torch.sub(self.p_own_buf_broadcast, self.p_obs_buf_broadcast, out=self.ownobs_diff_vector_buff)
        
        self._debug['p_own'] = self.p_own_buf_broadcast.squeeze(3)
        self._debug['r_own'] = self.rad_own_buf
        self._debug['p_obs'] = self.p_obs_buf_broadcast.squeeze(0).squeeze(1)
        self._debug['r_obs'] = self.rad_obs_buf

        # Compute L2 norm (distance between sphere centers)
        torch.norm(self.ownobs_diff_vector_buff, dim=-1, keepdim=True, out=self.pairwise_surface_dist_buf)
        ## Subtract radius sums to get surface-to-surface distance
        self.pairwise_surface_dist_buf.sub_(self.pairwise_radsum_broadcast)

        # _debug_using_wta = self.prior_rule == 'pose_wta' 
        # print(f'DEBUG: using wta: {_debug_using_wta}')
        # print(f'trust: {self.wta_trust}')
        # print(f'weight: {self.cost_weight}')
        
        # Reset RRC buffer (M: NXH) 
        # self.cost_type = 'linear'
        self.tmp_cost_mat_buf_sparse.zero_() # set reset cost matrix buffer (rows are rollouts, cols are sparse steps)
        
        if self.cost_type == 'storm_binary': # 1 where collision, 0 where not
            print(f'DEBUG: in storm_binary cost type')
            min_dist_surf2surf = torch.amin(self.pairwise_surface_dist_buf, dim=tuple(range(2, self.pairwise_surface_dist_buf.ndim)))
            self.tmp_cost_mat_buf_sparse.copy_(min_dist_surf2surf.lt(0.01)) 
            # print(f'DEBUG: tmp_cost_mat_buf_sparse max: {self.tmp_cost_mat_buf_sparse.max()}, min: {self.tmp_cost_mat_buf_sparse.min()}')
            
        elif self.cost_type == 'linear':
            # print(f'DEBUG: in linear cost type')
            if self.prior_rule == 'pose_wta' and len(self.pose_wta_conflict_resolution): # pose wta conflict resolution is used
            
                p_own_err, q_own_err = self.pose_wta_conflict_resolution['own_errors']
                subto_goal_errs = self.pose_wta_conflict_resolution['subto_errors']
                
                for st_idx in subto_goal_errs.keys(): # for every robot self is subscribed to
                    p_err_subto, q_err_subto = subto_goal_errs[st_idx]
                    # if p_err_subto > p_own_err: # self has lower error than subto, so self is the winner. Self will set subto's distance to a very high distance to ignore it in action
                    # subto_indices = self.col_with_idx_map[st_idx]['valid_indices']
                    robot_map = self.col_with_idx_map[st_idx]
                    start_idx_subto = robot_map['start_idx']
                    end_idx_subto = robot_map['end_idx']

                    # We now set self.pairwise_surface_dist_buf to a very high distance for the subto spheres (to ignore them in collision check and prioritize ourselves on top of them)
                    
                    # Make prioritization expression (alpha) between self and other (subto) = err_ratio ** tau
                    err_ratio = p_own_err / p_err_subto
                    # print(f'err_ratio: {err_ratio}')   
                    tau = self.wta_trust
                    alpha = err_ratio ** tau 
                    # print(f'alpha: {alpha}')

                    
                    # Make masking f:
                    # now we ge the min distance to any st (subto) sphere on each rollout (dim 0) step h (dim 1)
                    with torch.no_grad():  # Reduce memory overhead for intermediate computations
                        min_dcol_self_subto_s2s = torch.amin(self.pairwise_surface_dist_buf[:, :, :, start_idx_subto:end_idx_subto, :], dim=tuple(range(2, self.pairwise_surface_dist_buf.ndim)))
                        torch.relu_(min_dcol_self_subto_s2s)  # In-place operation to reduce memory
                        th = self.safety_margin # safety margin, which if distance < th we start to penalize
                        
                        d_col = min_dcol_self_subto_s2s # min distance to a sphere of subto (surface to surface distance)
                        # Inline in-place f_mask: f = 1.0 + (-(1.0 / margin)) * x
                        f = d_col.mul_(-(1.0 / th)).add_(1.0)
                        f.relu_() # A matrix with values >= 0, where > 0 means collision/margin breach
                        penalty_from_subto = f.mul_(alpha) # penalize self 
                        self.tmp_cost_mat_buf_sparse.add_(penalty_from_subto)
                        
            else: # without GPDB
                # raise ValueError("This is deprecated, should not be used")
                pass 
                print(f'WARNING: NO COST WAS CALCULATED')
                # min_dist_surf2surf = torch.amin(self.pairwise_surface_dist_buf, dim=tuple(range(2, self.pairwise_surface_dist_buf.ndim)))
                # # Remove negative distances ("sphere to sphere penetrations", replace them with 0)
                # min_dist_surf2surf = torch.max(min_dist_surf2surf, torch.zeros_like(min_dist_surf2surf)) # negative values (distances) mean intersection between spheres. We set distance to 0 instead, treating it just as contact between spheres.
                # self.tmp_cost_mat_buf_sparse = torch.ones_like(min_dist_surf2surf) / (min_dist_surf2surf + 1e-6) # cost[i,j] = 1 / min distance[i,j] 
                
                # # IF NOT USING GPDB PRIORITIZATION: MASK OUT (SET cost to0) WHEN SELF IS FAR ENOUGH FROM ANY OTHER ROBOT (SURFACE DIST >= SAFETY MARGIN)    
                # # Make a mask for the safety margin, to avoid punishing robot for being far enough (beyond margin) from other robots.
                # margin_mask = min_dist_surf2surf.lt(self.safety_margin).float() # 1 where minimal distance is less than required safety margin, 0 otherwise
                # # Mask out the cost where collision distance >  safety margin
                # self.tmp_cost_mat_buf_sparse.mul_(margin_mask) # set cost to 0 where collision distance >  safety margin

        
        # INTERPOLATION (OPTIONAL)-
        # IF SPARSE STEPS IS ON, INTERPOLATE
        # Interpolate the sparse costs over the horizon: (Project sparse results to full horizon, to get a valid cost matrix for the whole horizon)
        # print(f'torch.isnan(self.tmp_cost_mat_buf_sparse).any(): {torch.isnan(self.tmp_cost_mat_buf_sparse).any()}')
        
        self._project_sparse_to_full_horizon(self.tmp_cost_mat_buf_sparse, self.cost_mat_buf)

    
        # WEIGHTING:
        # MULTIPLY BY CONSTANT WEIGHT OF COST TERM (HYPER PARAM) 
        self.cost_mat_buf.mul_(self.cost_weight)
        # print(f'DEBUG: cost_mat_buf max: {self.cost_mat_buf.max()}, min: {self.cost_mat_buf.min()}')
        if torch.isnan(self.cost_mat_buf).any():
            print(f'DEBUG: cost_mat_buf has nan values!')
            return torch.zeros_like(self.cost_mat_buf)
        # CLAMPING: 
        # set upper bound to 100,000 to avoid numerical issues        
        self.cost_mat_buf.clamp_(max=100_000)
        
        # Clear intermediate computation results to free GPU memory
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()
        #     # Force synchronize to ensure memory is actually freed
        #     torch.cuda.synchronize()
        
        end_cost_fn_time = time.time()
        # print(f'DEBUG: cost_fn time: {end_cost_fn_time - start_cost_fn_time}')
        self.wall_clock_time += end_cost_fn_time - cost_fn_start_time
        self.wall_clock_steps += 1
        if self.wall_clock_steps % 100 == 0:
            print(f'DEBUG: cost_fn wall clock freq: {self.wall_clock_steps / self.wall_clock_time}')
        return self.cost_mat_buf

    
    
    # def cost_fn(self, prad_own_R: torch.Tensor):
    #     """Ultra-optimized collision cost computation."""
        
        
            
    #     # Update transformation matrix if needed
    #     if self.transform_matrix_dirty or self.rotation_matrix is None:
    #         self.update_world_pose(self.X)
        
    #     # Ensure transformation matrices are available
    #     if self.rotation_matrix is None or self.world_translation is None:
    #         raise RuntimeError("Transformation matrices not initialized")
        
    #     # Extract positions from robot frame poses
    #     robot_positions = prad_own_R[:, :, :, :3]  # (n_rollouts, H, n_spheres, 3)
        
    #     # ULTRA-FAST transformation using pre-computed matrix
    #     self.p_own_buf_unfiltered = transform_positions_with_precomputed_matrix(
    #         robot_positions, self.rotation_matrix, self.world_translation
    #     )
        
    #     # filter out the timesteps that are not needed
    #     p_spheres_tmp_time_filtered = torch.index_select(self.p_own_buf_unfiltered, 1, self.sampling_timesteps_tensor) # Select timesteps
    #     # filter out the spheres that are not needed
    #     self.p_own_buf = torch.index_select(p_spheres_tmp_time_filtered, 2, self.valid_own_spheres_tensor)
        
    #     # Update broadcast-ready buffer
    #     self.p_own_buf_broadcast[:, :, :, 0, :] = self.p_own_buf
        
    #     # Fused difference computation: own_pos - obs_pos
    #     # Broadcasting: [n_rollouts, n_sampling_steps, n_valid_own, 1, 3] - [1, n_sampling_steps, 1, n_valid_obs, 3]
    #     torch.sub(self.p_own_buf_broadcast, self.p_obs_buf_broadcast, out=self.ownobs_diff_vector_buff)
        
    #     self._debug['p_own'] = self.p_own_buf_broadcast.squeeze(3).cpu().numpy()
    #     self._debug['r_own'] = self.rad_own_buf.cpu().numpy()
    #     self._debug['p_obs'] = self.p_obs_buf_broadcast.squeeze(0).squeeze(1).cpu().numpy()
    #     self._debug['r_obs'] = self.rad_obs_buf.cpu().numpy()

    #     # Compute L2 norm (distance between sphere centers)
    #     torch.norm(self.ownobs_diff_vector_buff, dim=-1, keepdim=True, out=self.pairwise_surface_dist_buf)
    #     ## Subtract radius sums to get surface-to-surface distance
    #     self.pairwise_surface_dist_buf.sub_(self.pairwise_radsum_broadcast)

    #     # _debug_using_wta = self.prior_rule == 'pose_wta' 
    #     # print(f'DEBUG: using wta: {_debug_using_wta}')
    #     # print(f'trust: {self.wta_trust}')
    #     # print(f'weight: {self.cost_weight}')
        
    #     # Reset RRC buffer (M: NXH) 
    #     self.tmp_cost_mat_buf_sparse.zero_() # set reset cost matrix buffer (rows are rollouts, cols are sparse steps)
        
    #     if self.cost_type == 'storm_binary': # 1 where collision, 0 where not
    #         min_dist_surf2surf = torch.amin(self.pairwise_surface_dist_buf, dim=tuple(range(2, self.pairwise_surface_dist_buf.ndim)))
    #         self.tmp_cost_mat_buf_sparse = torch.ones_like(min_dist_surf2surf) 
    #         mask_out_of_collision = min_dist_surf2surf.lt(0.01).float() # 1 where minimal distance is less than required safety margin, 0 otherwise
    #         self.tmp_cost_mat_buf_sparse.mul_(mask_out_of_collision) # set cost to 0 where collision distance >  safety margin
    #         print(f'DEBUG: tmp_cost_mat_buf_sparse max: {self.tmp_cost_mat_buf_sparse.max()}, min: {self.tmp_cost_mat_buf_sparse.min()}')
            
    #     elif self.cost_type == 'linear':
    #         if self.prior_rule == 'pose_wta' and len(self.pose_wta_conflict_resolution): # pose wta conflict resolution is used
            
    #             p_own_err, q_own_err = self.pose_wta_conflict_resolution['own_errors']
    #             subto_goal_errs = self.pose_wta_conflict_resolution['subto_errors']
                
    #             for st_idx in subto_goal_errs.keys(): # for every robot self is subscribed to
    #                 p_err_subto, q_err_subto = subto_goal_errs[st_idx]
    #                 # if p_err_subto > p_own_err: # self has lower error than subto, so self is the winner. Self will set subto's distance to a very high distance to ignore it in action
    #                 # subto_indices = self.col_with_idx_map[st_idx]['valid_indices']
    #                 robot_map = self.col_with_idx_map[st_idx]
    #                 start_idx_subto = robot_map['start_idx']
    #                 end_idx_subto = robot_map['end_idx']

    #                 # We now set self.pairwise_surface_dist_buf to a very high distance for the subto spheres (to ignore them in collision check and prioritize ourselves on top of them)
                    
    #                 # Make prioritization expression (alpha) between self and other (subto) = err_ratio ** tau
    #                 err_ratio = p_own_err / p_err_subto   
    #                 tau = self.wta_trust
    #                 alpha = err_ratio ** tau 
                    
    #                 # Make masking f:
    #                 # now we ge the min distance to any st (subto) sphere on each rollout (dim 0) step h (dim 1)
    #                 with torch.no_grad():  # Reduce memory overhead for intermediate computations
    #                     min_dcol_self_subto_s2s = torch.amin(self.pairwise_surface_dist_buf[:, :, :, start_idx_subto:end_idx_subto, :], dim=tuple(range(2, self.pairwise_surface_dist_buf.ndim)))
    #                     torch.relu_(min_dcol_self_subto_s2s)  # In-place operation to reduce memory
    #                     th = self.safety_margin # safety margin, which if distance < th we start to penalize
    #                     d_col = min_dcol_self_subto_s2s # min distance to a sphere of subto (surface to surface distance)
    #                     f = f_mask(d_col, th) # A matrix with values between 0 and 1, where 1 means collision, 0 means dist(self, subto) >= th (>= safety margin)
    #                     penalty_from_subto = alpha * f # penalize self 
    #                     self.tmp_cost_mat_buf_sparse += penalty_from_subto
                        
    #         else: # without GPDB
    #             # raise ValueError("This is deprecated, should not be used")
    #             pass 
    #             print(f'WARNING: NO COST WAS CALCULATED')
    #             # min_dist_surf2surf = torch.amin(self.pairwise_surface_dist_buf, dim=tuple(range(2, self.pairwise_surface_dist_buf.ndim)))
    #             # # Remove negative distances ("sphere to sphere penetrations", replace them with 0)
    #             # min_dist_surf2surf = torch.max(min_dist_surf2surf, torch.zeros_like(min_dist_surf2surf)) # negative values (distances) mean intersection between spheres. We set distance to 0 instead, treating it just as contact between spheres.
    #             # self.tmp_cost_mat_buf_sparse = torch.ones_like(min_dist_surf2surf) / (min_dist_surf2surf + 1e-6) # cost[i,j] = 1 / min distance[i,j] 
                
    #             # # IF NOT USING GPDB PRIORITIZATION: MASK OUT (SET cost to0) WHEN SELF IS FAR ENOUGH FROM ANY OTHER ROBOT (SURFACE DIST >= SAFETY MARGIN)    
    #             # # Make a mask for the safety margin, to avoid punishing robot for being far enough (beyond margin) from other robots.
    #             # margin_mask = min_dist_surf2surf.lt(self.safety_margin).float() # 1 where minimal distance is less than required safety margin, 0 otherwise
    #             # # Mask out the cost where collision distance >  safety margin
    #             # self.tmp_cost_mat_buf_sparse.mul_(margin_mask) # set cost to 0 where collision distance >  safety margin

        
    #     # INTERPOLATION (OPTIONAL)-
    #     # IF SPARSE STEPS IS ON, INTERPOLATE
    #     # Interpolate the sparse costs over the horizon: (Project sparse results to full horizon, to get a valid cost matrix for the whole horizon)
    #     self._project_sparse_to_full_horizon(self.tmp_cost_mat_buf_sparse, self.cost_mat_buf)
    
    #     # WEIGHTING:
    #     # MULTIPLY BY CONSTANT WEIGHT OF COST TERM (HYPER PARAM) 
    #     self.cost_mat_buf.mul_(self.cost_weight)

    #     # CLAMPING: 
    #     # set upper bound to 100,000 to avoid numerical issues        
    #     self.cost_mat_buf = torch.min(self.cost_mat_buf, torch.ones_like(self.cost_mat_buf) * 100_000)
        
    #     # Clear intermediate computation results to free GPU memory
    #     if hasattr(torch.cuda, 'empty_cache'):
    #         torch.cuda.empty_cache()
    #         # Force synchronize to ensure memory is actually freed
    #         torch.cuda.synchronize()
        
    #     return self.cost_mat_buf

    def update_robot_spheres(self, robot_id: int, p_robot_spheres: torch.Tensor):
        """
        Update the positions of spheres for a specific "other" robot in the concatenated obstacle tensor.
        
        Args:
            robot_id: ID of the robot whose spheres to update
            p_robot_spheres: tensor of shape [H, n_robot_spheres, 3]. The sphere positions for this robot.
        """
        if robot_id not in self.col_with_idx_map:
            print(f"Warning: Robot {robot_id} not found in collision mapping")
            return
        
        robot_map = self.col_with_idx_map[robot_id]
        start_idx = robot_map['start_idx']
        end_idx = robot_map['end_idx']
        valid_indices = robot_map['valid_indices']
        
        # Filter robot spheres based on its exclusion list (if sparse mode is used)
        filtered_spheres = p_robot_spheres[:, valid_indices, :]  # [H, n_valid_robot_spheres, 3]
        
        # Then filter by timesteps (if sparse mode is used)
        filtered_spheres = torch.index_select(filtered_spheres, 0, self.sampling_timesteps_tensor)  # [n_sampling_steps, n_valid_robot_spheres, 3]
        
        # Update the relevant slice of the obstacle buffer
        self.p_obs_buf[:, start_idx:end_idx, :] = filtered_spheres
        
        # Update broadcast-ready buffer
        self.p_obs_buf_broadcast[0, :, 0, start_idx:end_idx, :] = filtered_spheres



if __name__ == "__main__":
    x = [0] + list(np.cumsum([1,3,4]))[:-1]
    print(x)