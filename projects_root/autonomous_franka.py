try:
    import isaacsim
except ImportError:
        pass

from typing import List, Optional
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from curobo.rollout.arm_reacher import ArmReacher
from curobo.rollout.cost.custom import arm_reacher
from curobo.wrap.wrap_mpc import WrapMpc
import torch
from typing import Callable, Dict, Union
import carb
import numpy as np
from abc import abstractmethod
from torch.func import vmap


from omni.isaac.core import World # https://forums.developer.nvidia.com/t/cannot-import-omni-isaac-core/242977/3
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.objects import DynamicCuboid
from isaacsim.util.debug_draw import _debug_draw # isaac 4.5
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.types import JointsState as isaac_JointsState
import omni.kit.commands as cmd
from pxr import Gf

from projects_root.utils.helper import add_robot_to_scene
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.frame_utils import FrameUtils
from projects_root.utils.transforms import transform_pose_between_frames, transform_poses_batched
# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Sphere, WorldConfig, Cuboid, Mesh
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
# from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import load_yaml
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.tensor import T_DOF
from curobo.types.state import FilterCoeff
from curobo.wrap.reacher.motion_gen import (MotionGen,MotionGenConfig,MotionGenPlanConfig, MotionGenResult,PoseCostMetric,)
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, PoseCostMetric
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import DynamicObsCollPredictor

def spawn_target(path="/World/target", position=np.array([0.5, 0.0, 0.5]), orientation=np.array([0, 1, 0, 0]), color=np.array([0, 1, 0]), size=0.05):
    """ 
    Create a target pose "hologram" in the simulation. By "hologram", 
    we mean a visual representation of the target pose that is not used for collision detection or physics calculations.
    In isaac-sim they call holograms viual objects (like visual coboid and visual spheres...)
    """
    target = cuboid.VisualCuboid(
        path,
        position=position,
        orientation=orientation,
        color=color,
        size=size,
    )
    
    return target

class AutonomousFranka:
    
    instance_counter = 0
    def __init__(self,robot_cfg, world:World, p_R=np.array([0.0,0.0,0.0]),q_R=np.array([1,0,0,0]), p_T_R=np.array([0.5,0.0,0.5]), q_T_R=np.array([0, 1, 0, 0]), target_color=np.array([0.0,1.0,0.0]), target_size=0.05):
        """
        Spawns a franka robot in the scene andd setting the target for the robot to follow.
        All notations will follow Drake. See: https://drake.mit.edu/doxygen_cxx/group__multibody__quantities.html#:~:text=Typeset-,Monogram,-Meaning%E1%B5%83
        https://drake.mit.edu/doxygen_cxx/group__multibody__notation__basics.html https://drake.mit.edu/doxygen_cxx/group__multibody__frames__and__bodies.html
        Args:
            world (_type_): simulator world instance.
            p_R: Position vector from Wo (W (World frame) origin (o)) to Ro (R's origin (robot's base frame)), expressed in the world frame W (implied).
            q_R: Frame R's (second R, representing robot's base frame) orientation (first R, representing rotation) in the world frame W  . Quaternion (w,x,y,z)
            p_T_R: Position vector from Wo (W (World frame) origin (o)) to To (target's origin), expressed in the robot frame R.
            q_T_R: Frame T's (representing target's base frame) orientation (R, representing rotation) in the robot frame . Quaternion (w,x,y,z)

        """
        # simulator paths etc.
        self.instance_id = AutonomousFranka.instance_counter
        self.world = world
        self.world_root ='/World'
        self.robot_name = f'robot_{self.instance_id}'
        self.subroot_path = f'{self.world_root}/world_{self.robot_name}'

        self.n_coll_spheres = 65 # num of self spheres which can be used for collision checking
        self.valid_coll_spheres_idx = np.arange(self.n_coll_spheres - 4)
        self.n_coll_spheres_valid = self.n_coll_spheres - 4 # Should be number of valid spheres of the robot (ignoring 4 spheres which are not valid due to negative radius)
        
        # robot base frame settings (static, since its an arm and not a mobile robot. Won't change)
        self.p_R = p_R  
        self.q_R = q_R 
        self.X_R = list(p_R) + list(q_R) # robot base frame in world frame
        
        # target settings
        X_target_R = list(p_T_R) + list(q_T_R)
        X_target_W = transform_pose_between_frames(X_target_R, self.X_R) # target frame from robot frame to world frame
        self._p_initTarget = X_target_W[:3] #.position # initial target frame position (expressed in the world frame W)
        self._q_initTarget = X_target_W[3:] #.quaternion # initial target frame rotation (expressed in the world frame W)
        
        self.initial_target_color = target_color
        self.initial_target_size = target_size
        
        self.p_solverTarget = None # current target position of robot as set to the planner (solver). Not necessarily synced with (may be behind of) the target in the scene.
        self.q_solverTarget = None # current target orientation of robot as set to the planner (solver). Not necessarily synced with (may be behind of) the target in the scene.

        self.robot_cfg = robot_cfg # the section under the key 'robot_cfg' in the robot config file (yml). https://curobo.org/tutorials/1_robot_configuration.html#tut-robot-configuration
        self.j_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"] # joint names for the robot
        self.initial_joint_config = self.robot_cfg["kinematics"]["cspace"]["retract_config"] # initial ("/retract") joint configuration for the robot
        self.tensor_args = TensorDeviceType()

        # self.cu_stat_obs_world_model = world_collision_model # self._init_curobo_stat_obs_world_model() # will be initialized in the _init_curobo_stat_obs_world_model method. Static obstacles world configuration for curobo collision checking.
        self.solver = None # will be initialized in the init_solver method.
        self._vis_spheres = None # for visualization of robot spheres
        self.crm = CudaRobotModel(CudaRobotModelConfig.from_data_dict(self.robot_cfg)) # https://curobo.org/_api/curobo.cuda_robot_model.cuda_robot_model.html#curobo.cuda_robot_model.cuda_robot_model.CudaRobotModelConfig
        self.obs_viz = [] # for visualization of robot spheres
        self.obs_viz_obs_names = []
        self.obs_viz_prim_path = f'/obstacles/{self.robot_name}'
        AutonomousFranka.instance_counter += 1
    
    def get_num_of_sphers(self, valid_only:bool=True):
        return self.n_coll_spheres if not valid_only else self.n_coll_spheres_valid
    
    def get_world_model(self):
        return self.solver.world_coll_checker.world_model

    def get_cchecker(self):
        return self.solver.world_coll_checker
    
    def _spawn_robot_and_target(self, usd_help:UsdHelper):
        X_R = Pose.from_list(list(self.p_R) + list(self.q_R)) # 
        usd_help.add_subroot(self.world_root, self.subroot_path, X_R)
        
        self.robot, self.prim_path = add_robot_to_scene(self.robot_cfg, self.world, subroot=self.subroot_path+'/', robot_name=self.robot_name, position=self.p_R, orientation=self.q_R, initialize_world=False) # add_robot_to_scene(self.robot_cfg, self.world, robot_name=self.robot_name, position=self.p_R)
        self.target = spawn_target(self.world_root+f'/{self.robot_name}_target', self._p_initTarget, self._q_initTarget, self.initial_target_color, self.initial_target_size)
        
  
    @abstractmethod
    def _check_prerequisites_for_syncing_target_pose(self, real_target_position:np.ndarray, real_target_orientation:np.ndarray,sim_js:None) -> bool:
        pass

    def set_new_target_for_solver(self, real_target_position:np.ndarray, real_target_orientation:np.ndarray,sim_js=None):
        
        """ 
        Checking if the target pose in the solver needs to be reset. 
        If it needs to be reset, the target pose to the solver is re-written and the method returns True.
        Otherwise, the method returns False.
        Anyway, we are not resetting the target pose in the solver, but only updating it if needed.
        The update in the solver will be done on a seperate method (here we just check if the update is needed and update the fields the solver will take the new values).


        real_target_position: target position in world frame
        real_target_orientation: target orientation in world frame
        real_target_position_prev_ts: target position in previous time step
        real_target_orientation_prev_ts: target orientation in previous time step
        sim_js: joint state of the robot (in simulation representation, not curobo representation)

        Returns:
            _type_: _description_
        """
        
        if self.p_solverTarget is None:
            self.p_solverTarget = real_target_position + 1.0 # to force the first sync
            self.q_solverTarget = real_target_orientation
            
        sync_target = self._check_prerequisites_for_syncing_target_pose(real_target_position, real_target_orientation, sim_js)

        if sync_target:
            self.p_solverTarget = real_target_position
            self.q_solverTarget = real_target_orientation
            return True
        
        else:
            return False
    
    def get_last_synced_target_pose(self):
        return Pose(position=self.tensor_args.to_device(self.p_solverTarget),quaternion=self.tensor_args.to_device(self.q_solverTarget),)
            
    def _post_init_solver(self):
        return None

    @abstractmethod
    def init_solver(self, *args, **kwargs):
        pass
    
    def _check_target_pose_changed(self, real_target_position, real_target_orientation) -> bool:
        return np.linalg.norm(real_target_position - self.p_solverTarget) > 1e-3 or np.linalg.norm(real_target_orientation - self.q_solverTarget) > 1e-3

    def _check_target_pose_static(self, real_target_position, real_target_orientation) -> bool:
        if not hasattr(self, '_real_target_pos_prev_t'):
            self._real_target_pos_prev_t = real_target_position
            self._real_target_orient_prev_t = real_target_orientation
        
        is_static = np.linalg.norm(real_target_position - self._real_target_pos_prev_t) == 0.0 and np.linalg.norm(real_target_orientation - self._real_target_orient_prev_t) == 0.0
        self._real_target_pos_prev_t = real_target_position
        self._real_target_orient_prev_t = real_target_orientation
        return is_static
    
    def update_real_target(self, real_target_position, real_target_orientation):
        self.target = spawn_target(self.target_path, real_target_position, real_target_orientation, self.initial_target_color, self.initial_target_size)
        
    def _check_robot_static(self, sim_js) -> bool:
        return np.max(np.abs(sim_js.velocities)) < 0.2
    
    def init_joints(self, idx_list:list):
        """Set the maximum efforts for the robot.
        Args:
          
        """
        # robot.robot._articulation_view.initialize()
        self.robot.set_joint_positions(self.initial_joint_config, idx_list) 
        self.robot._articulation_view.set_max_efforts(values=np.array([5000 for _ in range(len(idx_list))]), joint_indices=idx_list)

    def get_robot_as_spheres(self, cu_js, express_in_world_frame=False) -> list[Sphere]:
        """Get the robot as spheres from the curobot joints state.
        # NOTE: spheres are expressed in the robot base frame and not in the world frame. Shifting to the world frame requires adding the robot base frame position to the sphere position.
        Args:
            cu_js (_type_): curobo joints state
        Returns:
            list[Sphere]: list of spheres
        """
        assert isinstance(self.solver, MpcSolver) or isinstance(self.solver, MotionGen), "Solver not initialized"
        sph_list = self.solver.kinematics.get_robot_as_spheres(cu_js.position)[0] # at this point each sph.position is expressed in the robot base frame, not in the world frame
        if express_in_world_frame:
            for sph in sph_list:
                sph.position = sph.position + self.p_R # express the spheres in the world frame
                sph.pose[:3] = sph.pose[:3] + self.p_R
        return sph_list

    def visualize_robot_as_spheres(self, cu_js):
        if cu_js is None:
            return
        sph_list = self.get_robot_as_spheres(cu_js, express_in_world_frame=True)
        if self._vis_spheres is None: # init visualization spheres
            self._vis_spheres = []
            for si, s in enumerate(sph_list):
                sp = sphere.VisualSphere(
                            prim_path=f"/curobo/robot_{self.instance_id}_sphere_" + str(si),
                            position=np.ravel(s.position),
                            radius=float(s.radius),
                            color=np.array([0, 0.8, 0.2]),
                        )
                self._vis_spheres.append(sp)

        else: # update visualization spheres
            for si, s in enumerate(sph_list):
                if not np.isnan(s.position[0]):
                    self._vis_spheres[si].set_world_pose(position=np.ravel(s.position))
                    self._vis_spheres[si].set_radius(float(s.radius))

    def update_obs_viz(self,p_spheres:torch.Tensor):
        for i in range(len(self.obs_viz)):
            self.obs_viz[i].set_world_pose(position=np.array(p_spheres[i].tolist()), orientation=np.array([1., 0., 0., 0.]))

    def add_obs_viz(self,p_sphere:torch.Tensor,rad_sphere:torch.Tensor, obs_name:str,h=0,h_max=30,material=None):
        
        obs_viz = VisualSphere(
            prim_path=f"{self.obs_viz_prim_path}/{obs_name}",
            position=np.ravel(p_sphere),
            radius=float(rad_sphere),
            color=np.array([1-(h/h_max),1-(h/h_max),1-(h/h_max)]),
            visual_material = material
            )
        self.obs_viz.append(obs_viz)

    def get_dof_names(self):
        return self.robot.dof_names

    def get_sim_joint_state(self) -> isaac_JointsState:
        return self.robot.get_joints_state()
    
    def get_curobo_joint_state(self, sim_js, zero_vel:bool) -> JointState:
        """Returns the curobo joint configuration (robot joint state represented as a JointState object,
        which is curobo's representation of the robot joint state) from the simulation joint state (the joint state of the robot as returned by the simulation).
        For more details about JointState see https://curobo.org/advanced_examples/4_robot_segmentation.html
        Args:
            sim_js (_type_): the joint state of the robot as returned by the simulation.
            zero_vel (bool): should multiply the velocities by 0.0 to set them to zero (differs for robot types - MPC and Cumotion).

        Returns:
            JointState: the robot's joint configuration in curobo's representation.
        """
        if sim_js is None:
            sim_js = self.get_sim_joint_state()
        position = self.tensor_args.to_device(sim_js.positions)
        velocity = self.tensor_args.to_device(sim_js.velocities) * 0.0 if zero_vel else self.tensor_args.to_device(sim_js.velocities)
        acceleration = self.tensor_args.to_device(sim_js.velocities) * 0.0
        jerk = self.tensor_args.to_device(sim_js.velocities) * 0.0
        cu_js = JointState(position=position,velocity=velocity,acceleration=acceleration,jerk=jerk,joint_names=self.get_dof_names()) # joint_names=self.robot.dof_names) 
        return cu_js
    

    def get_current_spheres_state(self,express_in_world_frame:bool=True, valid_only=True,zero_vel=False):
        cu_js = self.get_curobo_joint_state(self.get_sim_joint_state(),zero_vel=zero_vel) # zero vel doesent matter since we are getting sphere poses and radii
        link_spheres_R = self.crm.compute_kinematics_from_joint_state(cu_js).get_link_spheres()
        p_link_spheres_R = link_spheres_R[:,:,:3].cpu() # position of spheres expressedin robot base frame
        if express_in_world_frame:
            p_link_spheres_W = p_link_spheres_R + self.p_R
            p_link_spheres_F = p_link_spheres_W
        else:
            p_link_spheres_F = p_link_spheres_R
        p_link_spheres_F = p_link_spheres_F.squeeze(0)
        rad_link_spheres = link_spheres_R[:,:,3].cpu().squeeze(0)
        
        if valid_only:
            sphere_indices = torch.nonzero(rad_link_spheres > 0, as_tuple=True)[0] # valid is positive radius
            p_link_spheres_F = p_link_spheres_F[sphere_indices]
            rad_link_spheres = rad_link_spheres[sphere_indices]
        else:
            sphere_indices = torch.arange(p_link_spheres_F.shape[0]) # all sphere indices

        return p_link_spheres_F, rad_link_spheres, sphere_indices
        
    @abstractmethod
    def apply_articulation_action(self, art_action:ArticulationAction):
        pass

    def get_H_steps_plan(self, in_task_space:bool, H:int=-1):
        """Get the H steps plan from the solver.
        Args:
            in_task_space (bool): if True, return the plan in task space, otherwise return the plan in joint space.
            H (int, optional): the number of steps in the plan. If -1, return the entire plan. Defaults to -1.
        Returns:
            torch.Tensor: the H steps plan.
        """
        pass
    

    def integrate_acc(self, acceleration: T_DOF,cmd_joint_state: JointState, dt_planning: float) -> JointState:
        """
        This function integrates the acceleration to get the velocity and position of the joint state.
        Given a joint state and an acceleration, it returns the next joint state after integrating the acceleration.
        
        NOTE! This function is a cloned version with no significant changes of the function "integrate_acc" in curobo/src/curobo/util/state_filter.py.
        The reason for cloning this is because when calling the original function, it changes internal attributes of the kinematics_model object and this is not something we wanted.
        All the changes we made here compared to the original function are just to avoid changing the internal attributes of the kinematics_model object.
        The logic of the function is the same as in the original function,
        The original function is called during the MPC solver step, in order to compute how the acceleration (which is the mpc policy action) in a given input joint state will affect the state and change the joints velocity and position, 
        explained in more detail in the  "how it works" section below.


        
        
        
        How it works?
        First, the integration causes the next joint state's velocity to be the current velocity plus the acceleration times the time step. 
        That means that we just change our cosntant velocity at previous state to a new constant velocity which is the previous velocity plus the acceleration times the duration of the step (dt_planning).
        Then, we compute the new joint position as the previous position plus the velocity times the duration of the step (dt_planning).
        
        Example:
            In:
                acceleration = [0.1, -0.2, -0.4, 0.5, 0.0, 0.4, 0.8] # the input command -(7 dof). rad/sec^2
                current_joint_state: # start state
                    .acceleration = [WE DONT CARE] (because we override it in the new state)
                    .velocity = [2.0, 0.4, 0.2, -0.4, -0.4, 1.5, 0.1] # the velocity of the joints at the start state (rad/sec)            
                    .position = [-0.5, 0.0, 0.2, 0.0, -0.3, 0.2, 0.0] # the position of the joints at the start state (rad)
                    
                dt_planning = 0.1 # normally less, 0.5 only for the example
            Out:
                new_joint_state.acceleration # the acceleration of the joints at the beginning of the next step:  [0.1, -0.2, -0.4, 0.5, 0.0, 0.4, 0.8] # copied from the input command
                new_joint_state.velocity # the velocity of the joints at the beginning of the next step: [2.01, 0.38, 0.16, -0.35, -0.4, 1.54, 0.18] (Way of computing: [2.0 + 0.1*0.1, 0.4 + -0.2*0.1, 0.2 + -0.4*0.1, -0.4 + 0.5*0.1, -0.4 + 0.0*0.1, 1.5 + 0.4*0.1, 0.1 + 0.8*0.1])
                new_joint_state.position # the position of the joints at the beginning of the next step: [-0.299,  0.038,  0.216, -0.035, -0.34 ,  0.354,  0.018] (Way of computing: [for i in range(len(current_joint_state.position)): current_joint_state.position[i] + new_joint_state.velocity[i]*0.1])
                
            
        Args:
            acceleration (T_DOF): A tensor of shape (num of dogs,) representing the acceleration action (command) to apply to the joint state.
            cmd_joint_state (JointState): some joint state of the robot to apply the acceleration to (contains it's current position, velocity, acceleration, jerk).
            dt_planning (float): the duration (in seconds) of the step to integrate the acceleration over. Necessary to determine the new state's position (after the acceleration to the joints was applied, and therefore the velocity is changed).
            Normally, set this to the the time you assume that elapses between each two consecutive steps in the control loop (for example in the MPC case btw, this is also the time that the planner considers between each two consecutive steps in the horizon, passed by step_dt in the MpcSolverConfig).


        Returns:
            JointState: The new joint state (the state at the beginning of the next step (at the end of the current step),
            after integrating the new acceleration for the duration of dt_planning).
        """
        next_joint_state = cmd_joint_state.clone() # next joint state after integrating the acceleration
        next_joint_state.acceleration[:] = acceleration # set the acceleration at the new state to the input command acceleration
        next_joint_state.velocity[:] = cmd_joint_state.velocity + next_joint_state.acceleration * dt_planning # compute the new velocity given the current velocity, the acceleration and the dt which means for how long the acceleration is applied
        next_joint_state.position[:] = cmd_joint_state.position + next_joint_state.velocity * dt_planning # compute the new position given the current position and the new velocity
        if cmd_joint_state.jerk is None:
            next_joint_state.jerk = acceleration * 0.0 # it's not used for computations, but it has to be initiated to avoid exceptions
        return next_joint_state
    
    def filter_joint_state(self, js_state_prev:Union[JointState,None], js_state_new:JointState, filter_coefficients:FilterCoeff):
        """ Reducing the new state by a weighted sum of the previous state and the new state (like a step size to prevent sharp changes).
        
        # NOTE! Similar to integrate_acc, this is a cloned version of another function from original curobo code.
        As in integrate_acc, the logic of the function is the same as in the original function, and it was cloned to avoid changing the internal attributes and cause unexpected side effects.

        The original function is: filter_joint_state at curobo/src/curobo/util/state_filter.py (row 60)
        The original function is used by the mpc solver to reduce the sharpness of changes between following joint states, to simulate a smoother and more realistic robot movement () .
        Its invoking the blend() function at curobo/src/curobo/types/state.py row 170 
        The goal of the original function is to reduce the new state by a weighted sum of the previous state and the new state (like a step size to prevent sharp changes).
        (If there is no previous state, it just returns the new state as the weighted sum is 100% of the new state)

        Motivation of implementing this function and not using the original one:
        Since there they save the previous state in the object itself, we had to clone the function to avoid changing the internal attributes and cause unexpected side effects.
        The original function is used with the coefficients of the FilterCoeff class, taken from "self.filter_coeff" in the JointState object (see call atrow 68 curobo/src/curobo/util/state_filter.py).
        (Similar to the motivation of implementing integrate_acc and not using the original one)
        Args:
            js_state_prev (JointState): the previous joint state (position, velocity, acceleration, jerk). (Jerk is not used though)
            js_state_new (JointState): the new joint state (position, velocity, acceleration, jerk). (Jerk is not used though)
            filter_coefficients (FilterCoeff): the filter coefficients. Replacing curobo/src/curobo/util/state_filter.py row 68 self.filter_coeff passed argument. For stable computations, 
            use the original coefficients, which are saved in js_state_prev.filter_coeff. See example in get_plan() of the mpc autonomous franka example.

        Returns:
            JointState: A re-weighted joint state (blending the previous and new states).
        """
        if js_state_prev is None:
            return js_state_new
        
        # re-weighting the new state to be closer to the previous state
        js_state_new.position[:] = filter_coefficients.position * js_state_new.position + (1.0 - filter_coefficients.position) * js_state_prev.position
        js_state_new.velocity[:] = filter_coefficients.velocity * js_state_new.velocity + (1.0 - filter_coefficients.velocity) * js_state_prev.velocity
        js_state_new.acceleration[:] = filter_coefficients.acceleration * js_state_new.acceleration + (1.0 - filter_coefficients.acceleration) * js_state_prev.acceleration
        js_state_new.jerk[:] = filter_coefficients.jerk * js_state_new.jerk + (1.0 - filter_coefficients.jerk) * js_state_prev.jerk
        return js_state_new
    
    @abstractmethod
    def init_col_predictor(self,obs_groups_nspheres:list[int]=[], manually_express_p_own_in_world_frame:bool=False) -> DynamicObsCollPredictor:
        pass
    
class FrankaMpc(AutonomousFranka):
    def __init__(self, 
                robot_cfg, 
                world:World, 
                usd_help:UsdHelper, 
                p_R=np.array([0.0,0.0,0.0]), 
                q_R=np.array([1,0,0,0]), 
                p_T_R=np.array([0.5, 0.0, 0.5]), 
                q_T_R=np.array([0, 1, 0, 0]), 
                target_color=np.array([0, 0.5, 0]), 
                target_size=0.05, 
                cost_live_plotting_cfg:dict={'live_plotting':False, 'save_plots':False},
                override_particle_file:str=None
                ):
        """
        Spawns a franka robot in the scene andd setting the target for the robot to follow.

        Args:
            world (_type_): _description_
            robot_name (_type_): _description_
            p_R (_type_): _description_
            q_R (_type_): _description_
           p_T (_type_): _description_
            q_T_R (_type_): _description_
            target_color (_type_): _description_
            target_size (_type_): _description_
            cost_live_plotting_cfgs (_type_): _description_
        """
        super().__init__(robot_cfg, world, p_R, q_R, p_T_R, q_T_R, target_color, target_size)
        # self.robot_cfg["kinematics"]["collision_sphere_buffer"] += 0.02  # Add safety margin (making collision spheres larger, you can see the difference if activeating the VISUALIZE_ROBOT_COL_SPHERES flag)
        self._spawn_robot_and_target(usd_help)
        self.articulation_controller = self.robot.get_articulation_controller()
        self._cmd_state_full = None
        self.override_particle_file = override_particle_file # New. settings in the file will overide the default settings in the default particle_mpc.yml file. For example, num of optimization steps per time step.
        self.cost_live_plotting_cfg = cost_live_plotting_cfg
        

        
        # self.cfg = load_yaml(self.override_particle_file)
        # if self.cfg is None:
        #     raise ValueError(f"Failed to load YAML file: {self.override_particle_file}")
            
        self.H = load_yaml(self.override_particle_file)["model"]["horizon"]
        self.num_particles = load_yaml(self.override_particle_file)["mppi"]["num_particles"]

        
  

    
    def init_solver(self,world_model, collision_cache, step_dt_traj_mpc,debug=False):
        """Initialize the MPC solver.

        Args:
            world_cfg (_type_): _description_
            collision_cache (_type_): _description_
            step_dt_traj_mpc (_type_): _description_
            dynamic_obs_coll_predictor (_type_): _description_
        """
        # if hasattr(self, "dynamic_obs_col_pred"):
        #     dynamic_obs_coll_predictor = self.dynamic_obs_col_pred
        # else:
        #     dynamic_obs_coll_predictor = None

        mpc_config = MpcSolverConfig.load_from_robot_config(
            self.robot_cfg, #  Robot configuration. Can be a path to a YAML file or a dictionary or an instance of RobotConfig https://curobo.org/_api/curobo.types.robot.html#curobo.types.robot.RobotConfig
            world_model, #  World configuration. Can be a path to a YAML file or a dictionary or an instance of WorldConfig. https://curobo.org/_api/curobo.geom.types.html#curobo.geom.types.WorldConfig
            use_cuda_graph=not debug, # Use CUDA graph for the optimization step. If you want to set breakpoints in the cost function, set this to False.
            use_cuda_graph_metrics=True, # Use CUDA graph for computing metrics.
            use_cuda_graph_full_step=False, #  Capture full step in MPC as a single CUDA graph. This is experimental and might not work reliably.
            self_collision_check=True, # Enable self-collision check during MPC optimization.
            collision_checker_type=CollisionCheckerType.MESH, # type of collision checker to use. See https://curobo.org/get_started/2c_world_collision.html#world-collision 
            collision_cache=collision_cache,
            use_mppi=True,  # Use Model Predictive Path Integral for optimization
            use_lbfgs=False, # Use L-BFGS solver for MPC. Highly experimental.
            use_es=False, # Use Evolution Strategies (ES) solver for MPC. Highly experimental.
            store_rollouts=True,  # Store trajectories for visualization
            step_dt=step_dt_traj_mpc,  # NOTE: Important! step_dt is the time step to use between each step in the trajectory. If None, the default time step from the configuration~(particle_mpc.yml or gradient_mpc.yml) is used. This dt should match the control frequency at which you are sending commands to the robot. This dt should also be greater than the compute time for a single step. For more info see https://curobo.org/_api/curobo.wrap.reacher.html
            # dynamic_obs_checker=dynamic_obs_coll_predictor, # New
            override_particle_file=self.override_particle_file, # New
            cost_live_plotting_cfg=self.cost_live_plotting_cfg # New
        )
        
        self.solver = MpcSolver(mpc_config)
        
        self._post_init_solver()
        retract_cfg = self.solver.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
        joint_names = self.solver.rollout_fn.joint_names
        state = self.solver.rollout_fn.compute_kinematics(JointState.from_position(retract_cfg, joint_names=joint_names))
        self.current_state = JointState.from_position(retract_cfg, joint_names=joint_names)
        retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
        
        # Set up goal pose (target position and orientation)
        goal_mpc = Goal(current_state=self.current_state, goal_state=JointState.from_position(retract_cfg, joint_names=joint_names), goal_pose=retract_pose,)

        # Initialize MPC solver with goal
        goal_buffer = self.solver.setup_solve_single(goal_mpc, 1)
        self.goal_buffer = goal_buffer
        self.solver.update_goal(self.goal_buffer)
        mpc_result = self.solver.step(self.current_state, max_attempts=2)



    def update_solver_target(self):
        # Express the target in the robot's base frame instead of the world frame (required for the solver)
        p_solverTarget_R, q_solverTarget_R = FrameUtils.world_to_F(self.p_R, self.q_R, self.p_solverTarget, self.q_solverTarget)
        X_solverTarget_R = Pose(position=self.tensor_args.to_device(p_solverTarget_R),quaternion=self.tensor_args.to_device(q_solverTarget_R))    
        
        # Update the goal buffer and the solver and in the cuda graph buffer
        self.goal_buffer.goal_pose.copy_(X_solverTarget_R) 
        self.solver.update_goal(self.goal_buffer)

    def _check_prerequisites_for_syncing_target_pose(self, real_target_position:np.ndarray, real_target_orientation:np.ndarray,sim_js:None) -> bool:
        has_target_pose_changed = self._check_target_pose_changed(real_target_position, real_target_orientation)
        return has_target_pose_changed

    def get_curobo_joint_state(self, sim_js=None,zero_vel=True) -> JointState:
        """See super().get_curobo_joint_state() for more details.
        """
        cu_js =  super().get_curobo_joint_state (sim_js, zero_vel)        
        cu_js = cu_js.get_ordered_joint_state(self.solver.rollout_fn.joint_names)
        return cu_js
    

    def update_current_state(self, cu_js):
        if self._cmd_state_full is None:
            self.current_state.copy_(cu_js)
        else:
            current_state_partial = self._cmd_state_full.get_ordered_joint_state(
                self.solver.rollout_fn.joint_names
            )
            self.current_state.copy_(current_state_partial)
            self.current_state.joint_names = current_state_partial.joint_names
        self.current_state.copy_(cu_js)


    def get_next_articulation_action(self, js_action):
        """Get articulated action from joint state action (supplied by MPC solver).
        Args:
            js_action  the next joints command (for current ts only)
        Returns:
            _type_: _description_
        """
        self._cmd_state_full = js_action
        idx_list = []
        common_js_names = []
        for x in self.get_dof_names():
            if x in self._cmd_state_full.joint_names:
                idx_list.append(self.robot.get_dof_index(x))
                common_js_names.append(x)
        self._cmd_state_full = self._cmd_state_full.get_ordered_joint_state(common_js_names)
        # art_action = ArticulationAction(self._cmd_state_full.position.cpu().numpy(),joint_indices=idx_list,) # old : with isaac 4
        art_action = ArticulationAction(self._cmd_state_full.position.view(-1).cpu().numpy(),joint_indices=idx_list,) # curobo 4.5 https://github.com/NVlabs/curobo/commit/0a50de1ba72db304195d59d9d0b1ed269696047f#diff-0932aeeae1a5a8305dc39b778c783b0b8eaf3b1296f87886e9d539a217afd207
        return art_action
    

    def apply_articulation_action(self, art_action: ArticulationAction,num_times:int=3):
        for _ in range(num_times):
            ans = self.articulation_controller.apply_action(art_action)
        return ans
    
    def get_trajopt_horizon(self):
        return self.H

    def set_new_target_for_solver(self, real_target_position:np.ndarray, real_target_orientation:np.ndarray,sim_js=None):
        solver_target_was_reset =  super().set_new_target_for_solver(real_target_position, real_target_orientation, sim_js)
        if solver_target_was_reset:
            # Express the target in the robot's base frame instead of the world frame (required for the solver)
            p_solverTarget_R, q_solverTarget_R = FrameUtils.world_to_F(self.p_R, self.q_R, self.p_solverTarget, self.q_solverTarget)
            X_solverTarget_R = Pose(position=self.tensor_args.to_device(p_solverTarget_R),quaternion=self.tensor_args.to_device(q_solverTarget_R))    
            
            # Update the goal buffer and the solver and in the cuda graph buffer
            self.goal_buffer.goal_pose.copy_(X_solverTarget_R) 
            self.solver.update_goal(self.goal_buffer)

        return solver_target_was_reset
    
    def get_policy_means(self):
        """Returning the mean values of the mpc policy (HX7 tensor).
        Each entry is an acceleration command for the corresponding joint.
        The accelerations are then applied at every time step, in order to compute the constant velocity of each joint during state and therfore its target position at the end of this command.
        This target position will be sent to the articulation controller.

        Returns:
            _type_: _description_
        """
        return self.solver.solver.optimizers[0].mean_action.squeeze(0)
    
    def get_col_pred(self):
        return self.get_cost_term("dynamic_obs").dynamic_obs_col_pred

    def get_mpc_solver(self) -> MpcSolver:
        """
        return the mpc solver object.
        """
        return self.solver
    
    def get_wrap_mpc(self) -> WrapMpc:
        """
        return the wrap mpc object of the solver.
        """
        return self.get_mpc_solver().solver
    
    def get_rollout_fn(self)-> ArmReacher:
        """
        return the rollout function of the solver.
        """
        return self.get_wrap_mpc().rollout_fn
    
    def get_safety_rollout(self) -> ArmReacher:
        """
        return the safety rollout function of the solver.
        """
        return self.get_wrap_mpc().safety_rollout
    
    def get_cost_term(self, cost_name:str):
        """
        Get a cost term from the  (which inherits from arm_base).
        Args:
            cost_name (str): the name of the cost term to get (e.g. "dynamic_obs", "pose_cost").
        Returns:
            the cost term.
        """

        # arm_reacher = self.get_arm_reacher()
        rollout_fn = self.get_rollout_fn()
        # first, search over original cost terms     
        for attr_name in rollout_fn.__dict__: # going over all attributes of the arm_reacher
            if hasattr(rollout_fn, cost_name):
                return getattr(rollout_fn, cost_name)

        # if not found, search over custom cost terms
        custom_cost_attrs = ['_custom_arm_base_costs', '_custom_arm_reacher_costs']
        for attr_name in custom_cost_attrs:
            if hasattr(rollout_fn, attr_name):
                custom_costs = getattr(rollout_fn, attr_name)
                if cost_name in custom_costs:
                    return custom_costs[cost_name]

        raise ValueError(f"Cost term '{cost_name}' not found in rollout_fn or its custom cost terms.")
    
    def get_plan(self, include_task_space:bool=True, n_steps:int=-1 ,valid_spheres_only = True):
        """
        Get the H steps plan from the solver. All positions are in the world frame.
        Args:
            in_task_space (bool): if True, return the plan in task space, otherwise return the plan in joint space.
            H (int, optional): the number of steps in the plan. If -1, return the entire plan. Defaults to -1.
        Returns:
            torch.Tensor: the H steps plan.
        """
        
        def map_nested_tensors(d: Dict[str, Union[dict, torch.Tensor]], fn: Callable[[torch.Tensor], torch.Tensor]) -> dict:
            """Recursively apply fn to all tensor leaves in a nested dict."""
            out = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    out[k] = map_nested_tensors(v, fn)
                elif isinstance(v, torch.Tensor):
                    out[k] = fn(v)
                else:
                    raise TypeError(f"Unsupported value type at key '{k}': {type(v)}")
            return out
       
        pi_mpc_means = self.get_policy_means() # (H x num of joints (7 in franka)) accelerations (each action is an acceleration vector)

        plan = {'joint_space':
                {     
                    'acc': pi_mpc_means, # these are mpc policy means
                    'vel': torch.zeros(pi_mpc_means.shape), # direct result of acc
                    'pos': torch.zeros(pi_mpc_means.shape) # direct result of vel
                }
            }
        
        _wrap_mpc = self.solver.solver
        _arm_reacher = _wrap_mpc.safety_rollout
        _kinematics_model = _arm_reacher.dynamics_model
        _state_filter = _kinematics_model.state_filter
        filter_coefficients_solver = _state_filter.filter_coeff # READ ONLY: the original coefficients from the mpc planner (used only to read from. No risk that will be changed unexpectdely)
        control_dt_solver = _state_filter.dt # READ ONLY: the delta between steps in the trajectory, as set in solver. This is what the mpc assumes time delta between steps in horizon is.s

        
        # translate the plan from joint accelerations only to joint velocities and positions 
        # js_state = self.get_curobo_joint_state() # current joint state (including pos, vel, acc)
        apply_js_filter = True # True: Reduce the step size from prev state to new state from 1 to something smaller (depends on the filter coefficients)
        custom_filter = False # True: Use a custom filter coefficients to play with the filter weights
        if apply_js_filter:
            if custom_filter:
                filter_coeff = FilterCoeff(0.01, 0.01, 0.0, 0.0) # custom one to play with the filter weights
            else:
                filter_coeff = filter_coefficients_solver # the one used by the mpc planner
        js_state_sim = self.get_sim_joint_state()
        n_dofs = pi_mpc_means.shape[1]
        js_state = JointState(torch.from_numpy(js_state_sim.positions[:n_dofs]), torch.from_numpy(js_state_sim.velocities[:n_dofs]), torch.zeros(n_dofs), self.get_dof_names(),torch.zeros(n_dofs), self.tensor_args)
        js_state_prev = None
        # js_state.jerk = torch.zeros_like(js_state.velocity) # we don't really need this for computations, but it has to be initiated to avoid exceptions in the filtering
        
        for h, action in enumerate(pi_mpc_means):
            if apply_js_filter:
                js_state = self.filter_joint_state(js_state_prev, js_state, filter_coeff) # 
            next_js_state = self.integrate_acc(action, js_state, control_dt_solver) # this will be the new joint state after applying the acceleration the mpc policy commands for this step for dt_planning seconds
            plan['joint_space']['vel'][h] = next_js_state.velocity.squeeze()
            plan['joint_space']['pos'][h] = next_js_state.position.squeeze()
            js_state = next_js_state # JointState(next_js_state.position, next_js_state.velocity, next_js_state.acceleration,js_state.joint_names, js_state.jerk)
            js_state_prev = js_state
        
        if include_task_space: # get plan in task space (robot spheres)            
            # compute forward kinematics
            p_eeplan, q_eeplan, _, _, p_linksplan, q_linksplan, prad_spheresPlan = self.crm.forward(self.tensor_args.to_device(plan['joint_space']['pos'])) # https://curobo.org/_api/curobo.cuda_robot_model.cuda_robot_model.html#curobo.cuda_robot_model.cuda_robot_model.CudaRobotModelConfig
            task_space_plan = {'ee': {'p': p_eeplan, 'q': q_eeplan}, 'links': {'p': p_linksplan, 'q': q_linksplan}, 'spheres': {'p': prad_spheresPlan[:,:,:3], 'r': prad_spheresPlan[:,:,3]}}
            plan['task_space'] = task_space_plan
            
            # remove spheres that are not valid (i.e. negative radius)
            if valid_spheres_only: # removes the (4) extra spheres reserved for simulating a picked object as part of the robot after picked
                plan['task_space']['spheres']['p'] = plan['task_space']['spheres']['p'][:, :self.n_coll_spheres_valid]
                plan['task_space']['spheres']['r'] = plan['task_space']['spheres']['r'][:, :self.n_coll_spheres_valid]
            

            # express in world frame:
            for key in plan['task_space'].keys():
                
                # if isinstance(plan['task_space'][key], dict) and 'p' in plan['task_space'][key].keys() and 'q' not in plan['task_space'][key].keys():
                #     pKey = plan['task_space'][key]['p']
                #     pKey = pKey.cpu() 
                #     pKey[...,:3] = pKey[...,:3] + self.p_R # # offset by robot origin to express in world frame (only position, radius is not affected)
                #     plan['task_space'][key]['p'] = pKey
            
                # express in world frame:
                # elif isinstance(plan['task_space'][key], dict) and 'p' in plan['task_space'][key].keys() and 'q' in plan['task_space'][key].keys():
                #     self_transform = [*list(self.p_R), *list(self.q_R)]
                #     pKey = plan['task_space'][key]['p']
                #     qKey = plan['task_space'][key]['q']
                #     for i, p in enumerate(pKey[...,:]):
                #         X_key_i_Robot = pKey[...,:][i].flatten().cpu().tolist() +  qKey[...,:][i].flatten().cpu().tolist()
                #         X_key_i_world = torch.tensor(transform_pose_between_frames(X_key_i_Robot, self_transform)) 
                #         pKey[...,:][i] = X_key_i_world[:3]
                #         qKey[...,:][i] = X_key_i_world[3:]
                #     plan['task_space'][key]['p'] = pKey
                #     plan['task_space'][key]['q'] = qKey
                
                if isinstance(plan['task_space'][key], dict) and 'p' in plan['task_space'][key].keys():
                    
                    self_transform = [*list(self.p_R), *list(self.q_R)]
                    pKey = plan['task_space'][key]['p']
                    if 'q' in plan['task_space'][key].keys():
                        qKey = plan['task_space'][key]['q']
                    else:
                        qKey = torch.empty(pKey.shape[:-1] + torch.Size([4]), device=pKey.device)
                        qKey[...,:] = torch.tensor([1,0,0,0],device=pKey.device, dtype=pKey.dtype)  # [1,0,0,0] is the identity quaternion
                    X_world = transform_poses_batched(torch.cat([pKey, qKey], dim=-1), self_transform)
                    pKey = X_world[...,:3]
                    qKey = X_world[...,3:]
                    plan['task_space'][key]['p'] = pKey
                    plan['task_space'][key]['q'] = qKey

                    # transform_poses_batched
                    # import itertools
                    # index_ranges = [range(s) for s in pKey.shape[:-1]]
    
                    # for nd_idx in itertools.product(*index_ranges):
                        
                    #     if 'q' in plan['task_space'][key].keys():
                    #         qKey = plan['task_space'][key]['q']
                    #         qKey_i = qKey[nd_idx].flatten().cpu().tolist()
                    #     else:
                    #         qKey_i = [1,0,0,0]

                    #     X_key_i_Robot = pKey[nd_idx].flatten().cpu().tolist() +  qKey_i
                    #     X_key_i_world = torch.tensor(transform_pose_between_frames(X_key_i_Robot, self_transform)) 
                    #     pKey[nd_idx] = X_key_i_world[:3]
                    #     if 'q' in plan['task_space'][key].keys():
                    #         qKey[nd_idx] = X_key_i_world[3:]
                    
                    # plan['task_space'][key]['p'] = pKey
                    # if 'q' in plan['task_space'][key].keys():
                    #     plan['task_space'][key]['q'] = qKey
                
            
            # take only the first n_steps
            if not n_steps == -1:        
                plan = map_nested_tensors(plan, lambda x: x[:n_steps])
    

        
        return plan 
        
       
    # def init_col_predictor(self,obs_groups_nspheres:list[int]=[], manually_express_p_own_in_world_frame:bool=False) -> DynamicObsCollPredictor:
    #     cost_weight = self.cfg['cost']['dynamic_obs']['weight']
    #     n_particles = self.num_particles 
    #     H = self.H # self.trajopt_tsteps - 1 if self.dilation_factor == 1.0 else self.trajopt_tsteps
    #     self.dynamic_obs_col_pred = DynamicObsCollPredictor(self.tensor_args,
    #                                                         None,
    #                                                         H,
    #                                                         n_particles,
    #                                                         self.n_coll_spheres_valid,
    #                                                         sum(obs_groups_nspheres),
    #                                                         cost_weight,
    #                                                         obs_groups_nspheres,
    #                                                         manually_express_p_own_in_world_frame,
    #                                                         torch.from_numpy(self.p_R))
    #     return self.dynamic_obs_col_pred
    
class FrankaCumotion(AutonomousFranka):
    def __init__(self, 
                # robot basic parameters
                robot_cfg, 
                world:World, 
                usd_help:UsdHelper, 
                p_R=np.array([0.0,0.0,0.0]), 
                q_R=np.array([1,0,0,0]), 
                p_T=np.array([0.5, 0.0, 0.5]), 
                q_T=np.array([0, 1, 0, 0]), 
                target_color=np.array([0, 0.5, 0]), 
                target_size=0.05, 
                # general parameters for planner
                reactive=False, 
                
                # Solver configuration parameters (parameters for MotionGenConfig)
                num_ik_seeds=32,
                trajopt_tsteps=32,
                trajopt_dt=0.15, 
                optimize_dt=True, 
                num_trajopt_seeds=12,
                num_graph_seeds=12,
                enable_graph_planner=False, 
                interpolation_dt=0.05,
                # Post trajectory optimization parameters
                dilation_factor=0.5,
                evaluate_interpolated_trajectory=True
                ):
        """
        Spawns a franka robot in the scene andd setting the target for the robot to follow.
        
        Args: (FOR MORE INFO SEE: https://curobo.org/_api/curobo.wrap.reacher.trajopt.html)
            world (_type_): World of the simulation
            robot_name (str): 
            p_R: initial position of the robot
            q_R: initial orientation of the robot
            p_T: initial position of the target
            q_T: initial orientation of the target
            target_color (np.ndarray, optional): color of the target. Defaults to np.array([0, 0.5, 0]).
            target_size (float, optional): size of the target. Defaults to 0.05.
            reactive (bool, optional): _description_. Defaults to False.
            
            # solver parameters:
            trajopt_tsteps (int, optional):trajopt_tsteps – Number of waypoints (time steps, states) to use for (in each of the trajectories during) trajectory optimization. Default of 32 is found to be a good number for most cases. # Includes first and last states.
            trajopt_dt (float, optional): trajopt_dt – Time step in seconds to use for trajectory optimization. A good value to start with is 0.15 seconds. This value is used to compute velocity, acceleration, and jerk values for waypoints through finite difference. Defaults to 0.15.
            optimize_dt (bool, optional): # Optimize dt during trajectory optimization. Default of True is recommended to find time-optimal trajectories. Setting this to False will use the provided trajopt_dt for trajectory optimization. Setting to False is required when optimizing from a non-static start state.. Defaults to True.
            num_trajopt_seeds (int, optional): _description_. Defaults to 12.
            num_graph_seeds (int, optional): _description_. Defaults to 12.
            num_ik_seeds (int, optional): num_ik_seeds Number of seeds to use for solving inverse kinematics. Default of 32 is found to be a good number for most cases. In sparse environments, a lower number of 16 can also be used. Note: in paper they advise to start with 500 for tuning. See in https://curobo.org/_api/curobo.wrap.reacher.motion_gen.html#curobo.wrap.reacher.motion_gen.MotionGenConfig
            interpolation_dt (float, optional): Time step in seconds to use for generating interpolated trajectory from optimized trajectory. Change this if you want to generate a trajectory with a fixed timestep between waypoints. Defaults to 0.05.
            enable_graph_planner (bool, optional): _description_. Defaults to False.
            dilation_factor(float, optional): Slowing down the final plan by this factor. Probably for debugging. Belongs to the "re-timing" process after trajectory optimization. See comment in init_plan_config(). See https://curobo.org/_api/curobo.wrap.reacher.motion_gen.html#curobo.wrap.reacher.motion_gen.MotionGenResult.retime_trajectory
            evaluate_interpolated_trajectory(bool, optional): I set it to False for the dynamic obstacles. Original docs: evaluate_interpolated_trajectory – Evaluate interpolated trajectory after optimization. Default of True is recommended to ensure the optimized trajectory is not passing through very thin obstacles.
        """
        super().__init__(robot_cfg, world, p_R, q_R, p_T, q_T, target_color, target_size)

        self.solver = None # motion generator
        self.past_cmd:JointState = None # last commanded joint state
        self.reactive = reactive # can start planning without waiting for the robot to be static (experimental and not recommended)
        self.num_targets = 0 # the number of the targets which are defined by curobo (after being static and ready to plan to) and have a successfull a plan for.
        self.max_attempts = 4 if not self.reactive else 1
        self.enable_finetune_trajopt = True if not self.reactive else False
        self.trim_steps = None if not self.reactive else [1, None]
        self.pose_metric = None
        self.constrain_grasp_approach = False
        self.reach_partial_pose = None
        self.hold_partial_pose = None
        self.cmd_plan = None # the global plan (commanded trajectory) while the robot is executing a plan, and None when its idle
        self.cmd_idx = 0 # pointer to the current timestep in the command plan (don't change this)

        # IK RELATED PARAMETERS:
        # todo: add ik parameters here
        self.num_ik_seeds = num_ik_seeds
        # GRAPH PLANNER RELATED PARAMETERS: (for graph planner (optional)- generating seeds for trajectory optimization)
        self.enable_graph_planner = enable_graph_planner
        self.num_graph_seeds = num_graph_seeds
        
        # TRAJECTORY OPTIMIZATION RELATED PARAMETERS:
        # For full docs and more variables see: https://curobo.org/_api/curobo.wrap.reacher.trajopt.html
        self.num_trajopt_seeds = num_trajopt_seeds # Num of seeds (inputs, initial trajectories) for trajectory optimization.
        self.trajopt_tsteps = trajopt_tsteps if not self.reactive else 40 # trajopt_tsteps – Number of waypoints to use for trajectory optimization. Default of 32 is found to be a good number for most cases.
        self.trajopt_dt = trajopt_dt if not self.reactive else 0.04 # time delta between steps in the trajectory during trajectory optimization.
        self.optimize_dt = optimize_dt if not self.reactive else False
        self.interpolation_dt = interpolation_dt if not self.reactive else self.trajopt_dt
        self.evaluate_interpolated_trajectory = evaluate_interpolated_trajectory
        # AFTER TRAJECTORY OPTIMIZATION RELATED PARAMETERS:
        self.dilation_factor = dilation_factor if not self.reactive else 1.0 


        # ---- Spawn the robot and target ----
        self._spawn_robot_and_target(usd_help)
        self.articulation_controller = self.robot.get_articulation_controller()
        
    def apply_articulation_action(self, art_action: ArticulationAction):
        self.cmd_idx += 1
        if self.cmd_idx >= len(self.cmd_plan.position): # NOTE: all cmd_plans (global plans) are at the same length from my observations (currently 61), no matter how many time steps (step_indexes) take to complete the plan.
            self.cmd_idx = 0
            self.cmd_plan = None
            self.past_cmd = None
        return self.articulation_controller.apply_action(art_action)

    
    
    def init_solver(self, world_model, collision_cache, debug=False):
        """Initialize the motion generator (cumotion global planner).

        Args:
            world_cfg (_type_): _description_
            collision_cache (_type_): _description_
            tensor_args (_type_): _description_
        """
    
        if hasattr(self, 'dynamic_obs_col_pred'):
            dynamic_obs_coll_predictor = self.dynamic_obs_col_pred
        else:
            dynamic_obs_coll_predictor = None
        # See very good explainations for all the paramerts here: https://curobo.org/_api/curobo.wrap.reacher.motion_gen.html#curobo.wrap.reacher.motion_gen.MotionGenConfig
        motion_gen_config = MotionGenConfig.load_from_robot_config( # solver config
            self.robot_cfg, # robot_cfg – Robot configuration to use for motion generation. This can be a path to a yaml file, a dictionary, or an instance of RobotConfig. See Supported Robots for a list of available robots. You can also create a a configuration file for your robot using Configuring a New Robot.
            world_model, # world_model – World configuration to use for motion generation. This can be a path to a yaml file, a dictionary, or an instance of WorldConfig. See Collision World Representation for more details.
            self.tensor_args, # tensor_args - Numerical precision and compute device to use for motion generation
            collision_checker_type=CollisionCheckerType.MESH, # collision_checker_type – Type of collision checker to use for motion generation. Default of CollisionCheckerType.MESH supports world represented by Cuboids and Meshes. See Collision World Representation for more details.
            num_ik_seeds=self.num_ik_seeds, # num_ik_seeds – Number of seeds to use for solving inverse kinematics. Default of 32 is found to be a good number for most cases. In sparse environments, a lower number of 16 can also be used. Note: in paper they advise to start with 500 for tuning.   
            num_trajopt_seeds=self.num_trajopt_seeds, # num_trajopt_seeds – Number of seeds to use for trajectory optimization per problem query. Default of 4 is found to be a good number for most cases. Increasing this will increase memory usage.
            num_graph_seeds=self.num_graph_seeds, # num_graph_seeds – Number of seeds to use for graph planner per problem query. When graph planning is used to generate seeds for trajectory optimization, graph planner will attempt to find collision-free paths from the start state to the many inverse kinematics solutions.
            interpolation_dt=self.interpolation_dt, # interpolation_dt – Time step in seconds to use for generating interpolated trajectory from optimized trajectory. Change this if you want to generate a trajectory with a fixed timestep between waypoints.
            collision_cache=collision_cache, # collision_cache – Cache of obstacles to create to load obstacles between planning calls. An example: {"obb": 10, "mesh": 10}, to create a cache of 10 cuboids and 10 meshes.
            optimize_dt=self.optimize_dt, # optimize_dt – Optimize dt during trajectory optimization. Default of True is recommended to find time-optimal trajectories. Setting this to False will use the provided trajopt_dt for trajectory optimization. Setting to False is required when optimizing from a non-static start state.
            trajopt_dt=self.trajopt_dt, # trajopt_dt – Time step in seconds to use for trajectory optimization. A good value to start with is 0.15 seconds. This value is used to compute velocity, acceleration, and jerk values for waypoints through finite difference.
            trajopt_tsteps=self.trajopt_tsteps, # trajopt_tsteps – Number of waypoints to use for trajectory optimization. Default of 32 is found to be a good number for most cases.
            trim_steps=self.trim_steps, # trim_steps – Trim waypoints from optimized trajectory. The optimized trajectory will contain the start state at index 0 and have the last two waypoints be the same as T-2 as trajectory optimization implicitly optimizes for zero acceleration and velocity at the last waypoint. An example: [1,-2] will trim the first waypoint and last 3 waypoints from the optimized trajectory.
            use_cuda_graph=not debug, # Record compute ops as cuda graphs and replay recorded graphs where implemented. This can speed up execution by upto 10x. Default of True is recommended. Enabling this will prevent changing solve type or batch size after the first call to the solver.
            evaluate_interpolated_trajectory=self.evaluate_interpolated_trajectory, # evaluate_interpolated_trajectory – Evaluate interpolated trajectory after optimization. Default of True is recommended to ensure the optimized trajectory is not passing through very thin obstacles.
            dynamic_obs_checker=dynamic_obs_coll_predictor, # New!
        )
        self.solver = MotionGen(motion_gen_config)
        if not self.reactive:
            print("warming up...")
            self.solver.warmup(enable_graph=not debug, warmup_js_trajopt=False)
        
        self.plan_config = self._init_plan_config()
        print("Curobo is Ready")

    def _init_plan_config(self):
        """Initialize the plan config for the motion generator.
        See all the documentation here: https://curobo.org/_api/curobo.wrap.reacher.motion_gen.html#curobo.wrap.reacher.motion_gen.MotionGenPlanConfig
        """
        return MotionGenPlanConfig(
            enable_graph=self.enable_graph_planner, # Use graph planner to generate collision-free seed for trajectory optimization.
            enable_graph_attempt=2, # Number of failed attempts at which to fallback to a graph planner for obtaining trajectory seeds.
            max_attempts=self.max_attempts, # Maximum number of attempts allowed to solve the motion generation problem.
            enable_finetune_trajopt=self.enable_finetune_trajopt, # Run finetuning trajectory optimization after running 100 iterations of trajectory optimization. This will provide shorter and smoother trajectories. When MotionGenConfig.optimize_dt is True, this flag will also scale the trajectory optimization by a new dt. Leave this to True for most cases. If you are not interested in finding time-optimal solutions and only want to use motion generation as a feasibility check, set this to False. Note that when set to False, the resulting trajectory is only guaranteed to be collision-free and within joint limits. When False, it's not guaranteed to be smooth and might not execute on a real robot.
            time_dilation_factor=self.dilation_factor, # Slow down optimized trajectory by re-timing with a dilation factor. This is useful to execute trajectories at a slower speed for debugging. Use this to generate slower trajectories instead of reducing MotionGenConfig.velocity_scale or MotionGenConfig.acceleration_scale as those parameters will require re-tuning of the cost terms while MotionGenPlanConfig.time_dilation_factor will only post-process the trajectory.
        )
    
    def _check_prerequisites_for_syncing_target_pose(self, real_target_position, real_target_orientation,sim_js) -> bool:
        robot_prerequisites = self._check_robot_static(sim_js) or self.reactive # robot is allowed to reset_command_plan global plan if stopped (in the non-reactive mode) or anytime in the reactive mode
        target_prerequisites = self._check_target_pose_changed(real_target_position, real_target_orientation) and self._check_target_pose_static(real_target_position, real_target_orientation)
        return robot_prerequisites and target_prerequisites
        
    def reset_command_plan(self, cu_js):

        """
        Replanning a new global plan and updating the command plan.

        To better understand the code, see:
        MotionGenResult: https://curobo.org/_api/curobo.wrap.reacher.motion_gen.html#curobo.wrap.reacher.motion_gen.MotionGenResult.optimized_plan
            interpolated_plan: interpolated solution, useful for visualization.

        """
        print("reset_command_planning a new global plan - goal pose has changed!")
            
        # Set EE teleop goals, use cube for simple non-vr init:
        # ee_translation_goal = self.p_solverTarget # cube position is the updated target pose (which has moved) 
        # ee_orientation_teleop_goal = self.q_solverTarget # cube orientation is the updated target orientation (which has moved)

        # compute curobo solution:
        p_solverTarget_R, q_solverTarget_R = FrameUtils.world_to_F(self.p_R, self.q_R, self.p_solverTarget, self.q_solverTarget) # TODO
        ik_goal = Pose(position=self.tensor_args.to_device(p_solverTarget_R), quaternion=self.tensor_args.to_device(q_solverTarget_R))
        self.plan_config.pose_cost_metric = self.pose_metric
        start_state = cu_js.unsqueeze(0) # cu_js is the current joint state of the robot
        goal_pose = ik_goal # ik_goal is the updated target pose (which has moved)
        
        result: MotionGenResult = self.solver.plan_single(start_state, goal_pose, self.plan_config) # https://curobo.org/_api/curobo.wrap.reacher.motion_gen.html#curobo.wrap.reacher.motion_gen.MotionGen.plan_single:~:text=GraphResult-,plan_single,-( , https://curobo.org/_api/curobo.wrap.reacher.motion_gen.html#curobo.wrap.reacher.motion_gen.MotionGenResult:~:text=class-,MotionGenResult,-(
        self.last_motion_gen_result = result 
        succ = result.success.item()  # an attribute of this returned object that signifies whether a trajectory was successfully generated. success tensor with index referring to the batch index.
        
        if self.num_targets == 1: # it's 1 only immediately after the first time it found a successfull plan for the FIRST time (first target).
            if self.constrain_grasp_approach:
                # cuRobo also can enable constrained motions for part of a trajectory.
                # This is useful in pick and place tasks where traditionally the robot goes to an offset pose (pre-grasp pose) and then moves 
                # to the grasp pose in a linear motion along 1 axis (e.g., z axis) while also constraining it's orientation. We can formulate this two step process as a single trajectory optimization problem, with orientation and linear motion costs activated for the second portion of the timesteps. 
                # https://curobo.org/advanced_examples/3_constrained_planning.html#:~:text=Grasp%20Approach%20Vector,behavior%20as%20below.
                # Enables moving to a pregrasp and then locked orientation movement to final grasp.
                # Since this is added as a cost, the trajectory will not reach the exact offset, instead it will try to take a blended path to the final grasp without stopping at the offset.
                # https://curobo.org/_api/curobo.rollout.cost.pose_cost.html#curobo.rollout.cost.pose_cost.PoseCostMetric.create_grasp_approach_metric
                self.pose_metric = PoseCostMetric.create_grasp_approach_metric() # 
            if self.reach_partial_pose is not None:
                # This is probably a way to update the cost metric for reaching a partial pose reaching (not sure how, no documentation).
                reach_vec = self.solver.tensor_args.to_device(args.reach_partial_pose)
                self.pose_metric = PoseCostMetric(
                    reach_partial_pose=True, reach_vec_weight=reach_vec
                )
            if self.hold_partial_pose is not None:
                # This is probably a way to update the cost metric for reaching a partial pose reaching (not sure how, no documentation).
                hold_vec = self.solver.tensor_args.to_device(args.hold_partial_pose)
                self.pose_metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=hold_vec)
        
        if succ: 
            print(f"target counter - targets with a reachible plan = {self.num_targets}") 
            self.num_targets += 1
            cmd_plan = result.get_interpolated_plan() # Get interpolated trajectory from the result. https://curobo.org/_api/curobo.wrap.reacher.motion_gen.html#curobo.wrap.reacher.motion_gen.MotionGenResult.get_interpolated_plan https://curobo.org/_api/curobo.wrap.reacher.motion_gen.html#curobo.wrap.reacher.motion_gen.MotionGenResult.interpolation_dt
            cmd_plan = self.solver.get_full_js(cmd_plan) # get the full joint state from the interpolated plan
            # get only joint names that are in both:
            self.idx_list = []
            self.common_js_names = []
            for x in self.get_dof_names():
                if x in cmd_plan.joint_names:
                    self.idx_list.append(self.robot.get_dof_index(x))
                    self.common_js_names.append(x)

            cmd_plan = cmd_plan.get_ordered_joint_state(self.common_js_names)
            self.cmd_plan = cmd_plan # global plan
            self.cmd_idx = 0 # commands executed from the global plan (counter)
        else:
            carb.log_warn("Plan did not converge to a solution: " + str(result.status))
            cmd_plan = None

    def get_plan(self, include_task_space:bool=True,to_go_only=True, plan_stage='final'):
        plan = self.get_current_plan_as_tensor(to_go_only, plan_stage)            
        if plan is None or not len(plan[0]):
            return 
        p_jsplan, vel_jsplan = plan[0], plan[1] # from current time step t to t+H-1 inclusive
        # Compute FK on plan: all poses and orientations are expressed in robot2 frame (R2). Get poses of robot2's end-effector and links in robot2 frame (R2) and spheres (obstacles) in robot2 frame (R2).
        p_eeplan, q_eeplan, _, _, p_linksplan, q_linksplan, prad_spheresPlan = self.crm.forward(p_jsplan, vel_jsplan) # https://curobo.org/_api/curobo.cuda_robot_model.cuda_robot_model.html#curobo.cuda_robot_model.cuda_robot_model.CudaRobotModelConfig
        
        valid_only = True # remove spheres that are not valid (i.e. negative radius)
        if valid_only:
            prad_spheresPlan = prad_spheresPlan[:, self.valid_coll_spheres_idx]
        # convert to world frame (W):
        p_rad_spheresfullplan = prad_spheresPlan[:,:,:].cpu() # copy of the spheres in robot2 frame (R2)
        p_rad_spheresfullplan[:,:,:3] = p_rad_spheresfullplan[:,:,:3] + self.p_R # # offset of robot2 origin in world frame (only position, radius is not affected)
        p_spheresfullplan = p_rad_spheresfullplan[:,:,:3]
        rad_spheres = p_rad_spheresfullplan[0,:,3] 

        return p_spheresfullplan, rad_spheres

    def get_curobo_joint_state(self, sim_js=None,zero_vel=False) -> JointState:
        """
        See super().get_curobo_joint_state() for more details.

        Args:
            sim_js (_type_): _description_

        Returns:
            JointState: _description_
        """
        cu_js = super().get_curobo_joint_state(sim_js, zero_vel)

        if not self.reactive: # In reactive mode, we will not wait for a complete stopping of the robot before navigating to a new goal pose (if goal pose has changed). In the default mode on the other hand, we will wait for the robot to stop.
                cu_js.velocity *= 0.0
                cu_js.acceleration *= 0.0

        if self.reactive and self.past_cmd is not None:
            cu_js.position[:] = self.past_cmd.position
            cu_js.velocity[:] = self.past_cmd.velocity
            cu_js.acceleration[:] = self.past_cmd.acceleration

        cu_js = cu_js.get_ordered_joint_state(self.solver.kinematics.joint_names) 


        return cu_js    
    
    def get_trajopt_horizon(self):
        return self.trajopt_tsteps

    def get_current_plan_as_tensor(self, to_go_only=True, plan_stage='final') -> torch.Tensor:
        """
        Returns the joint states at at the start of each command in the current plan and the joint velocity commands in the current plan.

        Args:
            to_go_only (bool, optional):If true, only the commands left to execute in the current plan are returned (else, all commands, including the ones already executed, are returned). Defaults to True.

        Returns:
            _type_: _description_
        """
        if self.cmd_plan is None: # robot is not following any plan at the moment
            return 
        if plan_stage == 'optimized': # Optimized plan: this is the plan right after optimization part in the motion generator. Length should be as trajopt_tsteps (todo: verify that).
            plan = self.last_motion_gen_result.optimized_plan # fixed size (trajopt_tsteps x dof_num)
            if plan is None:
                return 
        elif plan_stage == 'final': # Final plan: this is the plan which will be sent to controller (optimized, interpoladed, and time-dilated if applied)
            plan = self.cmd_plan # variable size (generated from the optimized plan)
        
        else:
            raise ValueError(f"Invalid plan stage: {plan_stage}")
        
        n_total = len(plan) # total num of commands (actions) to apply to controller in current command (global) plan
        n_applied = self.cmd_idx # number of applied actions from total command plan
        
        # n_to_go = n_total - n_applied # num of commands left to execute 
        start_idx = 0 if not to_go_only else n_applied
        start_q = plan.position[start_idx:] # at index i: joint positions just before applying the ith command
        vel_cmd = plan.velocity[start_idx:] # at index i: joint velocities to apply to the ith command
        return torch.stack([start_q, vel_cmd]) # shape: (2, len(plan), dof_num)
    
    def get_plan_dt(self, dt_type='optimized'):
        """
        Returns the time step of the plan.
        dt_type:
        optimized_dt – Time between steps in the optimized trajectory.
        interpolation_dt – Time between steps in the interpolated trajectory. If None, MotionGenResult.interpolation_dt is used.

        """
        if dt_type == 'optimized':
            return self.last_motion_gen_result.optimized_dt
        elif dt_type == 'interpolation':
            return self.last_motion_gen_result.interpolation_dt 

    def get_next_articulation_action(self,idx_list):
        next_cmd = self.cmd_plan[self.cmd_idx] # get the next joint command from the plan
        self.past_cmd = next_cmd.clone() # save the past command for future use
        next_cmd_joint_pos = next_cmd.position.cpu().numpy() # Joint configuration of the next command.
        next_cmd_joint_vel = next_cmd.velocity.cpu().numpy() # Joint velocities of the next command.
        art_action = ArticulationAction(next_cmd_joint_pos, next_cmd_joint_vel,joint_indices=idx_list,) # controller command
        return art_action


    def integrate_acc(self,qdd_des: T_DOF,cmd_joint_state: Optional[JointState] = None,
        dt: Optional[float] = None,
    ):
        dt = self.dt if dt is None else dt
        if cmd_joint_state is not None:
            if self.cmd_joint_state is None:
                self.cmd_joint_state = cmd_joint_state.clone()
            else:
                self.cmd_joint_state.copy_(cmd_joint_state)
        self.cmd_joint_state.acceleration[:] = qdd_des
        self.cmd_joint_state.velocity[:] = self.cmd_joint_state.velocity + qdd_des * dt
        self.cmd_joint_state.position[:] = (
            self.cmd_joint_state.position + self.cmd_joint_state.velocity * dt
        )
        # TODO: for now just have zero jerl:
        if self.cmd_joint_state.jerk is None:
            self.cmd_joint_state.jerk = qdd_des * 0.0
        else:
            self.cmd_joint_state.jerk[:] = qdd_des * 0.0
        return self.cmd_joint_state.clone()
        
    # def init_col_predictor(self,obs_groups_nspheres:list[int]=[], cost_weight:float=100, manually_express_p_own_in_world_frame:bool=True) -> DynamicObsCollPredictor:
    #     n_particles = self.num_trajopt_seeds 
    #     H = self.trajopt_tsteps # self.trajopt_tsteps - 1 if self.dilation_factor == 1.0 else self.trajopt_tsteps
    #     self.dynamic_obs_col_pred = DynamicObsCollPredictor(self.tensor_args,
    #                                                         None,
    #                                                         H,
    #                                                         n_particles,
    #                                                         self.n_coll_spheres_valid,
    #                                                         sum(obs_groups_nspheres),
    #                                                         cost_weight,
    #                                                         obs_groups_nspheres,
    #                                                         manually_express_p_own_in_world_frame,
    #                                                         torch.from_numpy(self.p_R))
    #     return self.dynamic_obs_col_pred
    
    