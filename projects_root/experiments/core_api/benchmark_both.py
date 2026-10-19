from __future__ import annotations
import argparse
import random
from typing import Optional, Union
import numpy as np
import os
import multiprocessing as mp
from curobo.util_file import load_yaml
import csv

stop_simulation = False

def signal_handler(signum, _frame):
    """
    signal handler for stopping the simulation when running in __main__
    """
    print(f"\nReceived {signum} – stopping…")
    global stop_simulation
    stop_simulation = True


def _get_scipy_rotation():
    from scipy.spatial.transform import Rotation as Rotation
    return Rotation

class PoseUtils:
        def __init__(self, seed:Optional[int]=None):
            self.seed = seed
            if seed is not None:
                self._local_rng = random.Random(seed)
            else:
                self._local_rng = random.Random()

        def sample_pos_in_box(self, box_center:list[float],box_dim:float,)->list[float]:     
            """
            sample a position inside an (imaginary) box.
            Args:
                box_center (list[float]): center position of the box (xyz in world frame)
                box_dim (float): dimension of the volume (side length for box)
            Returns:
                list[float]: sampled position (xyz in world frame)
            """
            
            ans = []
            for dim in range(3):
                delta = box_dim / 2 # diam/2 (radius) for sphere, (side length)/2 for box
                dim_translation = (self._local_rng.uniform(-delta, delta))
                ans.append(box_center[dim] + dim_translation)
            return ans
        
        def sample_pos_in_sphere(self, sphere_center:list[float], radius:float)->list[float]:     
            """
            sample a position inside an (imaginary) sphere.
            Args:
                volume_center_pos (list[float]): center position of the sphere (xyz in world frame)
                volume_dim (float): dimension of the sphere (diameter)

            Returns:
                list[float]: sampled position (xyz in world frame)
            """
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
        def rotate_quat(q_in:Union[np.ndarray, list[float]], euler_deg:Union[np.ndarray, list[float]],q_in_wxyz:bool=True, q_out_wxyz:bool=True)->list[float]:
            """
            given a quaternion and euler angles, return the new quaternion rotated by the euler angles.
            Args:
                q_in (Union[np.ndarray, list[float]]): quaternion (wxyz)
                euler_deg (Union[np.ndarray, list[float]]): euler angles (xyz)
                q_in_wxyz (bool): if True, q_in is in wxyz format, if False, q_in is in xyzw format
                q_out_wxyz (bool): if True, q_out is in wxyz format, if False, q_out is in xyzw format
            Returns:
                list[float]: new quaternion (wxyz)
            Example:
                q_in = [1,0,0,0] # (identity)
                euler_deg = [0,0,0] # no rotation
                q_out = [1,0,0,0] # (identity)
                q_in = [1,0,0,0] # (x-axis)
                euler_deg = [0,0,90] # 90 degrees around z-axis
                q_out = [0,0,0,1] # (z-axis) # TODO VERIFY THIS
            """
            # q_in: scalar-first (wxyz)
            # euler_deg: rotation to apply, in degrees (xyz)

            # Convert input quaternion to scalar-last for scipy
            if q_in_wxyz: # q_orig is wxyz
                q_in_scipy = [q_in[1], q_in[2], q_in[3], q_in[0]]
            else: # q_in is xyzw
                q_in_scipy = q_in
                
            r_in = _get_scipy_rotation().from_quat(q_in_scipy)

            # Create rotation from Euler angles (in degrees)
            # Default order is 'xyz', change if needed (e.g., 'zyx', 'xyz', etc.)
            r_delta = _get_scipy_rotation().from_euler('xyz', euler_deg, degrees=True)

            # Apply the new rotation
            r_new = r_delta * r_in  # r_delta is applied first

            # Convert result back to scalar-first
            q_new = r_new.as_quat()   # [x, y, z, w]
            if q_out_wxyz: # return wxyz format
                q_out = [q_new[3], q_new[0], q_new[1], q_new[2]]
            else: # return xyzw format
                q_out = q_new
            return q_out

def root(meta_cfg, out_path,stop_event, vis_mode:str, real_robot:bool=False):

    import argparse
    import os
    import shutil
    import numpy as np
    from torch.utils.checkpoint import Any
    import yaml
    from tqdm import tqdm
    from rich.progress import Progress
    from curobo.util_file import load_yaml


    #parser = argparse.ArgumentParser()
    # combo
    # parser.add_argument("--cfg", type=str, required=False, default="combo", help="path to meta config file") # --cfg combo for benchmark mode
    # parser.add_argument("--livestream", action="store_true", help="run in livestream mode")
    #args = parser.parse_args()

    # Isaac Sim
    try:
        import isaacsim
    except ImportError:
        pass
    from omni.isaac.kit import SimulationApp

    # Use simple, stable configuration like working examples
    simulation_app = SimulationApp({
        "headless": vis_mode == 'headless',
        "width": "1920", # if vis_mode != 'headless' else "800",
        "height": "1080", # if vis_mode != 'headless' else "600",
    })
    
    # simulation_app = SimulationApp({
    #     "headless": False, # vis_mode == 'headless',
    #     "width": "400", # if vis_mode != 'headless' else "800",
    #     "height": "300", # if vis_mode != 'headless' else "600",
    # })

    from projects_root.utils.helper import add_extensions 

    from isaacsim.core.utils.extensions import enable_extension
    if vis_mode == 'livestream':
        # Default Livestream settings, enable Livestream extension
        simulation_app.set_setting("/app/window/drawMouse", True)
        # _headless_mode = 'webrtc'
        _headless_mode = 'webrtc'
        # enable_extension("omni.kit.livestream.webrtc")
        init_livestream = True
    else:
        _headless_mode = True if vis_mode == 'headless' else None
        init_livestream = False
    add_extensions(simulation_app, headless_mode=_headless_mode, init_livestream_if_headless=init_livestream)
    # simulation_app.set_setting("/ngx/useDlss", False)
    # simulation_app.set_setting("/app/renderer/dlss/enable", False)
    import os
    from abc import abstractmethod
    from collections.abc import Callable
    from copy import copy, deepcopy
    import dataclasses
    from time import time, sleep
    import subprocess
    from threading import Event, Thread
    from typing import Optional, Tuple, Dict, Union, Callable
    from queue import Queue, Empty
    from typing_extensions import List
    import pickle
    import threading
    import torch
    import pandas as pd
    import random
    from datetime import datetime
    # os.environ.setdefault("MPLBACKEND", "Agg")  # Set non-interactive backend before any Matplotlib imports
    import numpy as np
    import asyncio # frame capturing
    import signal # simulation stopping
    # import sys
    import carb
    from omni.isaac.core import World
    from omni.isaac.core.objects import cuboid, sphere
    from omni.isaac.core.utils.types import ArticulationAction
    from omni.isaac.core.utils.types import JointsState as isaac_JointsState
    from isaacsim.core.utils.xforms import get_world_pose
    import omni
    from pxr import UsdGeom, Gf, Sdf, UsdPhysics
    from omni.isaac.core.objects import DynamicCuboid, VisualCuboid, VisualSphere, DynamicSphere, FixedCuboid, FixedSphere
    from omni.isaac.core.utils.viewports import set_camera_view
    from omni.isaac.core.utils.stage import open_stage, clear_stage # sim reset
    import omni.replicator.core as rep
    from omni.replicator.core import BasicWriter

    # frame capturing (extensions enabled lazily only if needed later)


    # CuRobo
    from curobo.types.tensor import T_DOF
    from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.geom.types import WorldConfig, Sphere
    from curobo.rollout.rollout_base import Goal
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import JointState, RobotConfig
    from curobo.types.state import JointState, FilterCoeff
    from curobo.util.logger import setup_curobo_logger
    from curobo.util.usd_helper import UsdHelper
    from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    from projects_root.utils.helper import add_robot_to_scene
    from projects_root.projects.dynamic_obs.dynamic_obs_predictor.runtime_topics import init_runtime_topics, get_topics
    from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import DynamicObsCollPredictor
    from curobo.rollout.cost.custom.arm_base.dynamic_obs_cost import DynamicObsCost
    from curobo.opt.particle.parallel_mppi import ParallelMPPI
    from curobo.rollout.arm_reacher import ArmReacher
    from curobo.wrap.wrap_mpc import WrapMpc
    from projects_root.projects.dynamic_obs.dynamic_obs_predictor.frame_utils import FrameUtils
    from projects_root.utils.world_model_wrapper import WorldModelWrapper
    from projects_root.utils.usd_pose_helper import get_stage_poses, list_relevant_prims
    from projects_root.utils.transforms import transform_poses_batched_optimized_for_spheres, transform_poses_batched, transform_robot_positions_to_world
    from projects_root.utils.draw import draw_points
    from projects_root.utils.colors import npColors
    from projects_root.utils.issacsim import  activate_gpu_dynamics
    from projects_root.real_world.servers.client_tools import get_joint_states, send_joint_commands
    

    class Stopwatch():
        def __init__(self):
            self.total = 0.0
        
        def on(self):
            self.start_time = time()
            
        def off(self):
            self.end_time = time()
            new =  self.end_time - self.start_time
            self.total += new
            return new
        
        def reset(self):
            self.total = 0.0
            self.start_time = 0.0
            self.end_time = 0.0
    
        
    class Plan:
        def __init__(self, cmd_idx=0, cmd_plan:Optional[JointState]=None):
            self.cmd_idx = cmd_idx
            self.cmd_plan = cmd_plan
        
        def _is_finished(self):
            exhausted_plan =  self.cmd_idx >= len(self.cmd_plan.position) - 1
            if exhausted_plan:
                print(f"DEBUG exhausted_plan: {exhausted_plan} after {self.cmd_idx} steps")
            return exhausted_plan
        
        def consume_action(self):
            if self.cmd_plan is None or self._is_finished(): # if no plan or plan is exhausted, reset the plan
                self.cmd_idx = 0
                self.cmd_plan = None
                return None
            else: # take from plan
                cmd = self.cmd_plan[self.cmd_idx]
                self.cmd_idx += 1
                return cmd
                
    class SimEnv:
        def __init__(self,stage):
            self.stage = stage
            self.scope_path = "/World/SpawnedObs"
            UsdGeom.Scope.Define(stage, Sdf.Path(self.scope_path))

        
        def step(self,**kwargs):
            pass
    class ObjectCfg:
        def __init__(self,
            obj_shape:str='cube',
            obj_lin_vel:list[float]=[0,0,0],
            obj_rigid_body_enabled:bool=False,
            obj_size=[0.5,0.5,0.5],
            obj_pos=[0,0,0],
        ):

            self.obj_shape = obj_shape
            self.obj_size = obj_size
            self.obj_lin_vel = obj_lin_vel
            self.obj_rigid_body_enabled = obj_rigid_body_enabled
            self.obj_pos = obj_pos
        
    class PrimsEnv(SimEnv):
        def __init__(self,
            world,
            pose_utils,
            obj_cfgs:list[ObjectCfg]
            # n_obs,
            # obj_shape='cube',
            # max_dim=0.5,
            # min_dim=0.1,
            # volume_center_pos=[0,0,1],
            # volume_shape='sphere',
            # volume_dim=1,
            # obj_lin_vel=[0,0,0],
            # obj_rigid_body_enabled=False,
            ):
            """
            static obstacles.
            Args:
                stage (Usd.Stage): stage to spawn the obstacles
                obj_shape (int): number of obstacles
                type (str): type of obstacle ('cube' or 'sphere')
                seed (int): seed for random number generator
                max_dim (float): maximum dimension of the obstacle (in meters): side length for cube, diameter for sphere
                min_dim (float): minimum dimension of the obstacle (in meters): side length for cube, diameter for sphere
                volume_center_pos (list): center of the obstacle volume (xyz in world frame)
                volume_shape (str): shape of the obstacle volume ('sphere' or 'box')
                volume_dim (float): dimension of the obstacle volume (radius for sphere, side length for box)
            
            """
            # relevant prim docs: https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html#objects
            # https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.api/docs/index.html?highlight=fixedcuboid#isaacsim.core.api.objects.FixedCuboid
            super().__init__(world.stage)
            self.world:World = world
            self.obj_cfgs = obj_cfgs      
            self._pose_utils = pose_utils
            self._local_rng = random.Random(self._pose_utils.seed)
            self._objs = []
            self._objs_prim_classes = []
            for obj_cfg in obj_cfgs:            
                prim_class = None
                if  obj_cfg.obj_shape == "cube":
                    if obj_cfg.obj_lin_vel != [0,0,0]:
                        if obj_cfg.obj_rigid_body_enabled:
                            prim_class = DynamicCuboid
                        else:
                            prim_class = VisualCuboid
                    else:
                        prim_class = FixedCuboid
                elif obj_cfg.obj_shape == "sphere":
                    if obj_cfg.obj_lin_vel != [0,0,0]:
                        if obj_cfg.obj_rigid_body_enabled:
                            prim_class = DynamicSphere
                        else:
                            prim_class = VisualSphere
                    else:
                        prim_class = FixedSphere
                else:
                    raise ValueError(f"Invalid object shape: {obj_cfg.obj_shape}")
                
                # self._objs.append(Obj(obj_cfg.obj_shape, obj_cfg.obj_size, obj_cfg.obj_lin_vel, obj_cfg.obj_rigid_body_enabled))
                self._objs_prim_classes.append(prim_class)
            for obj_idx, (prim_class,obj_cfg) in enumerate(zip(self._objs_prim_classes,obj_cfgs)):
                kwargs = {}
                
                # if prim_class in [DynamicCuboid, DynamicSphere]:
                #     kwargs["linear_velocity"] = np.array(self.obj_lin_vel)
                if prim_class in [DynamicCuboid, DynamicSphere]:
                    kwargs["linear_velocity"] = np.array(obj_cfg.obj_lin_vel)

                if prim_class in [VisualSphere, DynamicSphere, FixedSphere]:
                    kwargs["radius"] = obj_cfg.obj_size / 2.0
                if prim_class in [VisualCuboid, DynamicCuboid, FixedCuboid]:
                    kwargs["scale"] = np.array(obj_cfg.obj_size) # ~https://docs.isaacsim.omniverse.nvidia.com/latest/python_scripting/core_api_overview.html

                obj_name = "Cube" if obj_cfg.obj_shape == "cube" else "Sphere"
                obj_path = f"{self.scope_path}/{obj_name}_{obj_idx}"
                
                obj_pos = Gf.Vec3d(obj_cfg.obj_pos) # may need to change this to tuple (p,q)
                # sample position in volume
                # if self.obj_volume_shape == "sphere":
                #     obj_pos = Gf.Vec3d(self._pose_utils.sample_pos_in_sphere(self.obj_volume_center_pos, self.obj_volume_dim/2))
                # elif self.obj_volume_shape == "box":
                #     obj_pos = Gf.Vec3d(self._pose_utils.sample_pos_in_box(self.obj_volume_center_pos, self.obj_volume_dim))
                # else:
                #     raise ValueError(f"Invalid volume shape: {self.obj_volume_shape}")
                
                obj = prim_class(prim_path=obj_path, name=obj_path, position=obj_pos, **kwargs)
                self._objs.append(obj)

            
        def step(self,**kwargs):
            """
            update objects according to their lin_vel
            """
            for obj_idx, obj in enumerate(self._objs):
                prim_class = self._objs_prim_classes[obj_idx]
                if prim_class in [VisualSphere, VisualCuboid]: # visual objects with velocity
                    # Get current pose as numpy array
                    p, q = get_world_pose(obj.prim_path)
                    p_np = np.array(p)
                    vel_np = np.array(self.obj_cfgs[obj_idx].obj_lin_vel)
                    dt = self.world.get_physics_dt()

                    new_p = p_np + vel_np * dt

                    # Debug
                    # print(f"debug obj {obj_idx}: p {p_np} vel {vel_np} dt {dt} new_p {new_p}")

                    obj.set_world_pose(new_p, q)

    class SimTask:
        def __init__(self, 
            agent_task_cfgs:list[dict],
            world:World, 
            usd_help:UsdHelper,        
            tensor_args:TensorDeviceType,
            level:int,
            stat_man_cfg:dict,
            ):
            
            self.agent_task_cfgs = agent_task_cfgs
            self.world = world
            self.usd_help = usd_help
            self.tensor_args = tensor_args
            self.level = level
            
            self.link_name_to_path = [{} for _ in range(len(agent_task_cfgs))]
            self.link_path_to_prim = [{} for _ in range(len(agent_task_cfgs))]
            self.name_link_to_target = [{} for _ in range(len(agent_task_cfgs))]
            self.name_target_to_link = [{} for _ in range(len(agent_task_cfgs))]
            self.target_name_to_path = [{} for _ in range(len(agent_task_cfgs))]
            self.target_path_to_prim = [{} for _ in range(len(agent_task_cfgs))]
            self.goal_errors = [{} for _ in range(len(agent_task_cfgs))]
            self._link_name_to_pose = [{} for _ in range(len(agent_task_cfgs))]
            self._target_name_to_pose = [{} for _ in range(len(agent_task_cfgs))]
            self._last_update = [{} for _ in range(len(agent_task_cfgs))] # last updated goal poses in sim
            self.link_name_to_arm_base = [{} for _ in range(len(agent_task_cfgs))]

            self.stat_man = StatManager(world, **stat_man_cfg, unique_name='task_stats')
            
            self.n_agents = len(agent_task_cfgs)
            self.centrealized = self.n_agents == 1  
            # set targets to retract poses
            for a_idx, a_cfg in enumerate(agent_task_cfgs):
                for l_idx, link_name in enumerate(a_cfg.keys()):
                    l_path, l_prim, l_target_color, l_retract_pose, arm_base_pose = a_cfg[link_name]
                    l_pose =  l_retract_pose # get_world_pose(l_path) # world.stage.GetPrimAtPath(l_path).GetWorldPose() # l_prim.get_world_pose()
                    target_name = f"target_{a_idx}_{link_name}"
                    target_path = f"/World/{target_name}"
                    l_pose = np.ravel(l_pose.to_list())
                    target_prim = cuboid.VisualCuboid(target_path, position=np.array(l_pose[:3]), orientation=np.array(l_pose[3:]), color=np.array(l_target_color), size=0.08)
                    
                    self.link_name_to_path[a_idx][link_name] = l_path
                    self.link_path_to_prim[a_idx][l_path] = l_prim
                    self.name_link_to_target[a_idx][link_name] = target_name
                    self.name_target_to_link[a_idx][target_name] = link_name
                    self.target_name_to_path[a_idx][target_name] = target_path
                    self.target_path_to_prim[a_idx][target_path] = target_prim
                    self.goal_errors[a_idx][link_name] = (-1,-1)
                    self._last_update[a_idx][link_name] = l_retract_pose
                    self.link_name_to_arm_base[a_idx][link_name] = np.ravel(arm_base_pose)
            
        def get_arms_bases_center_pos(self)->np.ndarray:
            xs = []
            ys = []
            zs = []
            for a_idx in range(len(self.link_name_to_arm_base)):
                for link_name in self.link_name_to_arm_base[a_idx]:
                    arm_base_pose = self.link_name_to_arm_base[a_idx][link_name]
                    xs.append(arm_base_pose[0])
                    ys.append(arm_base_pose[1])
                    zs.append(arm_base_pose[2])

            return np.array([np.mean(xs), np.mean(ys), np.mean(zs)])

        def get_link_name_to_pose(self)->list[dict[str,tuple[np.ndarray, np.ndarray]]]:
            return self._link_name_to_pose
        
        def get_target_name_to_pose(self)->list[dict[str,tuple[np.ndarray, np.ndarray]]]:
            return self._target_name_to_pose
        
        def _update_err(self, link_name_to_error):
            for a_idx in range(len(self.agent_task_cfgs)):
                for link_name, error in link_name_to_error[a_idx].items():
                    self.goal_errors[a_idx][link_name] = error
        @abstractmethod
        def get_stat_vals(self, stat_names:list[str])->dict[str,Any]:
            """
            get the stats values for the given stat names
            """
            pass
        
        
        def _set_target_world_pose_by_target_name(self, a_idx:int, target_name:str, new_p_target:np.ndarray, new_q_target:np.ndarray):
            target_path = self.target_name_to_path[a_idx][target_name]
            target_prim = self.target_path_to_prim[a_idx][target_path]
            target_prim.set_world_pose(position=new_p_target, orientation=new_q_target) # set_world_pose(position=new_p_target, orientation=new_q_target)
        
        def _set_target_world_pose_by_link_name(self, a_idx:int, link_name:str, new_p_target:np.ndarray, new_q_target:np.ndarray):
            target_name = self.name_link_to_target[a_idx][link_name]
            self._set_target_world_pose_by_target_name(a_idx, target_name, new_p_target, new_q_target)
        
        def _get_link_errors(self) -> tuple[list[dict[str,tuple[float,float]]], list[dict[str,tuple[np.ndarray, np.ndarray]]], list[dict[str,tuple[np.ndarray, np.ndarray]]]]:
            n_agents = len(self.agent_task_cfgs)
            link_name_to_error = [{} for _ in range(n_agents)]
            target_name_to_pose = [{} for _ in range(n_agents)]
            link_name_to_pose = [{} for _ in range(n_agents)]
            
            for a_idx in range(len(self.agent_task_cfgs)):                
                for link_name, link_path in self.link_name_to_path[a_idx].items():
                    p_link, q_link = get_world_pose(link_path)
                    link_name_to_pose[a_idx][link_name] = (p_link, q_link)
                    target_name = self.name_link_to_target[a_idx][link_name]
                    target_path = self.target_name_to_path[a_idx][target_name]
                    target_prim = self.target_path_to_prim[a_idx][target_path]
                    p_target, q_target = target_prim.get_world_pose()
                    target_name_to_pose[a_idx][target_name] = (p_target, q_target)
                    pos_err = np.linalg.norm(p_target - p_link[:3])
                    rot_err = np.linalg.norm(get_per_axis_euler_error(q_link, q_target))
                    err = (pos_err, rot_err)
                    link_name_to_error[a_idx][link_name] = err
            return link_name_to_error, target_name_to_pose, link_name_to_pose
            
        def _parse_np_to_pose(self, link_name_to_next_target_pose_np:list[dict[str,tuple[np.ndarray, np.ndarray]]])->list[dict[str,Pose]]:
            # parse new targets from np to Pose
            link_name_to_next_target_pose = [{} for _ in range(len(link_name_to_next_target_pose_np))]
            for a_idx in range(len(link_name_to_next_target_pose_np)):
                for link_name in link_name_to_next_target_pose_np[a_idx].keys():
                    p_target_new_np, q_target_new_np = link_name_to_next_target_pose_np[a_idx][link_name]
                    new_target_as_pose = Pose(
                        position=self.tensor_args.to_device(p_target_new_np), 
                        quaternion=self.tensor_args.to_device(q_target_new_np)
                    )
                    link_name_to_next_target_pose[a_idx][link_name] = new_target_as_pose
            return link_name_to_next_target_pose

        def step(self)->list[dict[str,Pose]]:
            
            errors, target_name_to_pose, link_name_to_pose  = self._get_link_errors()
            self._link_name_to_pose = link_name_to_pose
            self._target_name_to_pose = target_name_to_pose
            self._update_err(errors)


            # Update targets in sim if required, based on the custom task logic
            link_name_to_next_target_pose_np = self._update_sim_targets(errors, target_name_to_pose, link_name_to_pose)
            
            if link_name_to_next_target_pose_np is not None: # got new target poses for some links
                # We'll update the state of task, so the solvers will react to it and change their goals accordingly
                link_name_to_next_target_pose = self._parse_np_to_pose(link_name_to_next_target_pose_np) # convert given np to Pose format
                for a_idx in range(len(link_name_to_next_target_pose)):
                    for link_name in link_name_to_next_target_pose[a_idx].keys():
                        self._last_update[a_idx][link_name] = link_name_to_next_target_pose[a_idx][link_name] # update link in task state
            
            return self._last_update # return updated task state
        
        
        def _tup_to_pose(self, p:np.ndarray,q: np.ndarray)->Pose:
            return Pose(position=self.tensor_args.to_device(p), quaternion=self.tensor_args.to_device(q))
        
        def _update_target(self, p:np.ndarray,q: np.ndarray, a_idx:int, link_name:str):
            self._last_update[a_idx][link_name] = self._tup_to_pose(p,q)


        
        def check_contact(self)->list[list[int]]:
            """
            return list of pairs of robot indices that are in contact.
            For each robot, return a list of indices of robots that are in contact with it.
            
            """
            return []
        
        def get_sphere_posrad(self)->list[tuple[np.ndarray, float]]:
            """
            return a list of sphere positions and radii
            for each robot, return a list of sphere positions and radii
            """
            return []
        
        @abstractmethod
        def _update_sim_targets(self,errors, target_name_to_pose, link_name_to_pose)->Optional[list[dict[str,tuple[np.ndarray, np.ndarray]]]]:
            """
            here you perform any updates to the targets in sim if required, based on the task logic and errors.
            If target pose needs to be change in the simulation, that's the place to do it
            That way you can write your customized tasks with custom logic.
            """
            
            pass
        
        def _get_targets_world_pose(self) -> list[dict[str,tuple[np.ndarray, np.ndarray]]]:
            n_agents = len(self.link_name_to_path)
            all_target_poses = [{} for _ in range(n_agents)]
            for a_idx in range(n_agents):
                for link_name in self.link_name_to_path[a_idx].keys():
                    target_name = self.name_link_to_target[a_idx][link_name]
                    target_path = self.target_name_to_path[a_idx][target_name]
                    target_prim = self.target_path_to_prim[a_idx][target_path]                    
                    p_target, q_target = target_prim.get_world_pose() # get_world_pose(target_path)
                    all_target_poses[a_idx][target_name] = (p_target, q_target)
            return all_target_poses

        def _set_targets_world_pose(self, new_target_poses:list[dict[str,tuple[np.ndarray, np.ndarray]]]):
            n_agents = len(self.link_name_to_path)
            for a_idx in range(n_agents):
                for link_name in new_target_poses[a_idx].keys():
                    target_name = self.name_link_to_target[a_idx][link_name]
                    target_path = self.target_name_to_path[a_idx][target_name]
                    target_prim = self.target_path_to_prim[a_idx][target_path]
                    p_target, q_target = new_target_poses[a_idx][link_name]
                    target_prim.set_world_pose(position=p_target, orientation=q_target)


        def get_goal_err_by_arm(self)->list[tuple[float,float]]:
            errors = []
            # arm_idx = 0
            for a_idx in range(len(self.goal_errors)):
                for link_name in self.goal_errors[a_idx].keys():
                    p_err, q_err = self.goal_errors[a_idx][link_name]
                    errors.append((p_err, q_err))
                    # arm_idx += 1
            return errors
        
    class FollowTask(SimTask):
        def __init__(self, 
                    agents_task_cfgs:list[dict], 
                    world:World, 
                    usd_help:UsdHelper, 
                    tensor_args:TensorDeviceType, 
                    level:int,
                    stats_cfg:dict,
                    pose_utils:PoseUtils,
                    initial_vel_direction='center',
                    # target_vel_noise=0.2,
                    # target_pos_noise=0.2,
                    
                    ):
            
            """
            level:

 
                




            
            
            """
            super().__init__(agents_task_cfgs, world, usd_help, tensor_args, level, stats_cfg)
            self._pose_utils = pose_utils
            self.target_name_to_target_lin_vel = [{} for _ in range(len(agents_task_cfgs))]
            self.initial_vel_direction = initial_vel_direction # 'center' or 'none'
            # self.target_vel_noise = target_vel_noise
            # self.target_pos_noise = target_pos_noise
            # Setup targets:
            self.link_name_to_target_vel = [{} for _ in range(self.n_agents)]
            
            link_name_to_target_pose_np = [{} for _ in range(self.n_agents)]
            # robots_center = self.get_arms_bases_center_pos()
            for a_idx in range(self.n_agents):
                for link_name in self.link_name_to_path[a_idx].keys():
                    robot_base_pos = self.link_name_to_arm_base[a_idx][link_name][:3]
                    init_target_pos = robot_base_pos #robots_center # robot_base_pos + self.targets_density * (robots_center - robot_base_pos) # target is halfway between robot and center of all robots
                    # if self.target_pos_noise > 0:
                    #     for axis in range(3):
                    #         init_target_pos[axis] += self._pose_utils._local_rng.sample(list(np.arange(-self.target_pos_noise/2, self.target_pos_noise/2, self.target_pos_noise/10)),1)[0]
                    
                    init_target_pos[2] += 0.75 # m above the base
                    init_target_quat = np.array([0,1,0,0])
                    link_name_to_target_pose_np[a_idx][link_name] = (init_target_pos, init_target_quat)
                    if self.initial_vel_direction == 'center':
                        tar_vel_direction = (init_target_pos - robot_base_pos) # getting away from the robot base
                    else:
                        tar_vel_direction = np.array([0,0,0])
                    tar_vel_direction = np.array([tar_vel_direction[0], tar_vel_direction[1], 0])
                    scaled_vel = tar_vel_direction  
                    # if self.target_vel_noise > 0:
                    #     scaled_vel += self._add_noise_to_target_vel()
                        
                    self.link_name_to_target_vel[a_idx][link_name] = scaled_vel 
                    self._last_update[a_idx][link_name] = Pose(position=self.tensor_args.to_device(init_target_pos), quaternion=self.tensor_args.to_device(init_target_quat))
            self._set_targets_world_pose(link_name_to_target_pose_np)
            
            self._is_initialized = False
            self._tphysics_at_last_update = -1.0 # will be initialized 



            
        
        # def _add_noise_to_target_vel(self):
        #     noise_range = np.arange(-self.target_vel_noise/2, self.target_vel_noise/2, self.target_vel_noise/10)
        #     noise = np.zeros(3)
        #     for axis in range(3):
        #         noise_axis = self._pose_utils._local_rng.sample(list(noise_range),1)[0]
        #         print(f'noise_axis: {noise_axis}')
        #         noise[axis] = noise_axis
        #     return noise
    
        def _update_sim_targets(self, errors, target_name_to_pose, link_name_to_pose) -> List[Dict[str, Tuple[np.ndarray]]] | None:
            if not self._is_initialized:
                self._is_initialized = True
                self._tphysics_at_last_update = self.world.current_time
                
            tphysics_cur = self.world.current_time

            # update target pose in sim according to target lin vel
            tphysics_since_update = tphysics_cur - self._tphysics_at_last_update
            # if tphysics_since_update > self.update_interval_tphys:
            self._tphysics_at_last_update = tphysics_cur
            for a_idx in range(len(self.target_name_to_target_lin_vel)):
                for link_name in self.link_name_to_path[a_idx].keys():
                    target_name = self.name_link_to_target[a_idx][link_name]     
                    p_target, q_target = target_name_to_pose[a_idx][target_name]
                    
                    arm_base = deepcopy(self.link_name_to_arm_base[a_idx][link_name][:3])
                    arm_body_center = [arm_base[0], arm_base[1], 0.4] # ~ half of ur5e length. Replace with other arm length if needed!
                    too_far_from_robot = np.linalg.norm(p_target - arm_body_center) > 0.2
                    if too_far_from_robot: # if target is too far from robot, sample a new position near the arm body center
                        # print(f'debug: target {target_name} too far from robot {link_name}')
                        
                        
                        # Sample new target start position!
                        p_target = arm_body_center + np.random.uniform(-0.05, 0.05, 3) # sample a new position near the arm body center
                        for i in range(3):
                            p_target[i] += self._pose_utils._local_rng.sample(list(np.arange(-0.05, 0.05, 0.005)),1)[0]
                        
                        q_target = np.array([0,1,0,0])
                        target_name_to_pose[a_idx][target_name] = (p_target, q_target)
                        

                        # self.link_name_to_target_vel[a_idx][link_name] = np.random.uniform(-0.3, 0.3, 3)
                        # Sample new target velocity!
                        for i in range(3):
                            self.link_name_to_target_vel[a_idx][link_name][i] += self._pose_utils._local_rng.sample(list(np.arange(-0.3, 0.3, 0.03)),1)[0]
                        
                        
                        
                        self._update_target(p_target, q_target, a_idx, link_name)
                        self._set_target_world_pose_by_link_name(a_idx, link_name, p_target, q_target)
                        continue

                    # if self.add_velocity_noise: # nudge the target velocity by a small amount                        
                    #     noise = self._add_noise_to_target_vel()
                    #     self.link_name_to_target_vel[a_idx][link_name] += noise
                    
                    target_lin_vel = self.link_name_to_target_vel[a_idx][link_name]
                    p_target_new = p_target + tphysics_since_update * np.array(target_lin_vel)
                    self._update_target(p_target_new, q_target, a_idx, link_name)
                    self._set_target_world_pose_by_link_name(a_idx, link_name, p_target_new, q_target)
                    
            return None
        
        def get_stat_vals(self, stat_names:list[str])->dict[str,Any]:
            
            stats = {}
            for stat_name in stat_names:
                if stat_name == 'arm_err':
                    val = self.get_goal_err_by_arm()
                    # val = deepcopy(self.goal_errors)
                else:
                    raise ValueError(f"Invalid stat name: {stat_name}")
                stats[stat_name] = val
            return stats

    

    
    class ManualTask(SimTask):
        def __init__(self, agents_task_cfgs, world, usd_help, tensor_args, stats_cfg):
            super().__init__(agents_task_cfgs, world, usd_help, tensor_args, 1, stats_cfg)
            
        def _update_sim_targets(self, errors, target_name_to_pose, link_name_to_pose)->Optional[list[dict[str,tuple[np.ndarray, np.ndarray]]]]:
            for a_idx in range(len(errors)):
                for link_name in errors[a_idx].keys():
                    p_err, q_err = errors[a_idx][link_name]
                    target_name = self.name_link_to_target[a_idx][link_name]
                    p_target, q_target = target_name_to_pose[a_idx][target_name]
                    self._last_update[a_idx][link_name] = Pose(position=self.tensor_args.to_device(p_target), quaternion=self.tensor_args.to_device(q_target))
        
        
        def get_stat_vals(self, stat_names:list[str])->dict[str,Any]:
            return {}
                
    class ReachTask(SimTask):
        """
        level:
           
            
            

        """
        def __init__(self, agents_task_cfgs, world, usd_help, tensor_args, level, stats_cfg,pose_utils, targets_box_dim=0.5, update_interval_tphys=1.0,p_err_threh=0.05,q_err_threh=0.5):
            
            super().__init__(agents_task_cfgs, world, usd_help, tensor_args, level, stats_cfg)
            
            self._pose_utils = pose_utils
            self.p_err_threh = p_err_threh
            self.q_err_threh = q_err_threh
            self.target_name_to_target_lin_vel = [{} for _ in range(len(agents_task_cfgs))]
            self.targets_box_dim = targets_box_dim # the greater, the more spread out the targets are
            robots_center = self.get_arms_bases_center_pos()
            self.link_name_to_target_box_center = [{} for _ in range(self.n_agents)]
            target_box_h = targets_box_dim/2 + 0.2
            overlap_mode_box_center = robots_center + np.array([0,0,target_box_h])
            self.overlapping_target_boxes = level in range(6,11)

            
            

            for a_idx in range(self.n_agents):
                for link_name in self.link_name_to_path[a_idx].keys():
                    if not self.overlapping_target_boxes: # non-overlapping target boxes (easier)
                        robot_base_pos = self.link_name_to_arm_base[a_idx][link_name][:3]
                        target_box_center = robot_base_pos + np.array([0,0,target_box_h]) # robot_base_pos + 0.5 * (robots_center - robot_base_pos)  
                    else: # overlapping target boxes (harder)
                        target_box_center = overlap_mode_box_center
                    self.link_name_to_target_box_center[a_idx][link_name] = target_box_center


            # Initialize targets:
            link_name_to_target_pose_np = [{} for _ in range(self.n_agents)]
            for a_idx in range(self.n_agents):
                for link_name in self.link_name_to_path[a_idx].keys():
                    target = self._sample_target_from_box(a_idx, link_name)
                    link_name_to_target_pose_np[a_idx][link_name] = target
                    self._update_target(target[0], target[1], a_idx, link_name)
                    self._set_target_world_pose_by_link_name(a_idx, link_name, target[0], target[1])
            

            # Set initial state for time-based updates:
            self._is_initialized = False
            self.update_interval_tphys = update_interval_tphys
            self._link_name_to_last_update_tphys = [{} for _ in range(self.n_agents)]
            for a_idx in range(self.n_agents):
                for link_name in self.link_name_to_path[a_idx].keys():
                    self._link_name_to_last_update_tphys[a_idx][link_name] = -1.0 



        def _sample_target_from_box(self, a_idx:int, link_name:str):
            init_target_pos = self._pose_utils.sample_pos_in_box(self.link_name_to_target_box_center[a_idx][link_name], self.targets_box_dim)
            init_target_quat = np.array([0,1,0,0])
            target = (init_target_pos, init_target_quat)
            return target
        
                
    
        def _update_sim_targets(self, errors, target_name_to_pose, link_name_to_pose) -> List[Dict[str, Tuple[np.ndarray]]] | None:
            if not self._is_initialized:
                self._is_initialized = True
                self._tphysics_at_last_update = self.world.current_time
                
            tphysics_cur = self.world.current_time

            # Check time limit (if passed, update all targets)
            # tphysics_since_update = tphysics_cur - self._tphysics_at_last_update
            # beyond_time_lim = tphysics_since_update > self.update_interval_tphys
            # if beyond_time_lim:
            #     self._tphysics_at_last_update = tphysics_cur
            self._arm_reached_goal_cur_iter = []
            self._arm_goal_update_cur_iter = []

            arm_idx = 0
            for a_idx in range(len(self.target_name_to_target_lin_vel)):
                for link_name in self.link_name_to_path[a_idx].keys():
                    
                    reached_goal = False
                    last_update_tphys = self._link_name_to_last_update_tphys[a_idx][link_name]
                    tphysics_since_update = tphysics_cur - last_update_tphys
                    beyond_time_lim = tphysics_since_update > self.update_interval_tphys
                    
                    if not beyond_time_lim:
                        p_err, q_err = errors[a_idx][link_name]
                        reached_goal = p_err < self.p_err_threh and  q_err < self.q_err_threh
                        if reached_goal:
                            self._arm_reached_goal_cur_iter.append(arm_idx)
                    
                    update_goal = beyond_time_lim or reached_goal
                    if update_goal:
                        self._link_name_to_last_update_tphys[a_idx][link_name] = tphysics_cur
                        new_p, new_q = self._sample_target_from_box(a_idx, link_name)
                        self._update_target(new_p, new_q, a_idx, link_name)
                        self._set_target_world_pose_by_link_name(a_idx, link_name, new_p, new_q)
                        self._arm_goal_update_cur_iter.append(arm_idx)
                    
                    arm_idx += 1
            return None

        def get_stat_vals(self, stat_names:list[str])->dict[str,Any]:
            stats = {}
            for stat_name in stat_names:
                if stat_name == 'arm_err':
                    val = self.get_goal_err_by_arm()
                elif stat_name == 'arm_changed':
                    val = self._arm_goal_update_cur_iter
                    # val = self.get_n_reached_by_arm()
                elif stat_name == 'arm_reached': 
                    val = self._arm_reached_goal_cur_iter
                else:
                    raise ValueError(f"Invalid stat name: {stat_name}")
                stats[stat_name] = val
            return stats
        


    class CbsMp1Task(ManualTask):
        def __init__(self, 
        agents_task_cfgs, 
        world, 
        usd_help, 
        tensor_args,
        level,stats_cfg,base_pose,spacing=1.0,noise=False,robot_base_radius=0.025,add_walls=False):
            super().__init__(agents_task_cfgs, world, usd_help, tensor_args,stats_cfg)

            self._is_initialized = False
            print(f'debug: base_pose: {base_pose}')
            print(f'debug: spacing: {spacing}')
            print(f'debug: noise: {noise}')
            print(f'debug: robot_base_radius: {robot_base_radius}')
            print(f'debug: add_walls: {add_walls}')
            self.start_poses = base_pose
            self.goal_poses = []
            self.robot_base_radius = robot_base_radius
            
            # set goals:
            # grid_dim = (len(self.start_poses)//2 +1 ) * d
            d_start = self.start_poses[0][0]
    #        d_current = d_start
            d_final = d_start * (len(self.start_poses)//2 +1)
            self.camera_eye_pos = [d_final/2, d_final/2, 3* d_final]
            self.camera_eye_view = [d_final/2, d_final/2, 0]
            
            for i in range(len(self.start_poses)):
                xyz = self.start_poses[i][:3]
                if i % 2 == 0:
                    g = [xyz[0],d_final,0] 
                else:
                    g = [d_final,xyz[1],0]
                    # d_current += d_start
                g.extend(deepcopy(self.start_poses[i][3:]))
                self.goal_poses.append(g)
            
            
            if add_walls: # add walls to the world
        
                # add wall parallel to y  axis
                wall1 = FixedCuboid(prim_path="/World/Xform/wall1", color=np.array([1.0, 0.0, 0.0]),position=np.array([d_final + 0.4 ,d_final/2 + 0.2,0]),scale=np.array([0.1,1.5*d_final,d_final]))
                # add wall parallel to x axis
                wall2 = FixedCuboid(prim_path="/World/Xform/wall2", color=np.array([1.0, 0.0, 0.0]),position=np.array([d_final/2 + 0.2,d_final + 0.4,0]),scale=np.array([1.5*d_final,0.1,d_final]))
                
            set_camera_view(eye=[0, 0, d_final*2], target=[d_final/2, d_final/2, 0], camera_prim_path="/OmniverseKit_Persp") # https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html#omni.isaac.core.utils.viewports.set_camera_view

        

        def _update_sim_targets(self, errors, target_name_to_pose, link_name_to_pose)->Optional[list[dict[str,tuple[np.ndarray, np.ndarray]]]]:
            if not self._is_initialized:
                self._is_initialized = True
                _link_name_to_target_pose = [{} for _ in range(len(self.goal_poses))]
                for a_idx in range(len(errors)):
                    for link_name in errors[a_idx].keys():
                        p_target, q_target = self.goal_poses[a_idx][:3], self.goal_poses[a_idx][3:]
                        _link_name_to_target_pose[a_idx][link_name] = (np.array(p_target), np.array(q_target))
                        self._last_update[a_idx][link_name] = Pose(position=self.tensor_args.to_device(p_target), quaternion=self.tensor_args.to_device(q_target))

                self._set_targets_world_pose(_link_name_to_target_pose)
            return None

        @staticmethod
        def get_agents_start_positions(n_agents, spacing=1.0,noise=False,robot_base_radius=0.025,add_walls=False):
            """
            half of the robots start at along x axis, and half along y"""
            s = []
            d = spacing  # - 2*robot_base_radius # distance from origin         
            for a_idx in range(n_agents):
                if noise:
                    d += random.uniform(-d/10, d/10)
                if a_idx % 2 == 0:
                    start_pos = [d,0,0]
                else:
                    start_pos = [0,d,0] 
                    d += spacing
                
                s.append(start_pos)
            print(f'debug: start_positions for CBSMP1: {s}')
            return s

        @staticmethod
        def get_agents_goal_positions(start_positions,grid_step_size=1):
            """
            half of the robots start with qw=1, and half with qw=0
            """
            grid_edge = max(start_positions[-1][0], start_positions[-1][1]) # that last agent is the farthest, so we use its x or y as the grid edge (we dont know if agent is on x or y axis so we take the max)
            d_final = grid_edge + grid_step_size # target distance from origin 
            g = []
            for sp in start_positions:
                if sp[1] == 0: # y = 0
                    g.append([sp[0],d_final, 0]) # goind from y = 0 to y = d_final
                else: # x = 0
                    g.append([d_final, sp[1], 0]) # going from x = 0 to x = d_final
            return g
    
        def check_contact(self)->list[list[int]]:
            eps = 0.01
            link_name_to_pose = self.get_link_name_to_pose() # this is relevant only for the tiny disk robot (only link = base link  = ee link so we can use it to check contact)
            contact_matrix = [[] for _ in range(len(link_name_to_pose))]
            
            for i in range(len(link_name_to_pose)):
                for j in range(i+1, len(link_name_to_pose)):
                    base_link_pos_i = link_name_to_pose[i]["ee_link"][0] # only for the tiny disk robot (base link  = ee link)
                    base_link_pos_j = link_name_to_pose[j]["ee_link"][0]
                    base_link_radius = self.robot_base_radius
                    surface_to_surface_dist = np.linalg.norm(base_link_pos_i - base_link_pos_j) - 2*base_link_radius
                    if surface_to_surface_dist < eps:
                        contact_matrix[i].append(j)
                        contact_matrix[j].append(i)
                        
            return contact_matrix
    class RealWorldVerticallBoard(SimTask):
        def __init__(self, agents_task_cfgs, world, usd_help, tensor_args, level,
                    stats_cfg,pose_utils, 
                    p_err_threh=0.1,q_err_threh=1000.0,
                    ):
            """
            Real world vertical board task with 9 fixed goal poses and fixed picking points.
            """
            super().__init__(agents_task_cfgs, world, usd_help, tensor_args, level, stats_cfg)
            self.pose_utils = pose_utils
            self._local_rng = random.Random(self.pose_utils.seed)
            self.n_agents = len(self.agent_task_cfgs)
            self._is_initialized = False
            self.p_err_threh = p_err_threh
            self.q_err_threh = q_err_threh
            time_string = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs('./board_task_logs', exist_ok=True)
            self.logs_csv_path = f'./board_task_logs/{time_string}_seed_{self.pose_utils.seed}.csv' 
            if os.path.exists(self.logs_csv_path):
                os.remove(self.logs_csv_path)
            self.csv_log = open(self.logs_csv_path, 'w')
            self.csv_writer = csv.writer(self.csv_log)
            self.csv_writer.writerow(['agent_idx', 't_placed','target_id'])
            
            # statistics:
            self.link_name_to_placed_in_board = [{} for _ in range(len(self.agent_task_cfgs))] 
            self.link_name_to_picked_from_back = [{} for _ in range(len(self.agent_task_cfgs))] 

            # Fixed Board Goal Poses
            self.board_goal_poses = [
                ([0.42, 0.15, 1.15], [1,0,0,0]),
                ([0.42, 0.0, 1.15], [1,0,0,0]),
                ([0.42, -0.15, 1.15], [1,0,0,0]),
                ([0.42, 0.15, 1.05], [1,0,0,0]),
                ([0.42, 0.0, 1.05], [1,0,0,0]),
                ([0.42, -0.15, 1.05], [1,0,0,0]),
                ([0.42, 0.15, 0.95], [1,0,0,0]),
                ([0.42, 0.0, 0.95], [1,0,0,0]),
                ([0.42, -0.15, 0.95], [1,0,0,0]),
            ]
            
            # Define behind arm goals (pick goals):
            self.link_name_to_pick_pose = [{} for _ in range(self.n_agents)]

            # Fixed Picking Points per arm index
            # Arm 0: [0, -0.4, 0.7]
            # Arm 1: [0, 0.4, 0.7]
            # Orientation: [0, 1, 0, 0] (facing down)
            
            self.pick_points = {
                0: ([0.25, 0.3, 0.6], [0,1,0,0]),
                1: ([0.25, -0.3, 0.6], [0,1,0,0])
            }

            for agent_idx in range(self.n_agents):
                # In centralized case, n_agents=1 but it might have multiple arms. 
                # However, SimTask structure usually assumes 1 agent = 1 robot/planner.
                # Just in case, we map agent_idx to the pick point.
                # If we have multiple arms in one agent (e.g. centralized), we need to check how links are mapped.
                # Based on BinTask, it iterates over links.
                
                # We will assume agent_idx corresponds to the arm index for picking point assignment if n_agents > 1
                # If n_agents == 1 (centralized), we iterate links and assign based on link index if possible, or just alternate?
                # The user requirement: "Robot arm 0... Robot arm 1..."
                
                # Let's try to infer arm index from agent index or link index
                
                start_arm_idx = agent_idx # Offset if multiple agents
                
                link_count = 0
                for link_name in self.link_name_to_path[agent_idx]:
                     # Determine effective arm index. 
                     # If decentralized, agent_idx is the arm index (0, 1)
                     # If centralized, likely 1 agent with multiple links.
                    effective_arm_idx = start_arm_idx + link_count
                    
                    if effective_arm_idx in self.pick_points:
                        pick_pose_data = self.pick_points[effective_arm_idx]
                        pick_p = np.array(pick_pose_data[0])
                        pick_q = np.array(pick_pose_data[1])
                        self.link_name_to_pick_pose[agent_idx][link_name] = (pick_p, pick_q)
                    else:
                        print(f"Warning: No pick point defined for arm index {effective_arm_idx}, using default 0")
                        pick_pose_data = self.pick_points[0]
                        pick_p = np.array(pick_pose_data[0])
                        pick_q = np.array(pick_pose_data[1])
                        self.link_name_to_pick_pose[agent_idx][link_name] = (pick_p, pick_q)

                    link_count += 1

            # setup tracking of board goals taking by each link:
            self._link_name_to_cur_boardgoal = [{} for _ in range(len(self.agent_task_cfgs))] 
            for a_idx in range(len(self.agent_task_cfgs)):
                for link_name in self.link_name_to_pick_pose[a_idx]: # Iterate based on configured links
                    self._link_name_to_cur_boardgoal[a_idx][link_name] = -1 

            self.max_concurrent_board_goals = 2 # Allow multiple
            self.force_unique_goals = True # Level 1 usually simple, but let's enforce unique to prevent collision at goal

        def _update_sim_targets(self, errors, target_name_to_pose, link_name_to_pose)->Optional[list[dict[str,tuple[np.ndarray, np.ndarray]]]]:
            
            self._last_step_picks = []
            self._last_step_drops = []

            _link_name_to_target_pose_np = [{} for _ in range(len(self.agent_task_cfgs))]
            
            if not self._is_initialized: # Initialize the targets
                self._is_initialized = True
                self._link_name_to_goal_type = [{} for _ in range(len(self.agent_task_cfgs))]
                self._goal_types = ['board', 'pick_point']
                goal_type = self._goal_types[1] # all arms start with pick point goal (picking)

                for a_idx in range(len(self.agent_task_cfgs)):
                    for link_name in link_name_to_pose[a_idx]:
                        self._link_name_to_goal_type[a_idx][link_name] = goal_type
                        # Set target to pick pose
                        _link_name_to_target_pose_np[a_idx][link_name] = self.link_name_to_pick_pose[a_idx][link_name]
            
            else: # Check updates
                arm_idx = 0
                for a_idx in range(len(self.agent_task_cfgs)):
                    for link_name in link_name_to_pose[a_idx]:
                        err_p, err_q =  errors[a_idx][link_name]
                        cur_goal_type = self._link_name_to_goal_type[a_idx][link_name]
                        
                        reached_goal = err_p < self.p_err_threh # and err_q < self.q_err_threh
                        
                        if reached_goal: 
                            if cur_goal_type == 'board': # PLACED at board
                                goal_type = 'pick_point' # Next is pick
                                goal_id = self._link_name_to_cur_boardgoal[a_idx][link_name] + 1
                                goal_pose = self.link_name_to_pick_pose[a_idx][link_name]
                                self._last_step_drops.append(arm_idx)
                                
                                # release board goal
                                self._link_name_to_cur_boardgoal[a_idx][link_name] = -1 
                                
                                self.csv_writer.writerow([a_idx, time(),goal_id])
                                self.csv_log.flush()

                                # print in green color
                                # print(f"\033[92mdebug: arm_idx {arm_idx}, placed {error_cm:.0f} cm\033[0m")

                            else: # PICKED from pick point
                                goal_type = 'board' # Next is board

                                # Choose a board goal
                                taken_goals = []
                                for a2_idx in range(len(self.agent_task_cfgs)):
                                    for link_name2 in self._link_name_to_cur_boardgoal[a2_idx]:
                                        if self._link_name_to_cur_boardgoal[a2_idx][link_name2] != -1:
                                            taken_goals.append(self._link_name_to_cur_boardgoal[a2_idx][link_name2])
                                
                                free_goals = [i for i in range(len(self.board_goal_poses)) if i not in taken_goals]
                                
                                if not free_goals and self.force_unique_goals:
                                    continue # Wait for free goal
                                
                                if self.force_unique_goals:
                                    options = free_goals
                                else:
                                    options = list(range(len(self.board_goal_poses)))
                                
                                if not options: # Should not happen if not forced unique or enough goals
                                     continue

                                new_goal_idx = self._local_rng.choice(options)
                                self._link_name_to_cur_boardgoal[a_idx][link_name] = new_goal_idx
                                
                                # Get goal pose components
                                goal_pos_list, goal_quat_list = self.board_goal_poses[new_goal_idx]
                                goal_pose = (np.array(goal_pos_list), np.array(goal_quat_list))
                                
                                self._last_step_picks.append(arm_idx)

                            _link_name_to_target_pose_np[a_idx][link_name] = goal_pose
                            self._link_name_to_goal_type[a_idx][link_name] = goal_type
                            # print in yellow
                            # print(f"\033[93mdebug: arm_idx {arm_idx} picked \033[0m")

                        else:
                            goal_id = self._link_name_to_cur_boardgoal[a_idx][link_name] + 1
                            error_cm = err_p * 100
                            print(f"debug board task: arm_idx {arm_idx}, goal id: {goal_id}, error(cm) {error_cm:.1f},tol(cm) {self.p_err_threh*100:.0f}")

                            # print in green
                            # print(f"\033[92mdebug: arm_idx {arm_idx}, placed with error {error_cm:.0f} cm\033[0m")
                        arm_idx += 1

            # update the targets in sim
            self._set_targets_world_pose(_link_name_to_target_pose_np)
            return _link_name_to_target_pose_np
        
        def get_stat_vals(self, stat_names:list[str])->dict[str,Any]:
            stats = {}
            for stat_name in stat_names:
                if stat_name == 'arm_err':
                    val = self.get_goal_err_by_arm()
                elif stat_name == 'arm_picks':
                    val = self._last_step_picks
                elif stat_name == 'arm_drops':
                    val = self._last_step_drops
                else:
                    raise ValueError(f"Invalid stat name: {stat_name}")
                stats[stat_name] = val
            return stats
    class BinTask(SimTask):
        def __init__(self, agents_task_cfgs, world, usd_help, tensor_args, level,
                    stats_cfg,pose_utils, wall_dims_hwd=np.array([0.5,0.5,0.3]), 
                    p_err_threh=0.05,q_err_threh=0.5,
                    bin_pos=None,
                    ):
            """
            all arms start with a behind-back goal. 
            level:
                1: 1-IN: one arm is allowed to go to the bin (1 bin goal at a time). 
                2: 2-IN-UNQIUE: 2 arms are allowed to go to the bin but goals are unique.
                3. ALL-IN-UNQIUE: all arms are allowed to go to the bin but goals are unique.
                4. 2-IN-NOT-UNQIUE: 2 arms are allowed to go to the bin and goals may not be unique.
                5. ALL-IN-NOT-UNQIUE: all arms are allowed to go to the bin and goals may not be unique.

            """
            super().__init__(agents_task_cfgs, world, usd_help, tensor_args, level, stats_cfg)
            self.pose_utils = pose_utils
            self._local_rng = random.Random(self.pose_utils.seed)
            self.n_agents = len(self.agent_task_cfgs)
            self._is_initialized = False
            self.p_err_threh = p_err_threh
            self.q_err_threh = q_err_threh
            # statistics:
            self.link_name_to_placed_in_bin = [{} for _ in range(len(self.agent_task_cfgs))] # statistics - num of placed items in bin
            self.link_name_to_picked_from_back = [{} for _ in range(len(self.agent_task_cfgs))] # statistics - num of picked items from back


            # Spawn Bin:
            height, width, depth = wall_dims_hwd
            if bin_pos is None:
                bin_pos = self.get_arms_bases_center_pos().tolist()

            dim0_step_size = width/2 +depth/2
            dim1_step_size = width/2 +depth/2
            dim2_step_size = height/2
            
            bin_to_out0_step = np.array([0,dim1_step_size,dim2_step_size])
            bin_to_out1_step = np.array([dim0_step_size,0,dim2_step_size])
            bin_to_out2_step = np.array([0,-dim1_step_size,dim2_step_size])
            bin_to_out3_step = np.array([-dim0_step_size,0,dim2_step_size])

            out_wall0_pos = bin_pos + bin_to_out0_step
            out_wall1_pos = bin_pos + bin_to_out1_step
            out_wall2_pos = bin_pos + bin_to_out2_step
            out_wall3_pos = bin_pos + bin_to_out3_step

            out_wall_scale = np.array([width,depth,height])
            in_wall_scale = np.array([width,depth/2,height])
            
            # Spawn internal bin walls:
            in_wall_pos = bin_pos + np.array([0,0,dim2_step_size])
            
            rotated_walls_quat = self.pose_utils.rotate_quat([1,0,0,0], [0,0,90], q_in_wxyz=True, q_out_wxyz=True)
            out_wall_1_quat = rotated_walls_quat
            out_wall_3_quat = rotated_walls_quat
            in_wall_1_quat = rotated_walls_quat
            root_stage = '/World/SpawnedObs/Bin'
            color = np.array([0.2, 0.2, 0.2])
        
            out_wall0 = FixedCuboid(prim_path=f"{root_stage}/Out0", 
                                color=color,position=out_wall0_pos, scale=out_wall_scale)
            out_wall1 = FixedCuboid(prim_path=f"{root_stage}/Out1", 
                                color=color,position=out_wall1_pos,orientation=out_wall_1_quat, scale=out_wall_scale)
            out_wall2 = FixedCuboid(prim_path=f"{root_stage}/Out2", 
                                color=color,position=out_wall2_pos, scale=out_wall_scale)
            out_wall3 = FixedCuboid(prim_path=f"{root_stage}/Out3", 
                                color=color,position=out_wall3_pos, orientation=out_wall_3_quat, scale=out_wall_scale)
            
            in_wall0 = FixedCuboid(prim_path=f"{root_stage}/In0", 
                                color=color,position=in_wall_pos, scale=in_wall_scale)
            in_wall1 = FixedCuboid(prim_path=f"{root_stage}/In1", 
                                color=color,position=in_wall_pos, orientation=in_wall_1_quat, scale=in_wall_scale)
            
            quarter0_pos = bin_pos + bin_to_out0_step / 2 + bin_to_out1_step /2


            quarter1_pos = bin_pos + bin_to_out0_step / 2 + bin_to_out3_step /2
            quarter2_pos = bin_pos + bin_to_out2_step / 2 + bin_to_out1_step /2
            quarter3_pos = bin_pos + bin_to_out2_step / 2 + bin_to_out3_step /2
            
            # Define bin goals (drop goals):
            self.bin_goal_poses = []
            for quarter_pos in [quarter0_pos, quarter1_pos, quarter2_pos, quarter3_pos]:
                goal_pos = copy(quarter_pos)
                goal_pos[2] = height + 0.35 # drop x cm above the bin
                goal_quat = np.array([0,1,0,0]) # facing down (grasp rotation)
                pose = (goal_pos, goal_quat)
                self.bin_goal_poses.append(pose)

            # Define behind arm goals (pick goals):
            self.link_name_to_pick_pose = [{} for _ in range(self.n_agents)]
            # is_centralized_planner = self.n_agents == 1
            bin_ground_pos = np.array(bin_pos[:3])


            for agent_idx in range(self.n_agents):
                for link_idx, link_name in enumerate(self.link_name_to_path[agent_idx]):
                    arm_base_pose = self.link_name_to_arm_base[agent_idx][link_name]
                    robot_ground_pos = arm_base_pose[:3]
                    robot_minus_bin =  robot_ground_pos - bin_ground_pos
                    behind_arm_goal_pos = robot_ground_pos + 0.75 * robot_minus_bin 
                    behind_arm_goal_pos[2] = behind_arm_goal_pos[2] + 0.7 # 0.5 m above the base
                    behind_arm_goal_quat = np.array([0,1,0,0]) # PoseUtils.rotate_quat([0,1,0,0], [0,0,180.0], q_in_wxyz=True, q_out_wxyz=True)# np.array([0,1,0,0]) # facing down (grasp rotation)
                    behind_arm_goal_pose = (behind_arm_goal_pos, behind_arm_goal_quat)
                    self.link_name_to_pick_pose[agent_idx][link_name] = behind_arm_goal_pose

        
                
        
            # setup tracking of bin goals taking by each link:
            self._link_name_to_cur_bingoal = [{} for _ in range(len(self.agent_task_cfgs))] # link name to current bin goal (index of the goal in self.bin_goal_poses)
            for a_idx in range(len(self.agent_task_cfgs)):
                for link_name in self.link_name_to_placed_in_bin[a_idx]:
                    self._link_name_to_cur_bingoal[a_idx][link_name] = -1 # no goals assigned yet (yet)

            # setup max concurrent bin goals:
            self.max_concurrent_bin_goals = None # num of links that can go to the bin at the same time
            if self.level == 1:
                self.max_concurrent_bin_goals = 1
            elif self.level == 2 or self.level == 4:
                self.max_concurrent_bin_goals = 2


            self.force_unique_goals = self.level < 4 # if we want each link to have a unique goal (not the same as other links)
        
        def _increase_picked_count(self, link_name:str, a_idx:int):
            if link_name not in self.link_name_to_picked_from_back[a_idx]:
                self.link_name_to_picked_from_back[a_idx][link_name] = 0
            self.link_name_to_picked_from_back[a_idx][link_name] += 1

        def _increase_placed_count(self, link_name:str, a_idx:int):
            if link_name not in self.link_name_to_placed_in_bin[a_idx]:
                self.link_name_to_placed_in_bin[a_idx][link_name] = 0
            self.link_name_to_placed_in_bin[a_idx][link_name] += 1

        def _rotate_bin_goal_for_robot(self, link_name:str, a_idx:int,bin_goal:tuple[np.ndarray, np.ndarray])->tuple[np.ndarray, np.ndarray]:
            original_goal_pos, original_goal_quat = bin_goal
            arm_base = self.link_name_to_arm_base[a_idx][link_name]
            arm_base_quat = arm_base[3:]
            arm_base_quat = PoseUtils.rotate_quat(arm_base_quat, [-180,0, 0], q_in_wxyz=True, q_out_wxyz=True) # udpside down
            new_goal = (original_goal_pos, np.array(arm_base_quat))
            return new_goal
        
        def _update_sim_targets(self, errors, target_name_to_pose, link_name_to_pose)->Optional[list[dict[str,tuple[np.ndarray, np.ndarray]]]]:
            
            self._last_step_picks = []
            self._last_step_drops = []

            _link_name_to_target_pose_np = [{} for _ in range(len(self.bin_goal_poses))]
            if not self._is_initialized: # Initialize the targets
                self._is_initialized = True
                self._link_name_to_goal_type = [{} for _ in range(len(self.agent_task_cfgs))]
                self._goal_types = ['bin', 'behind_arm']
                goal_type = self._goal_types[1] # all arms start with behind arm goal                    

                self._target_name_to_moving_twin_prim = [{} for _ in range(len(self.agent_task_cfgs))] # visual effects
                self._target_name_to_free_fall_count = [{} for _ in range(len(self.agent_task_cfgs))] # visual effects
                
                for a_idx in range(len(self.agent_task_cfgs)):
                    for link_name in link_name_to_pose[a_idx]:
                        self._link_name_to_goal_type[a_idx][link_name] = goal_type
                        if goal_type == 'bin':
                            goal_pose = self._local_rng.choice(self.bin_goal_poses)
                            _link_name_to_target_pose_np[a_idx][link_name] = self._rotate_bin_goal_for_robot(link_name, a_idx, goal_pose) # goal_pose
                        else: # behind arm
                            _link_name_to_target_pose_np[a_idx][link_name] = self.link_name_to_pick_pose[a_idx][link_name]
                    
                

            
            else: # check which agents reached their goal, and update the goal type for them (other agents keep their goal type)
                arm_idx = 0
                for a_idx in range(len(self.agent_task_cfgs)):
                    
                    for link_name in link_name_to_pose[a_idx]:
                        err_p, err_q =  errors[a_idx][link_name]
                        # print(f"debug err_p: {err_p}, err_q: {err_q}")
                        cur_goal_type = self._link_name_to_goal_type[a_idx][link_name]
                        target_name = self.name_link_to_target[a_idx][link_name]
                        twin_exists = target_name in self._target_name_to_moving_twin_prim[a_idx] # target has a carried twin for visual effect
                        target_prim = self.target_path_to_prim[a_idx][self.target_name_to_path[a_idx][target_name]]

                        reached_goal = err_p < self.p_err_threh and err_q < self.q_err_threh
                        # print(f'debug err_p: {err_p}, err_q: {err_q}, reached_goal: {reached_goal}')
                        if reached_goal: 
                            if cur_goal_type == 'bin': # PLACED

                                goal_pose = self.link_name_to_pick_pose[a_idx][link_name] # next goal pose
                                goal_type = 'behind_arm' # next goal type
                                # self._increase_placed_count(link_name, a_idx) # update stats
                                self._last_step_drops.append(arm_idx)
                                
                                # mark link as not having bin goal (it's status is now picking, not placing)
                                self._link_name_to_cur_bingoal[a_idx][link_name] = -1 # makrk link as not having bin goal
                                
                                # post-place visualization effect
                                if twin_exists: 
                                    twin = self._target_name_to_moving_twin_prim[a_idx][target_name]
                                    # reset target color to original
                                    target_prim.get_applied_visual_material().set_color(twin.get_applied_visual_material().get_color()) # set target to white (to differentiate from the moving twin)                          
                                    # hide the twin (we just placed the item in the bin, we dont need to render until picked up again)
                                    self._target_name_to_free_fall_count[a_idx][target_name] = 50 # retset to 10 steps to free fall
                                    
                            else: # PICKED
                                goal_type = 'bin' # next goal type

                                # Check availablity to take a new bin goal:
                                taken_bin_goals = []
                                for a2_idx in range(len(self.agent_task_cfgs)): # including self for the centralized case
                                    for link_name2 in self._link_name_to_cur_bingoal[a2_idx]:
                                        if self._link_name_to_cur_bingoal[a2_idx][link_name2] != -1: # if link is aiming to a bin goal
                                            taken_bin_goals.append(self._link_name_to_cur_bingoal[a2_idx][link_name2]) # add the bin goal to taken bin goals
                                free_bin_goals = [i for i in range(len(self.bin_goal_poses)) if i not in taken_bin_goals] # all bin goals that are not taken by other agents
                                n_free_bin_goals = len(free_bin_goals)
                                n_taken_bin_goals = len(taken_bin_goals)
                                n_total_bin_goals = n_free_bin_goals + n_taken_bin_goals
                                max_conc_bin_goals = self.max_concurrent_bin_goals if self.max_concurrent_bin_goals is not None else n_total_bin_goals
                                if n_taken_bin_goals >= max_conc_bin_goals: # no more bin goals to take
                                    continue

                                # Set new bin goal for the link (if possible)
                                options = [] # list of bin goal indices to choose from
                                if self.force_unique_goals: # if we want each link to have a unique goal (not the same as other links)
                                    if n_free_bin_goals > 0: # free bin goal exists
                                        options = free_bin_goals # limit options to free bin goals
                                    else: # no free bin goals, so we keep the same goal for now
                                        continue
                                else:
                                    options = list(range(len(self.bin_goal_poses))) # all bin goals are optional
                                new_bin_goal_idx = self._local_rng.choice(options) # index of free bin goal
                                self._link_name_to_cur_bingoal[a_idx][link_name] = new_bin_goal_idx # # occupy goal (mark as taken by this agent)
                                goal_pose = self._rotate_bin_goal_for_robot(link_name, a_idx, self.bin_goal_poses[new_bin_goal_idx]) # self.bin_goal_poses[new_bin_goal_idx] # New bin goal set

                                # uptdate stats (note that its done only after we actually set the new bin goal, so we count only one pick for each change from behind goal to bin goal)
                                # self._increase_picked_count(link_name, a_idx) # update stats
                                self._last_step_picks.append(arm_idx)
                                
                                # post-pick visual effects
                                if twin_exists: # if twin exists
                                    # show the twin (carried item)
                                    twin = self._target_name_to_moving_twin_prim[a_idx][target_name]
                                    twin.set_visibility(True)
                                    # set target color to white (to differentiate from the moving twin)
                                    original_color_darker = twin.get_applied_visual_material().get_color()/8
                                    # set target (around bin) color to original color but darker
                                    target_prim.get_applied_visual_material().set_color(original_color_darker)                           

                            _link_name_to_target_pose_np[a_idx][link_name] = goal_pose
                            self._link_name_to_goal_type[a_idx][link_name] = goal_type
                            
                        # visual effects:
                        else: # not yet in goal (still moving)
                            if cur_goal_type == 'bin': # carrying item to bin
                                target_path = self.target_name_to_path[a_idx][target_name]
                                target_prim = self.target_path_to_prim[a_idx][target_path]
                                link_pos, link_quat = link_name_to_pose[a_idx][link_name]                            
                                if target_name not in self._target_name_to_moving_twin_prim[a_idx]: # only once
                                    target_vis_material = target_prim.get_applied_visual_material()
                                    self._target_name_to_moving_twin_prim[a_idx][target_name] = cuboid.VisualCuboid(target_path + "_moving", position=link_pos, orientation=link_quat, color=target_vis_material.get_color(), size=target_prim.get_size())
                                else: # let the item create the illusion of picked and moves with the link
                                    self._target_name_to_moving_twin_prim[a_idx][target_name].set_world_pose(position=link_pos, orientation=link_quat)
                            
                            # visual free fall for placed item if needed:
                            else: # heading to behind arm goal
                                if target_name in self._target_name_to_free_fall_count[a_idx]:
                                    if self._target_name_to_free_fall_count[a_idx][target_name] > 0: 
                                        if twin_exists:
                                            twin = self._target_name_to_moving_twin_prim[a_idx][target_name]
                                            # update the twin position (free fall)
                                            twin_p, twin_q = twin.get_world_pose()
                                            twin_p[2] -= 0.01
                                            self._target_name_to_moving_twin_prim[a_idx][target_name].set_world_pose(position=twin_p, orientation=twin_q)
                                            
                                            # if falling for too long, hide the twin
                                            self._target_name_to_free_fall_count[a_idx][target_name] -= 1
                                            if self._target_name_to_free_fall_count[a_idx][target_name] == 0:
                                                twin.set_visibility(False)
                                            
                        arm_idx += 1

            # update the targets in sim
            self._set_targets_world_pose(_link_name_to_target_pose_np)
            return _link_name_to_target_pose_np
        
        def get_stat_vals(self, stat_names:list[str])->dict[str,Any]:
            
            stats = {}
            for stat_name in stat_names:
                if stat_name == 'arm_err':
                    val = self.get_goal_err_by_arm()
                elif stat_name == 'arm_picks':
                    val = self._last_step_picks
                elif stat_name == 'arm_drops':
                    val = self._last_step_drops
                else:
                    raise ValueError(f"Invalid stat name: {stat_name}")
                stats[stat_name] = val
            return stats
        

    class PlanPubSub:
        def __init__(self,
            pub_cfg:dict,
            sub_cfg:dict,
            valid_spheres:int,
            total_spheres:int,
            
            ):
            self.pub_cfg = pub_cfg
            self.sub_cfg = sub_cfg
            self.valid_spheres = valid_spheres
            self.total_spheres = total_spheres
            
        def should_pub_now(self, t:int)->bool:


            def bernoulli():
                return random.random() <= self.pub_cfg["pr"]
            if self.pub_cfg["is_dt_in_sec"]:
                if not hasattr(self, "_last_pub_time"):
                    self._last_pub_time = time()
                return time() - self._last_pub_time >= self.pub_cfg["dt"] and bernoulli()
            else:
                return t % self.pub_cfg["dt"] == 0 and bernoulli()
        

    class CuPlanner:
        def __init__(self,
            base_pose:list[float], 
            solver:Union[MotionGen, MpcSolver], 
            solver_config:Union[MotionGenConfig, MpcSolverConfig], 
            ordered_j_names:list[str],
            robot_cfg:dict
            ):
            
            self.base_pose = base_pose # robot base pose in world frame
            self.solver = solver
            self.solver_config = solver_config
            self.ordered_j_names = ordered_j_names
            self.is_cpred_initialized = False
            # setup all "goal-constrained" links (links that we can set goals for): ee (musts) + (optional) extra links:
            self.ee_link_name:str = self.solver.kinematics.ee_link # end effector link name, based on the robot config
            self.constrained_links_names:list[str] = copy(self.solver.kinematics.link_names) # all links that we can set goals for (except ee link), based on the robot config
            if self.ee_link_name in self.constrained_links_names: # ee link should not be in extra links, so we remove it
                self.constrained_links_names.remove(self.ee_link_name)
            
            # buffer for the state of the current planning goals (for ee link + extra constrained links)
            self.plan_goals:dict[str, Pose] = {} 

            self.crm = CudaRobotModel(CudaRobotModelConfig.from_data_dict(robot_cfg))
            
        def _set_goals_to_retract_state(self):
            """
            init the plan goals to be as the retract state
            """

            # get the current state of the robot (at retract configuration) :
            retract_kinematics_state = self.solver.kinematics.get_state(self.solver.get_retract_config().view(1, -1))
            links_retract_poses = retract_kinematics_state.link_pose
            ee_retract_pose = retract_kinematics_state.ee_pose

            initial_goals_R = {self.ee_link_name: ee_retract_pose}
            for link_name in self.constrained_links_names:
                initial_goals_R[link_name] = links_retract_poses[link_name]

            initial_goals_W = self.goals_dict_R_to_W(self.base_pose, initial_goals_R)
            self.plan_goals = initial_goals_W

        def yield_action(self, **kwargs)->Optional[JointState]:
            pass
        

        def _outdated_plan_goals(self, goals:dict[str, Pose]):
            """
            check if the current plan goals are outdated
            """
            for link_name, goal in goals.items():
                # if link_name not in self.plan_goals or torch.norm(self.plan_goals[link_name].position - goal.position) > 1e-3 or torch.norm(self.plan_goals[link_name].quaternion - goal.quaternion) > 1e-3:

                updated_goal_pos = goal.position.flatten().cpu().numpy()
                updated_goal_quat = goal.quaternion.flatten().cpu().numpy()

                current_solver_goal_pos = self.plan_goals[link_name].position.flatten().cpu().numpy()
                current_solver_goal_quat = self.plan_goals[link_name].quaternion.flatten().cpu().numpy()
                # print(f"DEBUG current_solver_goal_pos: {current_solver_goal_pos}, updated_goal_pos: {updated_goal_pos}")
                # print(f"DEBUG current_solver_goal_quat: {current_solver_goal_quat}, updated_goal_quat: {updated_goal_quat}")
                pos_changed = np.linalg.norm(current_solver_goal_pos - updated_goal_pos) > 1e-9
                rot_changed = np.linalg.norm(get_per_axis_euler_error(list(current_solver_goal_quat), list(updated_goal_quat))) > 1e-9
                # print(f"DEBUG pos_changed: {pos_changed}, rot_changed: {rot_changed}")

                if pos_changed or rot_changed:
                    return True
                
                
                # if pos_changed:

                # if link_name not in self.plan_goals or pos_changed or rot_changed:
                #     print(f"plan goals are outdated for link {link_name}")
                #     return True
                
            return False
        
        def convert_action_to_isaac(
                self, 
                full_js_action:JointState, 
                sim_js_names:list[str], 
                order_finder:Callable
            )-> ArticulationAction:

            """
            A utility function to convert curobo action to isaac sim action (ArticulationAction).
            """
            # get only joint names that are in both:
            art_action_idx_list = []
            common_js_names = []
            for x in sim_js_names:
                if x in full_js_action.joint_names:
                    art_action_idx_list.append(order_finder(x))
                    common_js_names.append(x)
        
            full_ordered_js_action = full_js_action.get_ordered_joint_state(common_js_names)
            articulation_action = ArticulationAction(
                full_ordered_js_action.position.view(-1).cpu().numpy(),
                # full_ordered_js_action.velocity.cpu().numpy(),
                joint_indices=art_action_idx_list,
            )
            return articulation_action
        
        def goals_dict_W_to_R(self, robot_pose:list[float], goals_W:dict[str, Pose])->dict[str, Pose]:
            """
            convert the goals to the robot's base frame
            """
            F = robot_pose
            goals_R = {}
            for link_name, goal in goals_W.items():
                p_goal_R, q_goal_R = FrameUtils.world_to_F(np.array(F[:3]), np.array(F[3:]), goal.position.flatten().cpu().numpy(), goal.quaternion.flatten().cpu().numpy())
                goals_R[link_name] = Pose(position=self.solver.tensor_args.to_device(torch.from_numpy(p_goal_R)), quaternion=self.solver.tensor_args.to_device(torch.from_numpy(q_goal_R)))
            return goals_R
        
        
        def goals_dict_R_to_W(self, robot_pose:list[float], goals_R:dict[str, Pose])->dict[str, Pose]:
            """
            convert the goals to the world frame
            """
            F = robot_pose
            goals_W = {}
            for link_name, goal in goals_R.items():
                p_goal_W, q_goal_W = FrameUtils.F_to_world(np.array(F[:3]), np.array(F[3:]), np.array(goal.position.flatten().cpu().numpy()), np.array(goal.quaternion.flatten().cpu().numpy()))
                goals_W[link_name] = Pose(position=self.solver.tensor_args.to_device(torch.from_numpy(p_goal_W)), quaternion=self.solver.tensor_args.to_device(torch.from_numpy(q_goal_W)))
            return goals_W

        @abstractmethod
        def get_estimated_plan(self, dof_names:list[str], n_col_spheres_valid:int, joints_state:isaac_JointsState, include_task_space:bool=True, n_steps:int=-1 , valid_spheres_only = True, naive:bool=False):
            pass

        
        def get_col_pred(self,**kwargs)->Optional[DynamicObsCollPredictor]:
            return None
        
        @abstractmethod
        def update_col_pred(self, plans_board, idx, col_pred_with, goal_errors=None):
            pass
        
        def get_col_pred_debug(self)->Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            return None
        
        def get_spheres_from_solkin(self,to_world_frame:bool=True)->torch.Tensor:
            """
            return spheres of robot- positions are in robot frame
            """
            spheres_pr_robot_frame =  deepcopy(self.solver.kinematics.robot_spheres) # Sx4 tensor row i =  [xi,yi,zi,ri] (S is the number of spheres)
            if spheres_pr_robot_frame is None:
                return torch.tensor([])
            if to_world_frame:
                robot_base_pos = self.solver.tensor_args.to_device(torch.tensor(self.base_pose[:3]))
                spheres_pr_robot_frame[:, :3] += robot_base_pos
            return spheres_pr_robot_frame
        
        def get_robot_link_meshes_from_solkin(self):
            # put that here only to note that it exists
            return self.solver.kinematics.get_robot_link_meshes() # Sx4 tensor row i =  [xi,yi,zi,ri] (S is the number of spheres)

        
        def attach_external_obj_from_robot(self, joint_state, external_robot):
            # put that here only to note that it exists
            return self.solver.kinematics.attach_external_objects_to_robot(joint_state, external_robot)
        

        def get_state_in_task_space(self, js:JointState, frame='W',):
            """
            js: JointState object
            frame: 'W' or 'R'
            robot_base_pose: list of 7 elements [x,y,z,qw, qx,qy,qz]
            """
            crm = self.crm
            robot_base_pose = self.base_pose

            js_tensor_2d = js.position # [1,DOF]

            # all the poses from next call are in robot frame:
            p_ee, q_ee, _, _, p_links, q_links, prad = crm.forward(js_tensor_2d) # https://curobo.org/_api/curobo.cuda_robot_model.cuda_robot_model.html#curobo.cuda_robot_model.cuda_robot_model.CudaRobotModelConfig
            
            d = {'ee': {'p': p_ee, 'q': q_ee}, 'links': {'p': p_links, 'q': q_links}, 'spheres': {'p': prad[:,:,:3], 'r': prad[:,:,3]}}
            
            # We first convert the poses to world frame
            if frame == 'W':
                if not hasattr(self, '_cached_base_pose_tensor'):
                    self._cached_base_pose_tensor = torch.tensor(robot_base_pose, device=prad.device, dtype=prad.dtype)
                    from projects_root.utils.transforms import create_optimized_collision_checker_buffers
                    self._cached_rot_mat, self._cached_trans_vec = create_optimized_collision_checker_buffers(1, 1, 1, self._cached_base_pose_tensor, prad.device)
                    self._cached_rot_mat_T = self._cached_rot_mat.T.contiguous()

                for key in d.keys():
                    if 'q' in d[key]:
                        # OPTIMIZED VERSION: Use ultra-fast specialized function
                        X_world = transform_poses_batched_optimized_for_spheres(torch.cat([d[key]['p'], d[key]['q']], dim=-1), self._cached_base_pose_tensor)
                        d[key]['p'] = X_world[...,:3]
                        d[key]['q'] = X_world[...,3:]
                    else:
                        orig_shape = d[key]['p'].shape
                        pKey_flat = d[key]['p'].view(-1, 3)
                        pKey_world_flat = torch.addmm(self._cached_trans_vec, pKey_flat, self._cached_rot_mat_T)
                        d[key]['p'] = pKey_world_flat.view(orig_shape)
                
            # elif frame == 'R2':

            return d    

        def kill_cost_plots(self):
            pass

    class MpcPlanner(CuPlanner):

        def __init__(self, base_pose:list[float], solver_config_dict: dict, robot_cfg:dict, world_cfg:WorldConfig, particle_file_path:str):
                    
            self.solver:MpcSolver # declaring for linter
            _solver_cfg = MpcSolverConfig.load_from_robot_config(
                robot_cfg,
                world_cfg,
                **solver_config_dict
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
            
            
            # _initial_ee_target_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
            # _initial_constrained_links_target_poses = {name: state.link_poses[name] for name in self.constrained_links_names}
            
            initial_goals_R = {self.ee_link_name: Pose(position=state.ee_pos_seq, quaternion=state.ee_quat_seq)}
            for link_name in self.constrained_links_names:
                initial_goals_R[link_name] = state.link_poses[link_name]
            
            initial_goals_W = self.goals_dict_R_to_W(self.base_pose, initial_goals_R)
            self.plan_goals = initial_goals_W
            
            goal_R = Goal(
                current_state=self.current_state,
                goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
                goal_pose=initial_goals_R[self.ee_link_name],
                links_goal_pose={link_name:initial_goals_R[link_name] for link_name in self.constrained_links_names}
            )
            
    
            
            self.solver_goal_buf_R = self.solver.setup_solve_single(goal_R, 1)
            self.solver.update_goal(self.solver_goal_buf_R)
            mpc_result = self.solver.step(self.current_state, max_attempts=2)
            
            # self.crm = CudaRobotModel(CudaRobotModelConfig.from_data_dict(robot_cfg))
            self.particle_cfg_dict = load_yaml(particle_file_path)
            self.particle_file_path = particle_file_path
            self.solver_cfg_dict = solver_config_dict        
            self.dyn_obs_cost_in_cfg = "cost" in self.particle_cfg_dict and "custom" in self.particle_cfg_dict["cost"] and "arm_base" in self.particle_cfg_dict["cost"]["custom"] and "dynamic_obs_cost" in self.particle_cfg_dict["cost"]["custom"]["arm_base"]
            
        def yield_action(self, goals:dict[str, Pose]):

            # if self._outdated_plan_goals(goals):
            # save the new goals for tracking in future if goals changed
            self.plan_goals = goals

            # update the solver goal buffer with the new goals (in robot frame)
            goals_R = self.goals_dict_W_to_R(self.base_pose, goals)
            self.solver_goal_buf_R.goal_pose.copy_(goals_R[self.ee_link_name]) # ee link goal
            for link_name in self.constrained_links_names: # extra links goals
                if link_name in goals_R:
                    self.solver_goal_buf_R.links_goal_pose[link_name] = goals_R[link_name]
            self.solver.update_goal(self.solver_goal_buf_R)
            
            mpc_result = self.solver.step(self.current_state, max_attempts=2)
            
            action = mpc_result.js_action
            self.cmd_state_full = action
            return mpc_result.js_action
        
        def update_state(self, cu_js:JointState):
            if self.cmd_state_full is None:
                self.current_state.copy_(cu_js)
            else:
                current_state_partial = self.cmd_state_full.get_ordered_joint_state(
                    self.solver.rollout_fn.joint_names
                )
                self.current_state.copy_(current_state_partial)
                self.current_state.joint_names = current_state_partial.joint_names

            self.current_state.copy_(cu_js)
        
        

        def convert_action_to_isaac(self, full_js_action:JointState, sim_js_names:list[str], order_finder:Callable)->ArticulationAction:
            """
            A utility function to convert curobo action to isaac sim action (ArticulationAction).
            """
            # get only joint names that are in both:
            art_action_idx_list = []
            common_js_names = []
            for x in sim_js_names:
                if x in full_js_action.joint_names:
                    art_action_idx_list.append(order_finder(x))
                    common_js_names.append(x)

            full_ordered_js_action = full_js_action.get_ordered_joint_state(common_js_names)
            self._cmd_state_full = full_ordered_js_action

            articulation_action = ArticulationAction(
                full_ordered_js_action.position.view(-1).cpu().numpy(),
                # full_ordered_js_action.velocity.cpu().numpy(),
                joint_indices=art_action_idx_list,
            )
            return articulation_action
        

        def get_policy_means(self) -> torch.Tensor:
            """Returning the mean values of the mpc policy ([Horizon x N actuated dofs] torch tensor) .
            Each entry entry i,j is the acceleration command for the joint j at the time step i (i.e. the i-th step in the horizon).
            The accelerations are then applied at every time step, in order to compute the constant velocity of each joint during state and therfore its target position at the end of this command.
            This target position will be sent to the articulation controller.

            Returns:
                policy means: torch.Tensor, [Horizon x N actuated dofs]
            """
            return self.solver.solver.optimizers[0].mean_action.squeeze(0)
        

        
        def get_estimated_plan(self, dof_names:list[str], n_col_spheres_valid:int, joints_state:isaac_JointsState, include_task_space:bool=True, n_steps:int=-1 , valid_spheres_only = True, naive:bool=False) -> Optional[dict]:
            """
            Get the H steps "estimated plan" for the robot of the mpc agent. 
            By "estimated" we mean that mpc doesent really have a plan, so instead we use the mean of the mpc policy to estimate the plan, 
            and then we "reverse" it (from acceleration to velocity to position) to get the estimated "plan" of the robot.

            Args:
                planner (MpcPlanner): the mpc planner
                include_task_space (bool): 
                    if True, return the plan in task space (All positions are in the world frame).
                    Plan includes collision spheres plan (with valid spheres only or not), and the plan of robot links (each for its frame pose).
                    else, return the plan in joint space (returns joint positions).

                n_steps (int): the number of steps in the plan. If -1, return the entire (full available horizon) plan. Defaults to -1.

                valid_spheres_only (bool): if True, return the plan with valid spheres only.

                joints_state (isaac_JointsState): the current joint state of the robot in isaac sim joint state format 
                (for real robot, wrap the joint states of the robot with this isaac_JointsState class).


            Returns:
                torch.Tensor: the H steps plan.
            """
            def _broadcast_first_step_over_horizon(plan:torch.Tensor) -> torch.Tensor:
                """Broadcast a tensor to a given shape."""
                old_shape = plan.shape
                first_step = plan[0]
                return first_step.expand(old_shape) # make the first step repeat over the horizon
            
            def map_nested_tensors(d: dict, fn: Callable[[torch.Tensor], torch.Tensor]) -> dict:
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
        
            pi_mpc_means = self.get_policy_means() # (H x num of joints) accelerations (each action is an acceleration vector)
            device = pi_mpc_means.device

            _state_filter = self.solver.solver.safety_rollout.dynamics_model.state_filter
            control_dt_solver = _state_filter.dt 

            n_dofs = pi_mpc_means.shape[1]
            
            # translate the plan from joint accelerations only to joint velocities and positions 
            # Vectorized PyTorch implementation using cumsum for sequential Euler integration:
            initial_vel = torch.from_numpy(joints_state.velocities[:n_dofs]).to(device, dtype=pi_mpc_means.dtype)
            initial_pos = torch.from_numpy(joints_state.positions[:n_dofs]).to(device, dtype=pi_mpc_means.dtype)
            
            dv = pi_mpc_means * control_dt_solver
            vel = initial_vel.unsqueeze(0) + torch.cumsum(dv, dim=0)
            
            dp = vel * control_dt_solver
            pos = initial_pos.unsqueeze(0) + torch.cumsum(dp, dim=0)

            plan = {'joint_space':
                    {     
                        'acc': pi_mpc_means, 
                        'vel': vel, 
                        'pos': pos 
                    }
                }
            
            if include_task_space: # get plan in task space (robot spheres)            
                # compute forward kinematics
                start_forward_time = time()
                p_eeplan, q_eeplan, _, _, p_linksplan, q_linksplan, prad_spheresPlan = self.crm.forward(self.solver.tensor_args.to_device(plan['joint_space']['pos'])) # https://curobo.org/_api/curobo.cuda_robot_model.cuda_robot_model.html#curobo.cuda_robot_model.cuda_robot_model.CudaRobotModelConfig
                forward_time = time() - start_forward_time
                # print(f'debug: forward kinematics time: {forward_time}')
                
                task_space_plan = {'ee': {'p': p_eeplan, 'q': q_eeplan}, 'links': {'p': p_linksplan, 'q': q_linksplan}, 'spheres': {'p': prad_spheresPlan[:,:,:3], 'r': prad_spheresPlan[:,:,3]}}
                plan['task_space'] = task_space_plan
                
                # remove spheres that are not valid (i.e. negative radius)
                if valid_spheres_only: # removes the (4) extra spheres reserved for simulating a picked object as part of the robot after picked
                    plan['task_space']['spheres']['p'] = plan['task_space']['spheres']['p'][:, :n_col_spheres_valid]
                    plan['task_space']['spheres']['r'] = plan['task_space']['spheres']['r'][:, :n_col_spheres_valid]
                

                if not hasattr(self, '_cached_base_pose_tensor'):
                    p_sample = plan['task_space']['spheres']['p']
                    wp_tensor = torch.tensor(self.base_pose, device=p_sample.device, dtype=p_sample.dtype)
                    self._cached_base_pose_tensor = wp_tensor
                    from projects_root.utils.transforms import create_optimized_collision_checker_buffers
                    self._cached_rot_mat, self._cached_trans_vec = create_optimized_collision_checker_buffers(1, 1, 1, wp_tensor, p_sample.device)
                    self._cached_rot_mat_T = self._cached_rot_mat.T.contiguous()

                # express spheres in world frame using pre-calculated matrix and an optimized single operation (addmm)
                # We skip ee/links intentionally to preserve original semantics
                pKey = plan['task_space']['spheres']['p']
                orig_shape = pKey.shape
                pKey_flat = pKey.view(-1, 3)
                pKey_world_flat = torch.addmm(self._cached_trans_vec, pKey_flat, self._cached_rot_mat_T)
                plan['task_space']['spheres']['p'] = pKey_world_flat.view(orig_shape)

            
                    
                
                # take only the first n_steps
                if not n_steps == -1:        
                    plan = map_nested_tensors(plan, lambda x: x[:n_steps])
            if naive:
                plan = map_nested_tensors(plan, _broadcast_first_step_over_horizon)
            # print(plan)
            return plan 
        
        def get_col_pred_debug(self):
            col_pred = self.get_col_pred()
            debug = col_pred._debug
            p_obs = debug['p_obs'].detach().cpu().numpy() if isinstance(debug['p_obs'], torch.Tensor) else debug['p_obs']
            p_own = debug['p_own'].detach().cpu().numpy() if isinstance(debug['p_own'], torch.Tensor) else debug['p_own']
            r_obs = debug['r_obs'].detach().cpu().numpy() if isinstance(debug['r_obs'], torch.Tensor) else debug['r_obs']
            r_own = debug['r_own'].detach().cpu().numpy() if isinstance(debug['r_own'], torch.Tensor) else debug['r_own']
            return p_obs, p_own, r_obs, r_own

            
        def _filter_joint_state(self, js_state_prev:Union[JointState,None], js_state_new:JointState, filter_coefficients:FilterCoeff):
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
                use the original coefficients, which are saved in js_state_prev.filter_coeff. See example in get_estimated_plan() of the mpc autonomous franka example.

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
        
        def _integrate_acc(self, acceleration: T_DOF,cmd_joint_state: JointState, dt_planning: float) -> JointState:
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
        
        def _get_arm_reacher(self)->ArmReacher:
            try:
                optimizer = self._get_wrap_mpc_optimizer()
                if optimizer is not None and hasattr(optimizer, 'rollout_fn'):
                    return optimizer.rollout_fn
                else:
                    print("Warning: optimizer is None or doesn't have rollout_fn")
                    return None
            except Exception as e:
                print(f"Error getting arm reacher: {e}")
                return None
        
        def get_col_pred(self)->Optional[DynamicObsCollPredictor]:
            for instance in self._get_custom_arm_base_costs().values():
                if isinstance(instance, DynamicObsCost):
                    return instance.col_pred
            raise ValueError("No col_pred found in the rollout_fn")

        
        def kill_cost_plots(self):
            """Aggressively close all matplotlib figures and threads"""
            # First try the standard method
            try:
                arm_reacher = self._get_arm_reacher()
                if arm_reacher is not None and hasattr(arm_reacher, 'kill_live_plot'):
                    arm_reacher.kill_live_plot()
                    print("Called arm_reacher.kill_live_plot()")
                else:
                    print("Warning: arm_reacher is None or doesn't have kill_live_plot method")
            except Exception as e:
                print(f"Error in standard kill_live_plot: {e}")
            
            

        def update_col_pred(self, plans_board, idx, col_pred_with, goal_errors, plans_lock:Optional[Lock]=None):
            
            def _get_rads_own():
                return plans_board[idx]['task_space']['spheres']['r'][0]
            
            def _get_rads_robotj(j):
                return plans_board[j]['task_space']['spheres']['r'][0]
            
            def _get_plan_robotj(j):
                return plans_board[j]['task_space']['spheres']
            
            col_pred = self.get_col_pred()

            if col_pred is None:
                return
            if not self.is_cpred_initialized:
                if plans_lock is not None:
                    with plans_lock:
                        rads_own = _get_rads_own()
                else:
                    rads_own = _get_rads_own()
                col_pred.set_own_rads(rads_own)
                self.is_cpred_initialized = True
            else:
                # Update positions for each robot that this robot collides with
                for j in col_pred_with:
                    if plans_board[j] is None:
                        continue
                    
                    rad_spheres_robotj = _get_rads_robotj(j)
                    col_pred.set_obs_rads_for_robot(j, rad_spheres_robotj) # assuming that they can be changed...
                    H = self.particle_cfg_dict['model']['horizon']
                    
                    plan_robot_j = _get_plan_robotj(j) 
                    plan_robot_j_horizon = plan_robot_j['p'][:H]
                    col_pred.update_robot_spheres(j, plan_robot_j_horizon)
                    
                    if col_pred.prior_rule == 'pose_wta' and goal_errors[idx] is not None:
                        subto_to_err = {}
                        for subto in col_pred.col_with_idx_map.keys(): # all agents colliding with
                            if goal_errors[subto] is not None:
                                p_err_subto, q_err_subto = goal_errors[subto]
                                subto_to_err[subto] = (p_err_subto, q_err_subto)
                        
                        col_pred.pose_wta_conflict_resolution['own_errors'] = goal_errors[idx] 
                        col_pred.pose_wta_conflict_resolution['subto_errors'] = subto_to_err
                    
                    
        def get_rollouts_in_world_frame(self):
            """
            Get visual rollouts transformed to world frame for visualization.
            
            Returns:
                torch.Tensor: Visual rollouts with poses in world frame
            """
            p_visual_rollouts_robotframe = self.solver.get_visual_rollouts()
            q_visual_rollouts_robotframe = torch.empty(p_visual_rollouts_robotframe.shape[:-1] + torch.Size([4]), device=p_visual_rollouts_robotframe.device)
            q_visual_rollouts_robotframe[...,:] = torch.tensor([1,0,0,0], device=p_visual_rollouts_robotframe.device, dtype=p_visual_rollouts_robotframe.dtype) 
            visual_rollouts = torch.cat([p_visual_rollouts_robotframe, q_visual_rollouts_robotframe], dim=-1)                
            visual_rollouts = transform_poses_batched(visual_rollouts, self.base_pose)
            return visual_rollouts
        

            



            # retract_cfg = self.solver.rollout_fn.dynamics_model.get_retract_config().clone().unsqueeze(0)
            # joint_names = self.solver.rollout_fn.joint_names
            # state = self.solver.rollout_fn.compute_kinematics(
            #     JointState.from_position(retract_cfg, joint_names=joint_names)
            # )
            # self.current_state = JointState.from_position(retract_cfg, joint_names=joint_names)
            
            
            # # _initial_ee_target_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
            # # _initial_constrained_links_target_poses = {name: state.link_poses[name] for name in self.constrained_links_names}
            
            # initial_goals_R = {self.ee_link_name: Pose(position=state.ee_pos_seq, quaternion=state.ee_quat_seq)}
            # for link_name in self.constrained_links_names:
            #     initial_goals_R[link_name] = state.link_poses[link_name]
            
            # initial_goals_W = self.goals_dict_R_to_W(self.base_pose, initial_goals_R)
            # robot_context["cur_ee_link_pose"] = self.solver.kinematics.get_link_pose(self.ee_link_name)
            # for link_name in self.constrained_links_names:
            #     robot_context["link_name_to_pose"][link_name] = self.solver.kinematics.get_link_pose(link_name)
            #     robot_context["name_link_to_target"][link_name] = self.solver.kinematics.get_link_target(link_name)
            #     robot_context["target_name_to_pose"][link_name] = self.solver.kinematics.get_link_target(link_name)

    class CumotionPlanner(CuPlanner):
                    
        def __init__(self,
                    base_pose:list[float],
                    motion_gen_config:MotionGenConfig, 
                    plan_config:MotionGenPlanConfig, 
                    warmup_config:dict,
                    robot_cfg:dict
                ):
            """
            Cumotion planning kit. Can accept goals for end effector and optional extra links (e.g. "constrained" links).
            To use with multi arm, pass inputs (robot config, urdf, etc) as in this example: curobo/examples/isaac_sim/multi_arm_reacher.py
            robot config for example: curobo/src/curobo/content/configs/robot/dual_ur10e.yml

            To use with single arm, pass inputs (robot config, urdf, etc) as in this example: curobo/examples/isaac_sim/motion_gen_reacher.py
            robot config for example: curobo/src/curobo/content/configs/robot/ur10e.yml or franka.yml
            """
            solver = MotionGen(motion_gen_config)
            solver_config = motion_gen_config
            ordered_j_names = solver.kinematics.joint_names
            
            super().__init__(base_pose, solver, solver_config, ordered_j_names, robot_cfg)
            self.plan_config = plan_config
            self.warmup_config = warmup_config
            self.solver:MotionGen = self.solver # only for linter 
            print("warming up...")
            self.solver.warmup(**self.warmup_config)
            self.plan = Plan()
            self._set_goals_to_retract_state()
        

                
        def _plan_new(self, 
                    cu_js:JointState,
                    goals:dict[str, Pose],
                    )->bool:
            """
            Making a new plan. return True if success, False otherwise
            """
        
            goals_R = self.goals_dict_W_to_R(self.base_pose, goals)
            ee_goal_R = goals_R[self.ee_link_name]
            extra_links_goals_R = {link_name:goals_R[link_name] for link_name in self.constrained_links_names}
            result = self.solver.plan_single(
                cu_js.unsqueeze(0), ee_goal_R, self.plan_config.clone(), link_poses=extra_links_goals_R
            )
            succ = result.success.item()  # ik_result.success.item()
            if succ:
                print("planned successfully, resetting plan...")
                self.plan.cmd_plan = result.get_interpolated_plan()
                self.plan.cmd_idx = 0
                self.plan_goals = deepcopy(goals)
                return True
            else:
                carb.log_warn("Plan did not converge to a solution: " + str(result.status))
                return False
        
        
        
        def _in_move(self, joint_velocities:np.ndarray):
            """
            check if the joints are in move
            """
            # print(f"joint_velocities={joint_velocities}")
            # print(f"max(abs(joint_velocities))={np.max(np.abs(joint_velocities))}")
            return np.max(np.abs(joint_velocities)) > 0.5
        
        def yield_action(self, goals:dict[str, Pose], cu_js:JointState, joint_velocities:np.ndarray,stop_when_goal_changed:bool=True):
            """
            goals: dict of link names (both end effector link and extra links) and their updated goal poses.
            cu_js: current curobo joint state of the robot.
            joint_velocities: current joint velocities of the robot, as measured by the robot (from simulation, sensors or other sources).
            returns:
                action: Union[ArticulationAction, None]
            """
            
            PLAN_NEW = 0 # REPLAN NEXT ACTION SEQUENCE (JOINT POSITIONS)
            STOP_IN_PLACE = 1 # SEND STOP COMMAND TO JOINT CONTROLLER (VELOCICY 0)
            CONSUME_FROM_PLAN = 2 # CONTINUE THE CURRENT ACTION SEQUENCE
            
            if self._outdated_plan_goals(goals):
                print(f'debug outdated plan goals')
                if self._in_move(joint_velocities):
                    print(f"DEBUG in move, stopping in place...")
                    code = STOP_IN_PLACE if stop_when_goal_changed else CONSUME_FROM_PLAN # STOP_IN_PLACE
                    if code == STOP_IN_PLACE:
                        if not hasattr(self, "_js_at_stop_cmd"):
                            self._js_at_stop_cmd = None  
                        if self._js_at_stop_cmd is None:
                            self._js_at_stop_cmd = deepcopy(cu_js.position)

                    print(f'debug: robot in move. a: {"consume" if code == CONSUME_FROM_PLAN else "stop"}')
                else:
                    print(f"debug: robot stopped. a: plan new")
                    self._js_at_stop_cmd = None # reset the js at stop cmd
                    code = PLAN_NEW
                    
                
            else:
                print(f'valid plan goals, consuming from plan...')
                code = CONSUME_FROM_PLAN
            print(f"DEBUG code: {code}")
            consume = True
            if code == PLAN_NEW:
                print(f'planning...')
                # print(f'debug: goals: {goals}')
                _success = self._plan_new(cu_js, goals)
                
            elif code == STOP_IN_PLACE:
                action = JointState(
                    position=self._js_at_stop_cmd, # cu_js.position,
                    # velocity=cu_js.velocity * 0.0,
                    joint_names=cu_js.joint_names,
                )
                print(f'stopping robot...')
                consume = False

            elif code == CONSUME_FROM_PLAN:
                print(f'consuming current plan...')
                pass
            else:
                raise ValueError(f"Invalid code: {code}")
            
            if consume:
                action = self.plan.consume_action() # returns None if no more actions to consume

            if action is not None:
                action = deepcopy(self.solver.get_full_js(action))
            
            return action
        

        def get_estimated_plan(self, dof_names:list[str], n_col_spheres_valid:int, joints_state:isaac_JointsState, include_task_space:bool=True, n_steps:int=-1 , valid_spheres_only = True, naive:bool=False):
            return None # TODO


    class SimRobot:
        def __init__(self, robot, path:str,visualize_col_spheres:dict,visualize_obj_bound_spheres:dict,visualize_plan:dict,visualize_mpc_ee_rollouts:dict,visualize_col_pred:dict,viz_color:str):
            self.robot = robot
            self.path = path # prim path of the robot in isaac sim
            self.articulation_controller = self.robot.get_articulation_controller()
    
            self._cur_js = None
            self.viz_color = viz_color
            
            # debugging variables
            self.viz_col_spheres_on_world = visualize_col_spheres['is_on_world']
            self.viz_col_spheres_on_robot = visualize_col_spheres['is_on_robot']
            self.viz_col_spheres_dt = visualize_col_spheres['ts_delta']
            self.viz_col_spheres_color = self.viz_color if visualize_col_spheres['color'] == 'viz_color' else visualize_col_spheres['color']

            self.viz_obj_bound_spheres_on = visualize_obj_bound_spheres['is_on']
            self.viz_obj_bound_spheres_dt = visualize_obj_bound_spheres['ts_delta']

            self.viz_plan_on = visualize_plan['is_on']
            self.viz_plan_dt = visualize_plan['ts_delta']
            self.viz_plan_color = self.viz_color if visualize_plan['color'] == 'viz_color' else visualize_plan['color']
            
            self.viz_mpc_ee_rollouts_on = visualize_mpc_ee_rollouts['is_on']
            self.viz_mpc_ee_rollouts_dt = visualize_mpc_ee_rollouts['ts_delta']
            self.viz_mpc_ee_rollouts_color =  visualize_mpc_ee_rollouts['color']
            
            self.viz_col_pred_dt = visualize_col_pred['ts_delta']
            self.viz_col_pred_own_on = visualize_col_pred['own']
            self.viz_col_pred_own_mean_only = visualize_col_pred['own_mean_only']
            self.viz_col_pred_own_color = visualize_col_pred['own_color']
            self.viz_col_pred_obs_on = visualize_col_pred['obs']
            self.viz_col_pred_obs_color = visualize_col_pred['obs_color']
            

        
        def get_js(self, sync_new=True) -> isaac_JointsState:
            if sync_new:
                js = self.robot.get_joints_state() # from simulation. TODO: change "robot" to sim_robot and add a real_robot attribute
                self._cur_js = js # update the last synced state

            else:
                js = self._cur_js # return the last synced state
            return js
        
    
            
        
        def update_robot_sim_spheres(self, subroot:str, visible:bool,a_idx:int, spheres_tensor:torch.Tensor,in_col:set[int]=set()):
            # if cu_js is None:
            #     return
            
            # spheres = solver.kinematics.get_robot_as_spheres(cu_js.position)[0]
            green = np.array([0, 0.8, 0.2])
            red = np.array([0.8, 0, 0])

            if not hasattr(self, "_vis_spheres"): # init visualization spheres
                self._vis_spheres = []
                for si, s in enumerate(spheres_tensor):
                    
                    sp = sphere.VisualSphere(
                        prim_path=f"{subroot}/R{a_idx}S{si}",
                        position=np.ravel(s[:3].cpu().numpy()),
                        orientation=np.array([1,0,0,0]),# np.ravel(s[3:7].cpu().numpy()),
                        # radius=float(s[7].cpu().item()),
                        radius=float(s[3].cpu().item()),
                        color=green,
                    )
                    self._vis_spheres.append(sp)
                    if not visible:
                        sp.set_visibility(False)

            else: # update visualization spheres
                for si, s in enumerate(spheres_tensor):
                    if si in in_col:
                        color = red
                    else:
                        color = green 
                    
                    self._vis_spheres[si].get_applied_visual_material().set_color(color)                           

                    if not np.isnan(s[0].item()):
                        self._vis_spheres[si].set_world_pose(
                            position=np.ravel(s[:3].cpu().numpy()),
                            orientation=np.array([1,0,0,0]),# np.ravel(s[3:7].cpu().numpy()),
                            )
                        # self._vis_spheres[si].set_radius(float(s[7].cpu().item()))
                        self._vis_spheres[si].set_radius(float(s[3].cpu().item()))
        @staticmethod
        def parse_viz_color(color:str)->list[float]:
            if color == 'green':
                return [0, 1, 0]
            elif color == 'red':
                return [1, 0, 0]
            elif color == 'blue':
                return [0, 0, 1]
            elif color == 'yellow':
                return [1, 1, 0]
            elif color == 'purple':
                return [1, 0, 1]
            elif color == 'orange':
                return [1, 0.5, 0]
            else:
                raise ValueError(f"Invalid color: {color}")


    # class CumotionPlanPublisher(PlanPublisher):
    #     pass

    def get_per_axis_euler_error(q1:list[float], q2:list[float])->float:
        """
        Calculate the rotation error between two quaternions (each wxyz).
        """
        euler1 = _get_scipy_rotation().from_quat(q1).as_euler('xyz', degrees=True)
        euler2 = _get_scipy_rotation().from_quat(q2).as_euler('xyz', degrees=True)
        euler_error = euler2 - euler1
        # Normalize angle to [-180, 180]
        euler_error = (euler_error + 180) % 360 - 180
        # print("Euler error (deg):", euler_error)
        return euler_error



    def publish_robot_context(robot_idx:int, robot_context:dict,robot_pose:list, n_obstacle_spheres:int, robot_sphere_count:int, mpc_cfg:dict, col_pred_with_robot:List[int], mpc_config_paths:List[str], robot_config_paths:List[str], robot_sphere_counts_split:List[Tuple[int, int]]):
        """
        Publish robot context ("topics") to the environment topics.
        TODO: Re-design this whole approach, give better names to the variables ()
        """

        # Populate robot context directly in env_topics[i]
        robot_context["env_id"] = 0
        robot_context["robot_id"] = robot_idx
        robot_context["robot_pose"] = robot_pose
        robot_context["n_obstacle_spheres"] = n_obstacle_spheres
        
        robot_context["n_own_spheres"] = robot_sphere_count
        robot_context["horizon"] = mpc_cfg["model"]["horizon"]
        robot_context["n_rollouts"] = mpc_cfg["mppi"]["num_particles"]
        robot_context["col_pred_with"] = col_pred_with_robot
        
        # Add new fields for sparse sphere functionality
        robot_context["mpc_config_paths"] = mpc_config_paths
        robot_context["robot_config_paths"] = robot_config_paths
        robot_context["robot_sphere_counts"] = robot_sphere_counts_split  # [(base, extra), ...]
        robot_context["link_name_to_pose"] = {}
        robot_context["name_link_to_target"] = {}
        robot_context["target_name_to_pose"] = {}

    def publish_to_context(robot_context:dict, key:str, value):
        robot_context[key] = value
        
    def calculate_robot_sphere_count(robot_cfg):
        """
        Calculate the number of collision spheres for a robot from its configuration.
        
        Args:
            robot_cfg: Robot configuration dictionary
            
        Returns:
            int: Total number of collision spheres (base + extra)
        """
        robots_collision_spheres_configs_parent_dir = "curobo/src/curobo/content/configs/robot"

        # Get collision spheres configuration
        collision_spheres = robot_cfg["kinematics"]["collision_spheres"]
        
        # Handle two cases:
        # 1. collision_spheres is a string path to external file (e.g., Franka)
        # 2. collision_spheres is inline dictionary (e.g., UR5e)
        if isinstance(collision_spheres, str):
            # External file case  
            collision_spheres_cfg = load_yaml(os.path.join(robots_collision_spheres_configs_parent_dir, collision_spheres))
            collision_spheres_dict = collision_spheres_cfg["collision_spheres"]
        else:
            # Inline dictionary case
            collision_spheres_dict = collision_spheres
        
        # Count spheres by counting entries in each link's sphere list
        sphere_count = 0
        for link_name, spheres in collision_spheres_dict.items():
            if isinstance(spheres, list):
                sphere_count += len(spheres)
        
        # Add extra collision spheres
        extra_spheres = robot_cfg["kinematics"].get("extra_collision_spheres", {})
        extra_sphere_count = 0
        for obj_name, count in extra_spheres.items():
            extra_sphere_count += count
            
        return sphere_count, extra_sphere_count

    class CuAgent:
    

        def __init__(self, 
                    idx:int,
                    tensor_args:TensorDeviceType,
                    planner:CuPlanner,
                    cu_world_wrapper_cfg:dict,
                    robot_cfg_path:str,
                    robot_cfg:dict,
                    sim_robot:Optional[SimRobot]=None,
                    plan_pub_sub:Optional[PlanPubSub]=None,
                    viz_color:str='orange',
                    stat_man_cfg:dict={},
                    world:Optional[World]=None,
                    # is_mobile:bool=False,
                    # mobile_base_link_subpath:str='',
                    ):
            
            self.idx = idx
            self.step_count = 0 # num of completed control iterations where agent was able to sense->pub->plan->execute an action
            self.planning_stopwatch = Stopwatch() # total planning time (in seconds) only planning time (no simulation related ops time)
            self.tensor_args = tensor_args
            self.planner = planner        
            self.base_pose = planner.base_pose
            self.sim_robot = sim_robot
            self.robot_cfg_path = robot_cfg_path
            self.robot_cfg = robot_cfg
            self.plan_pub_sub = plan_pub_sub
            self.viz_color = SimRobot.parse_viz_color(viz_color)
            self.stat_man:StatManager
            if world is not None:
                self.stat_man = StatManager(world, **stat_man_cfg) 
            

            # self.is_mobile = is_mobile
            # self.mobile_base_link_subpath = mobile_base_link_subpath

            # See wrapper's docstring to understand the motivation for the wrapper.
            _solver_wm = self.planner.solver.world_coll_checker.world_model
            assert isinstance(_solver_wm, WorldConfig) # only for linter
            self.cu_world_wrapper = WorldModelWrapper(
                world_config=_solver_wm,
                X_robot_W=np.array(self.base_pose), # robot base frame in world frame
                verbosity=cu_world_wrapper_cfg["verbosity"] if "verbosity" in cu_world_wrapper_cfg else 0
            )
            self.cu_world_wrapper_update_policy = {
                'never_add':cu_world_wrapper_cfg["never_add"], 
                'never_update':cu_world_wrapper_cfg["never_update"]
            }
            
            # async worker plumbing (queue-based, lock-free)
            self._worker_thread: Optional[Thread] = None
            self._worker_stop_event = Event()
            self._state_queue: "Queue[tuple[JointState, dict[str, Pose], Optional[np.ndarray]]]" = Queue(maxsize=1)
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

        def post_frame_state(self, cu_js:JointState, goals:dict[str, Pose], joint_velocities:Optional[np.ndarray]=None):
            # keep only latest
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
                    # update planner state (pure compute)
                    if isinstance(self.planner, MpcPlanner):
                        self.planner.update_state(cu_js)
                    # plan
                    if isinstance(self.planner, CumotionPlanner):
                        action = self.planner.yield_action(goals, cu_js, js_vel)
                    elif isinstance(self.planner, MpcPlanner):
                        action = self.planner.yield_action(goals)
                    else:
                        action = None
                    # store latest action (overwrite old)
                    while not self._action_queue.empty():
                        try:
                            self._action_queue.get_nowait()
                        except Empty:
                            break
                    self._action_queue.put_nowait(action)
                except Exception:
                    # swallow to keep worker alive
                    pass
            
        def reset_col_model_from_isaac_sim(self, usd_help:UsdHelper, robot_prim_path:str, ignore_substrings:List[str]):

            # Get world config from simulation
            isaac_cu_world_R = usd_help.get_obstacles_from_stage( 
                only_paths=['/World'], # look for obs only under the world prim path
                reference_prim_path=robot_prim_path, # obstacles are expressed in robot frame! (not world frame). That's why we marked it with 'R'
                ignore_substring=ignore_substrings
            )
            return self.reset_col_model(isaac_cu_world_R)

        def update_col_model_from_isaac_sim(self, 
                robot_prim_path:str,
                usd_help:UsdHelper, 
                ignore_list:List[str]=['/World/defaultGroundPlane'],
                paths_to_search_obs_under:List[str]=['/World'],
                sim_lock:Optional[Lock]=None,
                ):
            """
            Sensing obs from simulation and updating the world model
            """
            def _get_pose_dict_from_sim():
                pose_dict = get_stage_poses(
                usd_helper=usd_help,
                only_paths=paths_to_search_obs_under,
                reference_prim_path='/World', # poses are expressed in world frame
                ignore_substring=ignore_list,
            )
                return pose_dict
            
            def _get_world_cfg_from_stage():
                new_world_cfg:WorldConfig = usd_help.get_obstacles_from_stage(
                    only_paths=list(new_paths),
                    reference_prim_path=robot_prim_path,
                    ignore_substring=ignore_list,
                )
                return new_world_cfg

            # get poses of all obstacles in the world (in world frame)


            if sim_lock is not None:
                with sim_lock:
                    pose_dict = _get_pose_dict_from_sim()
            else:
                pose_dict = _get_pose_dict_from_sim()
            # Fast pose update (only pose updates, no world-model re-initialization as before)
            
            self.cu_world_wrapper.update_from_pose_dict(pose_dict) 
            current_paths = set(
                list_relevant_prims(usd_help, paths_to_search_obs_under, ignore_list)
            )
            new_paths = current_paths - self.cu_world_wrapper.get_known_prims()
            
            
            if new_paths: # Here we are being LAZY! we call expensive get_obstacled_from_stage() only if there are new obstacles!
                # print(f"[NEW OBSTACLES] {new_paths}")
                if sim_lock is not None:
                    with sim_lock:
                        new_world_cfg = _get_world_cfg_from_stage()
                else:
                    new_world_cfg = _get_world_cfg_from_stage()
                
                # Convert to collision check world
                new_world_cfg = new_world_cfg.get_collision_check_world()
                
                if new_world_cfg.objects:  # add only if we got actual obstacles
                    self.cu_world_wrapper.add_new_obstacles_from_cu_world(
                        cu_world_R=new_world_cfg,
                        silent=False,
                    )
                    # Track real obstacle names so future pose updates work
                    for obj in new_world_cfg.objects:
                        self.cu_world_wrapper.add_prim_to_known(obj.name)
        def reset_col_model(self, cu_world_R:WorldConfig):
            """
            Reset the collision model (with obstacles) from an input world config.
            To use not in simulation, pass your custom world config from real world.
            To use in isaac sim, see reset_col_model_from_isaac_sim()
            Args:
                cu_world_R: WorldConfig object that contains the current obstacle poses.
            """
            
            # Convert raw WorldConfig to collision check world WorldConfig! (Must!)
            cu_col_world_R = self.cu_world_wrapper.initialize_from_cu_world(cu_world_R)        
            # Update MPC world collision checker with the initialized world
            assert self.planner.solver.world_coll_checker is not None # only for linter (it's not None) 
            self.planner.solver.world_coll_checker.load_collision_model(cu_col_world_R) 
            
            # Set the collision checker reference in the wrapper
            self.cu_world_wrapper.set_collision_checker(self.planner.solver.world_coll_checker)
            
            # Record the prims that are currently considered obstacles (for easy lookup later when checking if update is needed)
            self.cu_world_wrapper.set_known_prims()

            print("WorldModelWrapper reset finished successfully!")
            print(f"Known prims in collision world: {self.cu_world_wrapper.get_known_prims()}")        

        def get_spheres_from_solkin(self,to_world_frame:bool=True)->torch.Tensor:
            return self.planner.get_spheres_from_solkin(to_world_frame)

        def update_col_pred(self, plans_board, mean_goal_err, plans_lock:Optional[Lock]=None):
            sub_to = self.plan_pub_sub.sub_cfg["to"] if isinstance(self.planner, MpcPlanner) else []    
            self.planner.update_col_pred(plans_board, self.idx, sub_to, mean_goal_err)
        
        def get_sphere_tensor_W(self, cu_js:JointState)->torch.Tensor:
            task_space_state_W = self.planner.get_state_in_task_space(cu_js, frame='W')                    
            spheres_W = task_space_state_W['spheres']
            #p_W, q_W, r_W = spheres_W['p'], spheres_W['q'], spheres_W['r']
            p_W, r_W = spheres_W['p'], spheres_W['r']
            sphere_tensor_W = torch.cat((p_W.squeeze(0), r_W.T),dim=1)
            return sphere_tensor_W
        
        def split_sphere_tensor_W_into_arms(self, sphere_tensor_W:torch.Tensor, n_arms:int)->list[torch.Tensor]:
            ans = []
            spheres_per_arm = sphere_tensor_W.shape[0] // n_arms
            for i in range(n_arms):
                ans.append(sphere_tensor_W[i*spheres_per_arm:(i+1)*spheres_per_arm])
            return ans

        def async_control_loop_sim(self, t_lock, sim_lock, plans_lock, goals_lock, debug_lock, stop_event, plans_board, get_t, pts_debug, usd_help:UsdHelper,
                                goals:Dict[str, Pose],sim_env:SimEnv,sim_task:SimTask):
            self._last_t = -1
            psw = self.planning_stopwatch
            
            while not stop_event.is_set():
                
                if self.sim_robot is not None:    

                    ctrl_dof_names = self.sim_robot.robot.dof_names
                    ctrl_dof_indices = self.sim_robot.robot.get_dof_index
                    
                    # get current global step index
                    with t_lock:
                        t = get_t()
                    if t == self._last_t:
                        sleep(1e-7)
                        continue

                    # Optional visualization flags
                    viz_plans, viz_plans_dt = self.sim_robot.viz_plan_on, self.sim_robot.viz_plan_dt
                    viz_cpred_dt = self.sim_robot.viz_col_pred_dt
                    viz_cpred_own = self.sim_robot.viz_col_pred_own_on
                    viz_cpred_obs = self.sim_robot.viz_col_pred_obs_on
                    viz_mpc_ee_rollouts, viz_mpc_ee_rollouts_dt = self.sim_robot.viz_mpc_ee_rollouts_on, self.sim_robot.viz_mpc_ee_rollouts_dt

                    # 1) Publish last-step estimated plan (if configured)
                    js = self.sim_robot.get_js(sync_new=False)
                    plan = None
                    if js is not None and self.plan_pub_sub is not None and self.plan_pub_sub.should_pub_now(t):
                        share_full_plan = self.is_full_plan_publisher()
                        psw.on()
                        try:
                            plan = self.planner.get_estimated_plan(
                                ctrl_dof_names,
                                self.plan_pub_sub.valid_spheres,
                                js,
                                valid_spheres_only=False,
                                naive=not share_full_plan,
                            )
                        finally:
                            psw.off()

                    if plan is not None:
                        plans_board[self.idx] = plan
                        if viz_plans and (t % viz_plans_dt == 0):
                            pts_debug.append({'points': plan['task_space']['spheres']['p'], 'color': self.sim_robot.viz_plan_color})

                    # 2) Sense obstacles and joints atomically w.r.t. simulator
                    with sim_lock:
                        self.update_col_model_from_isaac_sim(
                            self.sim_robot.path,
                            usd_help,
                            ignore_list=self.cu_world_wrapper_update_policy["never_add"] + self.cu_world_wrapper_update_policy["never_update"],
                            paths_to_search_obs_under=["/World"],
                        )
                        js = self.sim_robot.get_js(sync_new=True)
                        if js is None:
                            print("sim_js is None")
                            self._last_t = t
                            continue

                    # 3) Prepare current curobo state and update planner
                    _0 = self.tensor_args.to_device(js.positions) * 0.0
                    cu_js = JointState(
                        self.tensor_args.to_device(js.positions),
                        self.tensor_args.to_device(js.velocities),
                        _0,
                        ctrl_dof_names,
                        _0,
                    ).get_ordered_joint_state(self.planner.ordered_j_names)

                    if isinstance(self.planner, MpcPlanner):
                        self.planner.update_state(cu_js)

                    # Optional: build data for collision metric/stats (used by rec/spheres if needed)
                    task_space_state_R = self.planner.get_state_in_task_space(cu_js, frame='R')
                    spheres_R = task_space_state_R['spheres']
                    p_R, r_R = spheres_R['p'], spheres_R['r']
                    pr_R = torch.cat((p_R.squeeze(0), r_R.T), dim=1)

                    # 4) Update collision predictor from plans if subscribing
                    if self.is_plan_subscriber():
                        self.update_col_pred(plans_board)

                    # 5) Read latest goals snapshot and update robot context
                    with goals_lock:
                        current_goals = {ln: goals[ln] for ln in list(goals.keys())}
                        robot_context = get_topics().get_default_env()[self.idx]
                        robot_context["link_name_to_pose"] = sim_task.get_link_name_to_pose()[self.idx]
                        robot_context["name_link_to_target"] = sim_task.name_link_to_target[self.idx]
                        robot_context["target_name_to_pose"] = sim_task.get_target_name_to_pose()[self.idx]

                    # 6) Plan next action
                    psw.on()
                    try:
                        if isinstance(self.planner, CumotionPlanner):
                            action = self.planner.yield_action(current_goals, cu_js, js.velocities)
                        elif isinstance(self.planner, MpcPlanner):
                            action = self.planner.yield_action(current_goals)
                            if viz_mpc_ee_rollouts and (t % viz_mpc_ee_rollouts_dt == 0):
                                with debug_lock:
                                    pts_debug.append({'points': self.planner.get_rollouts_in_world_frame(), 'color': self.sim_robot.viz_mpc_ee_rollouts_color})
                        else:
                            raise ValueError(f"Invalid planner type: {type(self.planner).__name__}")
                    finally:
                        psw.off()

                    # 7) Act - apply articulation action atomically w.r.t. simulator
                    if action is not None:
                        with sim_lock:
                            isaac_action = self.planner.convert_action_to_isaac(action, ctrl_dof_names, ctrl_dof_indices)
                            self.sim_robot.articulation_controller.apply_action(isaac_action)
                        self.step_count += 1

                    # 8) Optional debug: visualize collision predictor data if publishing and multi-robot
                    if (viz_cpred_own or viz_cpred_obs) and (t % viz_cpred_dt == 0):
                        if self.is_full_plan_publisher():
                            debug_data = self.planner.get_col_pred_debug()
                            if debug_data is not None:
                                p_obs, p_own, r_obs, r_own = debug_data
                                with debug_lock:
                                    if self.sim_robot.viz_col_pred_own_mean_only:
                                        p_own = p_own.mean(axis=0)
                                    if viz_cpred_own:
                                        pts_debug.append({'points': p_own, 'color': self.sim_robot.viz_col_pred_own_color})
                                    if viz_cpred_obs:
                                        pts_debug.append({'points': p_obs, 'color': self.sim_robot.viz_col_pred_obs_color})

                    # mark this iteration's step index
                    self._last_t = t

                # small yield to avoid hogging CPU if no work
                sleep(1e-7)


        async def async_control_loop_sim_kit(self, get_t, stop_event, plans_board, usd_help:UsdHelper,
                                            goals:Dict[str, Pose], sim_env:SimEnv, sim_task:SimTask, pts_debug=None):
            """Async per-agent loop paced by Kit's runloop without locks.
            Mirrors Block B logic but runs on the main thread using next_update_async().
            """
            self._last_t = -1
            psw = self.planning_stopwatch
            try:
                import omni
            except Exception:
                omni = None
            app = omni.kit.app.get_app() if omni is not None else None

            while not stop_event.is_set():
                if app is not None:
                    await app.next_update_async()

                if self.sim_robot is None:
                    continue

                t = get_t()
                if t == self._last_t:
                    continue

                ctrl_dof_names = self.sim_robot.robot.dof_names
                ctrl_dof_indices = self.sim_robot.robot.get_dof_index

                viz_plans, viz_plans_dt = self.sim_robot.viz_plan_on, self.sim_robot.viz_plan_dt
                viz_cpred_dt = self.sim_robot.viz_col_pred_dt
                viz_cpred_own = self.sim_robot.viz_col_pred_own_on
                viz_cpred_obs = self.sim_robot.viz_col_pred_obs_on
                viz_mpc_ee_rollouts, viz_mpc_ee_rollouts_dt = self.sim_robot.viz_mpc_ee_rollouts_on, self.sim_robot.viz_mpc_ee_rollouts_dt

                # 1) Publish last-step estimated plan (if configured)
                js = self.sim_robot.get_js(sync_new=False)
                plan = None
                if js is not None and self.plan_pub_sub is not None and self.plan_pub_sub.should_pub_now(t):
                    share_full_plan = self.is_full_plan_publisher()
                    psw.on()
                    try:
                        plan = self.planner.get_estimated_plan(
                            ctrl_dof_names,
                            self.plan_pub_sub.valid_spheres,
                            js,
                            valid_spheres_only=False,
                            naive=not share_full_plan,
                        )
                    finally:
                        psw.off()

                if plan is not None:
                    plans_board[self.idx] = plan
                    if viz_plans and (t % viz_plans_dt == 0) and pts_debug is not None:
                        try:
                            pts_debug.append({'points': plan['task_space']['spheres']['p'], 'color': self.sim_robot.viz_plan_color})
                        except Exception:
                            pass

                # 2) Sense obstacles and joints (single-threaded via Kit runloop)
                self.update_col_model_from_isaac_sim(
                    self.sim_robot.path, 
                    usd_help, 
                    ignore_list=self.cu_world_wrapper_update_policy["never_add"] + self.cu_world_wrapper_update_policy["never_update"], 
                    paths_to_search_obs_under=["/World"],
                )
                js = self.sim_robot.get_js(sync_new=True) 
                if js is None:
                    self._last_t = t
                    continue
                        
                # 3) Prepare current curobo state and update planner
                _0 = self.tensor_args.to_device(js.positions) * 0.0
                cu_js = JointState(
                    self.tensor_args.to_device(js.positions),
                    self.tensor_args.to_device(js.velocities),
                    _0,
                    ctrl_dof_names,
                    _0,
                ).get_ordered_joint_state(self.planner.ordered_joint_names)
                        
                if isinstance(self.planner, MpcPlanner):
                    self.planner.update_state(cu_js)
                        
                # Optional: build data for collision metric/stats
                task_space_state_R = self.planner.get_state_in_task_space(cu_js, frame='R')
                spheres_R = task_space_state_R['spheres']
                p_R, r_R = spheres_R['p'], spheres_R['r']
                pr_R = torch.cat((p_R.squeeze(0), r_R.T), dim=1)
                        
                # 4) Update collision predictor from plans if subscribing
                if self.is_plan_subscriber():
                    self.update_col_pred(plans_board)
                                        
                # 5) Read latest goals snapshot and update robot context
                current_goals = {ln: goals[ln] for ln in list(goals.keys())}
                robot_context = get_topics().get_default_env()[self.idx]
                robot_context["link_name_to_pose"] = sim_task.get_link_name_to_pose()[self.idx]
                robot_context["name_link_to_target"] = sim_task.name_link_to_target[self.idx]
                robot_context["target_name_to_pose"] = sim_task.get_target_name_to_pose()[self.idx]

                # 6) Plan next action
                psw.on()
                try:
                    if isinstance(self.planner, CumotionPlanner):
                        action = self.planner.yield_action(current_goals, cu_js, js.velocities)
                    elif isinstance(self.planner, MpcPlanner):
                        action = self.planner.yield_action(current_goals)
                        if viz_mpc_ee_rollouts and (t % viz_mpc_ee_rollouts_dt == 0) and pts_debug is not None:
                            try:
                                pts_debug.append({'points': self.planner.get_rollouts_in_world_frame(), 'color': self.sim_robot.viz_mpc_ee_rollouts_color})
                            except Exception:
                                pass
                    else:
                        raise ValueError(f"Invalid planner type: {type(self.planner).__name__}")
                finally:
                    psw.off()
                        
                # 7) Act - apply articulation action (single-threaded via Kit runloop)
                if action is not None:
                    isaac_action = self.planner.convert_action_to_isaac(action, ctrl_dof_names, ctrl_dof_indices)
                    self.sim_robot.articulation_controller.apply_action(isaac_action)
                    self.step_count += 1

                # 8) Optional debug: visualize collision predictor data if publishing and multi-robot
                if (viz_cpred_own or viz_cpred_obs) and (t % viz_cpred_dt == 0) and pts_debug is not None:
                    if self.is_full_plan_publisher():
                        debug_data = self.planner.get_col_pred_debug()
                        if debug_data is not None:
                            p_obs, p_own, r_obs, r_own = debug_data
                            try:
                                if self.sim_robot.viz_col_pred_own_mean_only:
                                    p_own = p_own.mean(axis=0)
                                if viz_cpred_own:
                                    pts_debug.append({'points': p_own, 'color': self.sim_robot.viz_col_pred_own_color})
                                if viz_cpred_obs:
                                    pts_debug.append({'points': p_obs, 'color': self.sim_robot.viz_col_pred_obs_color})
                            except Exception:
                                pass

                self._last_t = t
            
        def async_control_step_sim(self,sim_lock,plans_lock,plans_board,goals,sim_task,usd_help):
            """
            like async_control_loop_sim, but only for one step
            """
            
            # sw = self.stats.sim_watch
            # pw = self.stats.plan_watch

            if self.sim_robot is None:
                return
            
            # sw.on()
            ctrl_dof_names = self.sim_robot.robot.dof_names # or from real
            ctrl_dof_indices = self.sim_robot.robot.get_dof_index # or from real
            js = self.sim_robot.get_js(sync_new=False) # get last step's joint state
        
            plan = None
            if js is not None: #and self.plan_pub_sub.should_pub_now(t):
                share_full_plan = self.is_full_plan_publisher() # naive means broadcase state as plan over horizon
                plan = self.planner.get_estimated_plan(ctrl_dof_names, self.plan_pub_sub.valid_spheres, js, valid_spheres_only=False, naive=not share_full_plan) # get last step's plan (naive <=> broadcast current pose as plan (not future steps))                    
            if plan is not None: # currently available in mpc only
                with plans_lock:
                    plans_board[self.idx] = plan


            self.update_col_model_from_isaac_sim(
                    self.sim_robot.path, 
                    usd_help, 
                    ignore_list=self.cu_world_wrapper_update_policy["never_add"] + self.cu_world_wrapper_update_policy["never_update"], 
                    paths_to_search_obs_under=["/World"]
                )

            js = self.sim_robot.get_js(sync_new=True) 
            if js is None:
                print("sim_js is None")
                return

            _0 = self.tensor_args.to_device(js.positions) * 0.0
            cu_js = JointState(self.tensor_args.to_device(js.positions),self.tensor_args.to_device(js.velocities), _0, ctrl_dof_names,_0).get_ordered_joint_state(self.planner.ordered_j_names)
            if isinstance(self.planner, MpcPlanner):
                self.planner.update_state(cu_js)
            
                # sense plans
                if self.is_plan_subscriber():
                    with plans_lock:
                        self.update_col_pred(plans_board)
        
            # # sense goals
            # goals = link_name_to_target_pose[self.idx]

            # plan
            
            # update robot context with current poses and target poses (for dynamic obs cost)
            if self.is_plan_subscriber(): 
                robot_context = get_topics().get_default_env()[self.idx]
                robot_context["link_name_to_pose"] = sim_task.get_link_name_to_pose()[self.idx]
                robot_context["name_link_to_target"] = sim_task.name_link_to_target[self.idx]
                robot_context["target_name_to_pose"] = sim_task.get_target_name_to_pose()[self.idx]

            # yield action
            if isinstance(self.planner, CumotionPlanner):
                action = self.planner.yield_action(goals, cu_js, js.velocities)
            elif isinstance(self.planner, MpcPlanner):
                action = self.planner.yield_action(goals)
                # if viz_mpc_ee_rollouts and t % viz_mpc_ee_rollouts_dt == 0:
                #     pts_debug[self.idx].append({'points': planner.get_rollouts_in_world_frame(), 'color': self.sim_robot.viz_mpc_ee_rollouts_color})
            else:
                raise ValueError(f"Invalid planner type")
            
            if action is not None:
                with sim_lock:
                    isaac_action = self.planner.convert_action_to_isaac(action, ctrl_dof_names, ctrl_dof_indices)
                    self.sim_robot.articulation_controller.apply_action(isaac_action)
            

        def is_plan_subscriber(self)->bool:
            return self.plan_pub_sub is not None and self.plan_pub_sub.sub_cfg["is_on"]

        def is_full_plan_publisher(self):
            return self.plan_pub_sub is not None and self.plan_pub_sub.pub_cfg["is_on"]

            

        @staticmethod
        def check_collisions_between_agents(agents_spheres:list[torch.Tensor])->list[list[list[tuple[int,int]]]]:
            """
            In - list of length N (num of agents) where each element is a tensor of shape Six8 (Si (num of spheres aegent i) x 8 (p, q, r)) 
            Out - list of length N where each element is a matrix in shape NxN (num of agents ^ 2) where out[i][j] = list of tuples (k,l) if agent i sphere k is in collision with agent j sphere l
            """
            ans = [[[] for _ in range(len(agents_spheres))] for _ in range(len(agents_spheres))]
            for i in range(len(agents_spheres)):
                if len(agents_spheres[i]) == 0:
                    continue
                for j in range(i+1, len(agents_spheres)):
                    if len(agents_spheres[j]) == 0:
                        continue
                    # sphere_tensor_i = agents_spheres[i]
                    # sphere_tensor_j = agents_spheres[j]
                    # # check if spheres are in collision
                    # sphere_tensor_i_expanded = sphere_tensor_i.unsqueeze(0).expand(sphere_tensor_j.shape[0], -1, -1)
                    # sphere_tensor_j_expanded = sphere_tensor_j.unsqueeze(1).expand(-1, sphere_tensor_i.shape[0], -1)
                    # sphere_centers_dist = torch.norm(sphere_tensor_i_expanded[:,:,:3] - sphere_tensor_j_expanded[:,:,:3], dim=2)
                    # sphere_radii_sum = sphere_tensor_i_expanded[:,:,3] + sphere_tensor_j_expanded[:,:,3]
                    # in_collision = sphere_centers_dist <= sphere_radii_sum
                    # collision_indices_agent_i  = torch.where(in_collision)[0]
                    # collision_indices_agent_j  = torch.where(in_collision)[1]
                    ans[i][j] = CuAgent.agent_to_agent_colcheck(agents_spheres[i], agents_spheres[j]) # [(k,l) for k,l in zip(collision_indices_agent_i.tolist(), collision_indices_agent_j.tolist())]
            return ans

        @staticmethod
        def agent_to_agent_colcheck(ai:torch.Tensor, aj:torch.Tensor):
            if len(ai) == 0 or len(aj) == 0:
                return []
            sphere_tensor_i = ai
            sphere_tensor_j = aj
            # check if spheres are in collision
            sphere_tensor_i_expanded = sphere_tensor_i.unsqueeze(0).expand(sphere_tensor_j.shape[0], -1, -1)
            sphere_tensor_j_expanded = sphere_tensor_j.unsqueeze(1).expand(-1, sphere_tensor_i.shape[0], -1)
            sphere_centers_dist = torch.norm(sphere_tensor_i_expanded[:,:,:3] - sphere_tensor_j_expanded[:,:,:3], dim=2)
            sphere_radii_sum = sphere_tensor_i_expanded[:,:,3] + sphere_tensor_j_expanded[:,:,3]
            in_collision = sphere_centers_dist <= sphere_radii_sum
            collision_indices_agent_i  = torch.where(in_collision)[0]
            collision_indices_agent_j  = torch.where(in_collision)[1]
            return [(k,l) for k,l in zip(collision_indices_agent_i.tolist(), collision_indices_agent_j.tolist())]
        
        def check_col_with_others(self,agents_spheres:list[torch.Tensor]):
            own_spheres = agents_spheres[self.idx]
            ans = []
            for j in range(len(agents_spheres)):
                if j == self.idx:
                    ans.append([])
                else:
                    cols_with_j = CuAgent.agent_to_agent_colcheck(own_spheres, agents_spheres[j])
                    ans.append(cols_with_j)
            return ans
            

    class StatManager:
        """
        Run statistics manager
        """
        def __init__(self,my_world, keys:list[str]=[], stat_names:list[list[str]]=[], collect_dt:Union[int,float,list[int],list[float]]=[], dt_key_type:bool='tstep', save:bool=False, verbose:Union[bool,list[bool]]=False,unique_name:str=''):
            self.my_world = my_world
            self.keys = keys
            
            self._last_update_time = []
            self.collect_dt = collect_dt if isinstance(collect_dt, list) else [collect_dt for _ in range(len(stat_names))]
            self.dt_key_type = dt_key_type if isinstance(dt_key_type, list) else [dt_key_type for _ in range(len(stat_names))]
            self.verbose = verbose if isinstance(verbose, list) else [verbose for  _ in range(len(stat_names))]
            self.should_save = save
            self.unique_name = unique_name
            
            self.stats = {}
            for i, stat_name in enumerate(stat_names):
                self.stats[stat_name] = []
                if self.dt_key_type[i] == 'tsec':
                    self._last_update_time.append(time())
                elif self.dt_key_type[i] == 'tstep': # in steps (tstep)
                    self._last_update_time.append(0)
                elif self.dt_key_type[i] == 'tphysics': # in simulated physics steps 
                    self._last_update_time.append(my_world.current_time - 1)
                else:
                    raise ValueError(f"Invalid dt_key_type: {self.dt_key_type[i]}")


        def _get_updated_keys(self, t_step, agent_step_count:int=-1):
            ans = []
            for i in range(len(self.keys)):
                if self.keys[i] == 'tsys':
                    ans.append(time())
                elif self.keys[i] == 'tphysics':
                    ans.append(self.my_world.current_time)
                elif self.keys[i] == 'w_step':
                    ans.append(t_step)
                elif self.keys[i] == 'a_step':  
                    ans.append(agent_step_count)
                else:
                    raise ValueError(f"Invalid key: {self.keys[i]}")

            return ans


        def get_now_update_names(self, tstep:int)->list[str]:
            stats_to_update = []
            for i, stat_name in enumerate(self.stats):
                if self.dt_key_type[i] == 'tsec':
                    now_time = time() # time in seconds
                elif self.dt_key_type[i] == 'tstep':
                    now_time = tstep # time in steps (pass sim step or agent step whatever you want)
                elif self.dt_key_type[i] == 'tphysics':
                    now_time = self.my_world.current_time 
                else:
                    raise ValueError(f"Invalid dt_key_type: {self.dt_key_type[i]}")
                
                update = now_time - self._last_update_time[i] >= self.collect_dt[i]
                if update:
                    stats_to_update.append(stat_name)
                
            return stats_to_update

        def update(self, stats:dict, tstep: Optional[int]=None, agent_step_count:int=-1):
            """
            stats_to_vals is a dict of stat_names to update(returned from get_stats_to_update):
            value: tat vals
            """
            if stats is None or len(stats) == 0:
                return
                
            for stat_name in stats:
                
                # keys to attach to stats
                keys = {}
                now_keys = self._get_updated_keys(tstep, agent_step_count)        
                for key_name, key_val in zip(self.keys, now_keys):
                    keys[key_name] = key_val
                
                # save keys + data
            
                if stat_name not in self.stats:
                    raise ValueError(f"StatManager {self.unique_name} does not have stat {stat_name}")
                
                
                self.stats[stat_name].append((keys, stats[stat_name]))
            
                i = list(self.stats.keys()).index(stat_name)
                # update last update time to know when to update next
                if self.dt_key_type[i] == 'tsec':
                    now_time = time() # time in seconds
                
                elif self.dt_key_type[i] == 'tstep':
                    if tstep is None:
                        raise ValueError(f"tstep is None but dt_in_sec is False (must pass tstep)")
                    now_time = tstep # time in steps (pass sim step or agent step whatever you want)
                
                elif self.dt_key_type[i] == 'tphysics':
                    now_time = self.my_world.current_time 
                
                else:
                    raise ValueError(f"Invalid dt_key_type: {self.dt_key_type[i]}")
                
                self._last_update_time[i] = now_time
                
                if self.verbose[i]:
                    print(f"stats: {self.unique_name}/{stat_name}/{now_time}: {stats[stat_name]}")
                

                
        @staticmethod
        def save(stat_managers:list, out_path:str):
            out = {}
            for i, stat_man in enumerate(stat_managers):
                if not stat_man.should_save:
                    continue
                if stat_man.unique_name in out:
                    new_name = f'{stat_man.unique_name}_{i}'
                    print(f"WARNING: StatManager {stat_man.unique_name} already exists, stat {stat_man.unique_name} will be renamed to {new_name}")
                    stat_man.unique_name = new_name                
                out[stat_man.unique_name] = stat_man.stats
            
            os.makedirs(out_path, exist_ok=True)
            stats_path = os.path.join(out_path, f'stats.pkl')
            with open(stats_path, "wb") as f:
                pickle.dump(out, f)
            print(f"Saved stats to {stats_path}")
            print(f"under keys: {list(out.keys())}")
            return out_path






    class FrameCapturer: # Currently captures 1 frame every 2 world ticks
        def __init__(self, frames_output_dir, camera_eye_pos=[0,-5,3], camera_eye_view=[0,0,0]):
            self.frames_dir = frames_output_dir
            # Create a camera for recording
            self.camera = rep.create.camera()
            
            # Position camera to see the scene
            with self.camera:
                rep.modify.pose(position=[0, -3, 3], look_at=[0, 0, 0.5])
            
            # Create render product
            self.render_product = rep.create.render_product(self.camera, (1280, 720))
            
            # Create writer for video output - RGB only
            self.writer = BasicWriter(
                output_dir=self.frames_dir,
                frame_padding=4,
                rgb=True # RGB only
            )
            
            # Attach writer to render product
            self.writer.attach([self.render_product])

            # Start background capture task
            self.capture_task = asyncio.ensure_future(self.capture_frames_async())
            
            
        
        async def capture_frames_async(self):
            # more info: https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_getting_started.html
            frame_count = 0
            while True:
                # Wait for the next world step to finish (simulate your own step)
                await asyncio.sleep(0)  # yield control so main loop can run my_world.step()
                
                # Trigger Replicator to save current frame without advancing physics
                await rep.orchestrator.step_async()
                frame_count += 1
                # print(f'debug: frame_count: {frame_count}')

        def finish(self, to_mp4_cfg):
            # rep.orchestrator.wait_until_complete()
            try:
                self.capture_task.cancel()
                self.writer.detach()

            except Exception as e:
                print(f'debug: error in finish: {e}')

            self.convert_frames_to_video(**to_mp4_cfg)
            
        def convert_frames_to_video(self,is_on, result_path='', video_fps=30, in_background=True):
            """Convert frames to video using OpenCV"""
            if not is_on:
                return
            # Get all frame files
            if result_path == '':
                result_path = f'{self.frames_dir}/as_video_{video_fps}fps.mp4'
            command = f'python projects_root/experiments/utils/convert_frames_to_video.py --method auto --input_dir {self.frames_dir} --output {result_path} --fps {video_fps} --remove_frames'
            shell = True 
            if in_background:
                subprocess.Popen(command, shell=shell)
            else:
                subprocess.run(command, shell=shell)
            print(f'Finished converting frames to video: {result_path}')
            

    def simulation_startup(simulation_app, my_world, cu_agents, real_robot:bool=False):
        """
        Initialize simulation and wait for it to start playing.
        
        Args:
            simulation_app: The simulation application instance
            my_world: The world instance
            cu_agents: List of CUDA agents
            i: Counter variable (default 0)
        
        Returns:
            int: Updated counter value
        """
        i = 0
        while simulation_app.is_running():
            
            my_world.step(render=True)
            if not my_world.is_playing():
                if i % 100 == 0:
                    print("**** Click Play to start simulation *****")
                i += 1
                continue
            
            step_index = my_world.current_time_step_index
            print(f'debug step_index: {step_index}, i: {i}')
            if step_index <= 10:
                # my_world.reset()
                for a in cu_agents:
                    if a.sim_robot is not None:
                        a.sim_robot.robot._articulation_view.initialize()

                        idx_list = [a.sim_robot.robot.get_dof_index(x) for x in a.robot_cfg["kinematics"]["cspace"]["joint_names"]]
                        if not real_robot: # in simulation- setting joints initial state to definitions in config 
                            a.sim_robot.robot._articulation_view.initialize()
                            idx_list = [a.sim_robot.robot.get_dof_index(x) for x in a.robot_cfg["kinematics"]["cspace"]["joint_names"]]
                            a.sim_robot.robot.set_joint_positions(a.robot_cfg["kinematics"]["cspace"]["retract_config"], idx_list)
                            a.sim_robot.robot.set_joint_velocities(np.zeros_like(a.robot_cfg["kinematics"]["cspace"]["retract_config"]), idx_list)

                        else: # on real robot - setting joints initial state to real robot joint states (positions and velocities)
                            # a.sim_robot.robot.set_joint_positions(a.robot_cfg["kinematics"]["cspace"]["retract_config"], idx_list) # from config
                            real_arm_positions, real_arm_velocities = get_joint_states()[a.idx] # left arm js or right arm js depending on a.idx
                            a.sim_robot.robot.set_joint_positions(real_arm_positions, idx_list)
                            a.sim_robot.robot.set_joint_velocities(real_arm_velocities, idx_list)
                            # a.sim_robot.robot.set_joint_velocities(np.zeros_like(a.robot_cfg["kinematics"]["cspace"]["retract_config"]), idx_list)
                        

                        
                        # max efforts
                        a.sim_robot.robot._articulation_view.set_max_efforts(
                            values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
                        )
                        _ = a.sim_robot.get_js(sync_new=True)
                    
            
            if step_index < 20:
                continue
            break
        return # Return the updated counter



    def fire_up_plotting_server(meta_cfg):
        # agents_with_plotting = [False] * len(meta_cfg["cu_agents"])
        for a_idx, a in enumerate(meta_cfg["cu_agents"]):
            try:
                plot_costs = a["mpc"]["mpc_solver_cfg"]["plot_costs"]
                if plot_costs:
                    from projects_root.utils.plotting_server import start_plotting_server, get_plotting_server
                    start_plotting_server(max_agents=len(meta_cfg["cu_agents"]))
                    print(f"debug: plotting server started...")
                    return True

            except:
                continue
        return False

    


    def get_simulation_timeouts(meta_cfg):
        """
        Get simulation timeouts from meta cfg.
        
        Args:
            meta_cfg: Meta cfg dictionary
        Returns:
        """
        tstep_timeout = meta_cfg["timeout"]["tstep"] if "timeout" in meta_cfg and "tstep" in meta_cfg["timeout"] else 10000
        sec_timeout = meta_cfg["timeout"]["tsec"] if "timeout" in meta_cfg and "tsec" in meta_cfg["timeout"] else 10000
        physics_timeout = meta_cfg["timeout"]["physics_tsec"] if "timeout" in meta_cfg and "physics_tsec" in meta_cfg["timeout"] else 10000
        
        
        return tstep_timeout, sec_timeout, physics_timeout

    def recursive_fill_from_default(a_cfg, default_cfg,use_deepcopy=False):
        for key in default_cfg:
            if key not in a_cfg:
                a_cfg[key] = deepcopy(default_cfg[key]) if use_deepcopy else default_cfg[key]
            elif isinstance(a_cfg[key], dict):
                recursive_fill_from_default(a_cfg[key], default_cfg[key],use_deepcopy=use_deepcopy)
        return a_cfg

    def free_memory(cu_agents, sim_task, sim_env, planner, my_world):
        # just before reset_stage()’s return True
        for a in cu_agents:
            if hasattr(a, "planner") and a.planner is not None:
                # break expensive reference cycles
                a.planner.kill_cost_plots()     # already there
                a.planner = None
                a.stat_man = None
        del cu_agents, sim_task, sim_env, planner, my_world

        import gc, torch
        gc.collect()                # run Python GC
        torch.cuda.empty_cache()    # release cached blocks to driver
        torch.cuda.ipc_collect()    # release CUDA IPC handles (optional)
        
        
    def main(meta_cfg, out_path, real_robot=False):
        
        # start real robot's joint state and joint command servers. Importatnt: using different python environment (using computer python environment to use ros)
        
        # os.system("python3 projects_root/real_world/servers/start_servers.py") # starting reader server (reads js from real robot) and writer server (writes joint commands to real robot)
        # print(f'SERVERS STARTED SUCCESSFULLY, NOW WAITING FOR 5 SECONDS TO LET THE SERVERS STABILIZE')
        # sleep(5)
        
        tsto, sto, pto = get_simulation_timeouts(meta_cfg)

        my_world = World(stage_units_in_meters=1.0)
        
        step_size = 1/60
        
        # my_world.set_physics_step_size(step_size*0.1) 

        # activate_gpu_dynamics(my_world)
        my_world.scene.add_default_ground_plane()
        stage = my_world.stage
        xform = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(xform)
        stage.DefinePrim("/curobo", "Xform")
        usd_help = UsdHelper()
        usd_help.load_stage(my_world.stage)
        tensor_args = TensorDeviceType()
        setup_curobo_logger("warn")


        if meta_cfg["sim_task"]["task_type"] == 'CBSMP1':
            start_positions = CbsMp1Task.get_agents_start_positions(len(meta_cfg["cu_agents"]),**meta_cfg["sim_task"]["task_cfgs"]["CBSMP1"])
            for a_idx, p in enumerate(start_positions):
                meta_cfg["cu_agents"][a_idx]["base_pose"][:3] = deepcopy(p)
                
            
            
        
            
        agent_cfgs = meta_cfg["cu_agents"]
        # init runtime topics (must be done before solvers are initialized (so that dynamic obs cost will be initialized properly))
        init_runtime_topics(n_envs=1, robots_per_env=len(agent_cfgs)) 
        runtime_topics = get_topics()
        env_topics:List[dict] = runtime_topics.get_default_env() if runtime_topics is not None else []


        robot_cfgs_paths = ['' for _ in range(len(agent_cfgs))]
        robot_cfgs = [{} for _ in range(len(agent_cfgs))]
        solver_cfgs = [{} for _ in range(len(agent_cfgs))]
        planner_type = ['' for _ in range(len(agent_cfgs))]
        mpc_particle_file_paths = ['' for _ in range(len(agent_cfgs))]
        mpc_particle_cfgs = [{} for _ in range(len(agent_cfgs))]
        cumotion_plan_cfgs = [{} for _ in range(len(agent_cfgs))]
        cumotion_warmup_cfgs = [{} for _ in range(len(agent_cfgs))]
        sphere_counts_splits = [(-1,-1) for _ in range(len(agent_cfgs))]
        sphere_counts_total = [-1 for _ in range(len(agent_cfgs))]
        col_pred_with = [[] for _ in range(len(agent_cfgs))]
        base_pose = [[] for _ in range(len(agent_cfgs))]
        pub_sub_cfgs = [{} for _ in range(len(agent_cfgs))]
        
        

            
        for a_idx, a_cfg in enumerate(agent_cfgs):
            # print(f'a_idx: {a_idx}')   
            # sleep(2)
            print(f'debug: a {a_idx} base pose before fill: {a_cfg["base_pose"]}')
            recursive_fill_from_default(a_cfg, meta_cfg["default"],use_deepcopy=True)
            print(f'debug: a {a_idx} base pose after fill: {a_cfg["base_pose"]}')

            robot_cfgs_paths[a_idx] = a_cfg["robot"]
            robot_cfgs[a_idx] = load_yaml(robot_cfgs_paths[a_idx])["robot_cfg"]
            # robot_cfgs[a_idx]["kinematics"]["collision_sphere_buffer"] += 0.02

            
            if real_robot:
                # SET RETRACT CONFIG TO BE REAL ROBOT JOINT STATE
                setup_joint_states = get_joint_states() # from real robot
                if setup_joint_states is not None:
                    agent_joint_positions = setup_joint_states[a_idx][0]
                    robot_cfgs[a_idx]["kinematics"]["cspace"]["retract_config"] = agent_joint_positions
                else:
                    print(f"Warning: Could not get joint states for retract config setup for agent {a_idx}")
                    raise ValueError(f"Could not get joint states from real robot for retract config setup for agent {a_idx}")

            # parse base rotation (if euler angles, convert to quaternion)
            base_pose[a_idx] = a_cfg["base_pose"]
            if len(base_pose[a_idx]) == 6: # base pose is from the form of [x,y,z,deg_x, deg_y, deg_z] (position and euler angles)
                base_pose[a_idx][3:] = PoseUtils.rotate_quat([1,0,0,0], base_pose[a_idx][3:], q_in_wxyz=True, q_out_wxyz=True)
                print(f'debug base_pose[a_idx] new: {base_pose[a_idx]}')
            elif len(base_pose[a_idx]) != 7: # base pose is from the form of [x,y,z,qw,qx,qy,qz] (position and quaternion)
                raise ValueError(f"Invalid base pose type: {type(base_pose[a_idx][0])}")


            # planner_type[a_idx] = a_cfg["planner"] if "planner" in a_cfg else meta_cfg["default"]["planner"]
            # sphere_counts_splits[a_idx] = calculate_robot_sphere_count(robot_cfgs[a_idx])
            # sphere_counts_total[a_idx] = sphere_counts_splits[a_idx][0] + sphere_counts_splits[a_idx][1]
    
            # fill missing values from default
            # recursive_fill_from_default(a_cfg, meta_cfg["default"],use_deepcopy=True)
            # if "plan_pub_sub" in a_cfg and "sub" in a_cfg["plan_pub_sub"]:
            #     pub_sub_cfgs[a_idx]["sub"] = a_cfg["plan_pub_sub"]["sub"]
            # else:
            #     pub_sub_cfgs[a_idx]["sub"] = deepcopy(meta_cfg["default"]["plan_pub_sub"]["sub"])
            pub_sub_cfgs[a_idx]["sub"] = a_cfg["plan_pub_sub"]["sub"]
            pub_sub_cfgs[a_idx]["pub"] = a_cfg["plan_pub_sub"]["pub"]
            pub_sub_cfgs[a_idx]["is_on"] = a_cfg["plan_pub_sub"]["is_on"]

            # parse sub to
            if pub_sub_cfgs[a_idx]["sub"]["to"] == "all":
                subto = [i for i in range(len(agent_cfgs)) if i != a_idx] # subscribe to plans from all robots (except itself)    
            else:
                subto = a_cfg["plan_pub_sub"]["sub"]["to"]
            pub_sub_cfgs[a_idx]["sub"]["to"] = subto
            col_pred_with[a_idx] = subto

            
            planner_type[a_idx] = a_cfg["planner"]
            if planner_type[a_idx] == 'mpc':
                solver_cfgs[a_idx] = a_cfg["mpc"]["mpc_solver_cfg"] # if "mpc" in a_cfg and "mpc_solver_cfg" in a_cfg["mpc"] else meta_cfg["default"]["mpc"]["mpc_solver_cfg"]
                mpc_particle_file_paths[a_idx] = solver_cfgs[a_idx]["override_particle_file"] # if "override_particle_file" in solver_cfgs[a_idx] else meta_cfg["default"]["mpc"]["mpc_solver_cfg"]["override_particle_file"]    
                mpc_particle_cfgs[a_idx] = load_yaml(mpc_particle_file_paths[a_idx])   
            else: # cumotion
                solver_cfgs[a_idx] = a_cfg["cumotion"]["motion_gen_cfg"] # if "cumotion" in a_cfg and "motion_gen_cfg" in a_cfg["cumotion"] else meta_cfg["default"]["cumotion"]["motion_gen_cfg"]
                cumotion_plan_cfgs[a_idx] = a_cfg["cumotion"]["motion_gen_plan_cfg"] # if "cumotion" in a_cfg and "motion_gen_plan_cfg" in a_cfg["cumotion"] else meta_cfg["default"]["cumotion"]["motion_gen_plan_cfg"]
                cumotion_warmup_cfgs[a_idx] = a_cfg["cumotion"]["warmup_cfg"] # if "cumotion" in a_cfg and "warmup_cfg" in a_cfg["cumotion"] else meta_cfg["default"]["cumotion"]["warmup_cfg"]

            sphere_counts_splits[a_idx] = calculate_robot_sphere_count(robot_cfgs[a_idx])
            sphere_counts_total[a_idx] = sphere_counts_splits[a_idx][0] + sphere_counts_splits[a_idx][1]
    
        for a_idx, a_cfg in enumerate(agent_cfgs):
            if a_cfg["planner"] == 'mpc':
                n_obstacle_spheres = sum(sphere_counts_total[other_idx] for other_idx in col_pred_with[a_idx])
                publish_robot_context(a_idx, env_topics[a_idx], base_pose[a_idx], n_obstacle_spheres, sphere_counts_total[a_idx], mpc_particle_cfgs[a_idx], col_pred_with[a_idx], mpc_particle_file_paths, robot_cfgs_paths, sphere_counts_splits)

        cu_agents:List[CuAgent] = []
        for a_idx, a_cfg in enumerate(agent_cfgs):
            usd_help.add_subroot('/World', f'/World/robot_{a_idx}', Pose.from_list(base_pose[a_idx]))
            robot, robot_prim_path = add_robot_to_scene(robot_cfgs[a_idx], my_world, subroot=f'/World/robot_{a_idx}', robot_name=f'robot_{a_idx}', position=base_pose[a_idx][:3], orientation=base_pose[a_idx][3:], initialize_world=False) # add_robot_to_scene(self.robot_cfg, self.world, robot_name=self.robot_name, position=self.p_R)
            robot.set_world_pose(base_pose[a_idx][:3], base_pose[a_idx][3:])
            sim_robot_cfg = a_cfg["sim_robot_cfg"] # if "sim_robot_cfg" in a_cfg else meta_cfg["default"]["sim_robot_cfg"]
            viz_color = a_cfg["viz_color"] # if "viz_color" in a_cfg else meta_cfg["default"]["viz_color"]
            sim_robot = SimRobot(robot, robot_prim_path, **sim_robot_cfg, viz_color=viz_color)
            world_cfg = WorldConfig()
            pose_utils = PoseUtils(meta_cfg["pose_utils"]["seed"])
            if planner_type[a_idx] == 'cumotion':
                _motion_gen_config = MotionGenConfig.load_from_robot_config(
                    robot_cfgs[a_idx],
                    world_cfg,
                    tensor_args,
                    **solver_cfgs[a_idx],
                )
                _plan_config = MotionGenPlanConfig(
                    **cumotion_plan_cfgs[a_idx],
                )
                _warmup_config = dict(cumotion_warmup_cfgs[a_idx])            
                planner = CumotionPlanner(base_pose[a_idx], _motion_gen_config, _plan_config, _warmup_config,robot_cfgs[a_idx])
            
            elif planner_type[a_idx] == 'mpc':
                planner = MpcPlanner(base_pose[a_idx], solver_cfgs[a_idx], robot_cfgs[a_idx], world_cfg, mpc_particle_file_paths[a_idx])
                # # Enable live plotting for the centralized plotter
                # try:
                #     arm_reacher = planner._get_arm_reacher()
                #     if arm_reacher is not None:
                #         arm_reacher.enable_live_plotting(True)
                #         print(f"✓ Enabled live plotting for agent {a_idx}")
                #     else:
                #         print(f"⚠ Could not get arm_reacher for agent {a_idx}")
                # except Exception as e:
                #     print(f"⚠ Failed to enable live plotting for agent {a_idx}: {e}")
            else:
                raise ValueError(f"Invalid planner type: {planner_type[a_idx]}")
            
            
            a_stat_man_cfg = deepcopy(a_cfg["stat_man_cfg"]) # if "stat_man_cfg" in a_cfg else deepcopy(meta_cfg["default"]["stat_man_cfg"])
            a_stat_man_cfg["unique_name"] = f'agent_{a_idx}'
            plan_pub_sub = None if (not pub_sub_cfgs[a_idx]["is_on"]) else PlanPubSub(pub_sub_cfgs[a_idx]["pub"], pub_sub_cfgs[a_idx]["sub"], sphere_counts_splits[a_idx][0], sphere_counts_total[a_idx])
            a = CuAgent(
                a_idx,
                tensor_args,
                planner, 
                cu_world_wrapper_cfg=a_cfg["cu_world_wrapper"], # if "cu_world_wrapper" in a_cfg else meta_cfg["default"]["cu_world_wrapper"],
                robot_cfg_path=robot_cfgs_paths[a_idx],
                robot_cfg=robot_cfgs[a_idx],
                sim_robot=sim_robot, # optional, when using simulation
                plan_pub_sub=plan_pub_sub, #PlanPubSub(pub_sub_cfgs[a_idx]["pub"], pub_sub_cfgs[a_idx]["sub"], sphere_counts_splits[a_idx][0], sphere_counts_total[a_idx]),
                viz_color=viz_color,
                stat_man_cfg=a_stat_man_cfg,
                world=my_world if sim_robot is not None else None,
                # is_mobile=a_cfg["is_mobile"] if "is_mobile" in a_cfg else meta_cfg["default"]["is_mobile"],
                # mobile_base_link_subpath=a_cfg["mobile_base_link_subpath"] if "mobile_base_link_subpath" in a_cfg else meta_cfg["default"]["mobile_base_link_subpath"],
            )

            cu_agents.append(a)

        # prepare task 

        agents_task_cfgs = []
        target_colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
        color_cnt = 0
        arm_poses = meta_cfg["sim_task"]["arm_poses"] if "arm_poses" in meta_cfg["sim_task"] else [[] for _ in range(len(cu_agents))] # arms base poses
        centrealized = len(cu_agents) == 1
        
        for a_idx, a in enumerate(cu_agents):
            if a.sim_robot is not None:
                cfg = {}
                links_with_target = [a.planner.ee_link_name, *[link_name for link_name in a.planner.constrained_links_names]]
                for l_idx, link_name in enumerate(links_with_target):
                    link_path = a.sim_robot.path + "/"  + link_name
                    link_prim = my_world.stage.GetPrimAtPath(link_path)
                    link_target_color = SimRobot.parse_viz_color(target_colors[color_cnt])
                    color_cnt += 1
                    if color_cnt >= len(target_colors):
                        color_cnt = 0

                    link_retract_pose = a.planner.plan_goals[link_name]
                    
                    if centrealized:
                        base_idx = l_idx
                    else:
                        base_idx = a_idx
            
                    cfg[link_name] = [link_path, link_prim, link_target_color, link_retract_pose, arm_poses[base_idx]]
                
                agents_task_cfgs.append(cfg)

        if len(agents_task_cfgs) > 0:
            
            sim_task_type = meta_cfg["sim_task"]["task_type"]
            sim_task_cfg = meta_cfg["sim_task"]["task_cfgs"][sim_task_type]
            stat_man_cfg = meta_cfg["sim_task"]["stat_man_cfgs"][sim_task_type]
            level = meta_cfg["sim_task"]["level"]
            if sim_task_type == 'reach':
                sim_task = ReachTask(agents_task_cfgs, my_world, usd_help, tensor_args,level,stat_man_cfg,pose_utils, **sim_task_cfg)
            elif sim_task_type == 'manual':
                sim_task = ManualTask(agents_task_cfgs, my_world, usd_help, tensor_args, stat_man_cfg)
            elif sim_task_type == 'follow':
                sim_task = FollowTask(agents_task_cfgs, my_world, usd_help, tensor_args,level,stat_man_cfg,pose_utils, **sim_task_cfg)
            elif sim_task_type == 'CBSMP1': # cbs multi-robot path planning paper: scenario 1
                sim_task = CbsMp1Task(agents_task_cfgs, my_world, usd_help, tensor_args,level,stat_man_cfg,base_pose,**sim_task_cfg)
            elif sim_task_type == 'bin':
                
                sim_task = BinTask(agents_task_cfgs, my_world, usd_help, tensor_args,level,stat_man_cfg,pose_utils,**sim_task_cfg)
            elif sim_task_type == 'board':
                sim_task = RealWorldVerticallBoard(agents_task_cfgs, my_world, usd_help, tensor_args,level,stat_man_cfg,pose_utils,**sim_task_cfg)
            else:
                raise ValueError(f"Invalid task type: {sim_task_type}")

        obj_cfgs = [ObjectCfg(**obj_cfg) for obj_cfg in meta_cfg["sim_env"]["cfg"]["obj_cfgs"]]
        sim_env = PrimsEnv(my_world, pose_utils, obj_cfgs)
        all_target_paths = [list(sim_task.target_path_to_prim[i].keys())[j] for i in range(len(cu_agents)) for j in range(len(sim_task.target_path_to_prim[i]))]
        # reset collision model for all agents, each agent ignores itslef, its targets and other agents' targets
        for a in cu_agents:
            if a.sim_robot is not None: # if in simulation
                _agent_sim_spheres_root = f'{sim_env.scope_path}/R{a.idx}'
                never_add = [a.sim_robot.path, *all_target_paths, "/curobo" , _agent_sim_spheres_root]
                a.cu_world_wrapper_update_policy["never_add"] += never_add
                a.reset_col_model_from_isaac_sim(usd_help, a.sim_robot.path, ignore_substrings=a.cu_world_wrapper_update_policy["never_add"])
        
        global plotting_alive
        plotting_alive = fire_up_plotting_server(meta_cfg)
        
        my_world.reset()
        my_world.play()
        i = 0
        plans_board:List[Optional[dict]] = [None for _ in range(len(agent_cfgs))] # plans will be stored here
        
        # setup simulation with dummy steps until initialized
        _ = simulation_startup(simulation_app, my_world, cu_agents, real_robot)
        
        # setup stats for simulation
        # sim_stat_man = StatManager(my_world, **meta_cfg["sim_stat_man_cfg"],unique_name='sim_stats')
        
        
        # frame capturing
        frame_capturing_cfg = meta_cfg["out"]["frame_cap"]
        should_capture_frames = frame_capturing_cfg["is_on"]
        if should_capture_frames:
            out_path_frames = os.path.join(out_path, "frames")
            os.makedirs(out_path_frames, exist_ok=True)
            if sim_task_type == 'CBSMP1':
                frame_capturer = FrameCapturer(out_path_frames, sim_task.camera_eye_pos, sim_task.camera_eye_view)
            else:
                frame_capturer = FrameCapturer(out_path_frames)

            
            
        t = 0  # tstep     
        physics_time_start = my_world.current_time # time in seconds in world clock (physics time)
        sim_time_start = time() # time in seconds in process clock (actual running start time)
        agents_spheres = [torch.tensor([]) for _ in range(len(cu_agents))] # for agent-to-agent collision check
        mean_goal_err:List[Optional[tuple[float, float]]] = [None for _ in range(len(cu_agents))] # used for pose wta conflict resolution
        n_arms = len(meta_cfg["sim_task"]["arm_poses"])
        viz_spheres_in_col = set() # visualize spheres in collision in red
        arm_to_arm_col_count = 0 # for debugging
        arm_to_env_col_count = 0 # for debugging
        outdir_name = out_path.split('/')[-1]
        global stop_simulation
        with Progress() as progress:
            task1 = progress.add_task(f"{outdir_name}", total=1000000000000000) # not a real progress bar, just for printing the name
            task2 = progress.add_task(f"Sim Steps (lim={tsto})", total=tsto)
            task3 = progress.add_task(f"Simulation Time [sec] (lim={sto} )", total=sto)
            task4 = progress.add_task(f"Physics Time [sec] (lim={pto})", total=pto)
            
            debug_ctrl_freq_steps = 0
            debug_ctrl_freq_time = time()
            
            if not meta_cfg["async"]: # sync mode
                
                read_subscribed_plans_start_time = 0.0
                read_subscribed_plans_elapsed_steps = 0
                
                pub_plan_elapsed_time = 0.0
                pub_plan_elapsed_steps = 0

                
                solver_time_elapsed_time = 0.0
                solver_time_elapsed_steps = 0
            
                while simulation_app.is_running():

                    prog_bar_tsys_iter_start = time()
                    prog_bar_tphys_iter_start = my_world.current_time 

                    world_step_start = time()
                    my_world.step(render=True)
                    world_step_end = time()
                    world_step_dt = world_step_end - world_step_start
                    # print(f"World step dt: {world_step_dt}")
                    pts_debug = []

                    # Updating targets. Updating targets in sim and return new target poses so planners can react
                    link_name_to_target_pose = sim_task.step()
                    l_name_to_goal_err = sim_task.goal_errors # link name to goal error (p,q) 
                    
                    # Updating obstacles. Updating obstacles in sim so they can be sensed by robots
                    sim_env.step()

                    
                    for a_idx, a in enumerate(cu_agents):
                        planner = a.planner
                        psw = a.planning_stopwatch
                        if a.sim_robot is not None:

                            viz_plans, viz_plans_dt = a.sim_robot.viz_plan_on, a.sim_robot.viz_plan_dt # debug
                            viz_col_spheres_world, viz_col_spheres_robot = a.sim_robot.viz_col_spheres_on_world, a.sim_robot.viz_col_spheres_on_robot
                            viz_col_spheres_dt = a.sim_robot.viz_col_spheres_dt # debug
                            viz_mpc_ee_rollouts, viz_mpc_ee_rollouts_dt = a.sim_robot.viz_mpc_ee_rollouts_on, a.sim_robot.viz_mpc_ee_rollouts_dt 
                            
                            viz_cpred_dt = a.sim_robot.viz_col_pred_dt
                            viz_cpred_own = a.sim_robot.viz_col_pred_own_on
                            viz_cpred_obs = a.sim_robot.viz_col_pred_obs_on

                            ctrl_dof_names = a.sim_robot.robot.dof_names 
                            ctrl_dof_indices = a.sim_robot.robot.get_dof_index 
                            
                            # *** publish plan *** 
                            # sw.on()
                            js = a.sim_robot.get_js(sync_new=False) # get last step's joint state
                            # sw.off()

                            
                            plan = None
                            agent_decentralized = a.plan_pub_sub is not None
                            if agent_decentralized:
                                if js is not None and a.plan_pub_sub.should_pub_now(t):
                                    # print(f'planning debug')
                                    share_full_plan = a.is_full_plan_publisher() # naive means broadcase state as plan over horizon
                                    psw.on()
                                    if a.idx == 0:
                                        pub_plan_start_time = time()
                                    plan = planner.get_estimated_plan(ctrl_dof_names, a.plan_pub_sub.valid_spheres, js, valid_spheres_only=False, naive=not share_full_plan) # get last step's plan (naive <=> broadcast current pose as plan (not future steps))                        
                                    elapsed = psw.off()
                                    if a.idx == 0:
                                        pub_plan_end_time = time()
                                        pub_plan_elapsed_time += pub_plan_end_time - pub_plan_start_time
                                        pub_plan_elapsed_steps += 1
                                        if pub_plan_elapsed_steps % 100 == 0:
                                            print(f'debug: plan get & publish frequency: {pub_plan_elapsed_steps / pub_plan_elapsed_time} hz')
                                    # print(f'debug agent: {a_idx} plan getting time: {elapsed}')
                                    # psw.off()
                                    if plan is not None: # currently available in mpc only
                                        plans_board[a.idx] = plan
                                        if viz_plans and t % viz_plans_dt == 0:
                                            pts_debug.append({'points': plan['task_space']['spheres']['p'], 'color': a.sim_robot.viz_plan_color})
                                    
                                    
            
                
                            # *** sense obstacles *** 
                            
                            a.update_col_model_from_isaac_sim(
                                a.sim_robot.path, 
                                usd_help, 
                                ignore_list=a.cu_world_wrapper_update_policy["never_add"] + a.cu_world_wrapper_update_policy["never_update"], 
                                paths_to_search_obs_under=["/World"]
                            )
                            
                            # *** sense joint positions and velocities *** 
                            # new for the real robot: set new sim robot state joint positions from real robot positions
                            idx_list = [a.sim_robot.robot.get_dof_index(x) for x in a.robot_cfg["kinematics"]["cspace"]["joint_names"]]
                            
                            if real_robot:
                               
                                # before reading joint state from sim, setting it to be as real state of the robot (for visualization in sim of the real staet)
                                # (we basically override the physics of the simulator and making it a visualization of the real world)
                                step_joint_states = get_joint_states()
                                if step_joint_states is not None:
                                    real_arm_positions = step_joint_states[a.idx][0] 
                                    real_arm_velocities = step_joint_states[a.idx][1]
                                    a.sim_robot.robot.set_joint_positions(real_arm_positions, idx_list) # from real arm positions to simulator
                                    a.sim_robot.robot.set_joint_velocities(real_arm_velocities, idx_list) # from real arm velocities to simulator
                            else: # simulation
                                pass # reading state from simulator
                            # Read joint states from simulator (that is already should bez updated to match real arm state)
                            js = a.sim_robot.get_js(sync_new=True)

                            if js is None:
                                print("sim_js is None")
                                continue
                            
                            psw.on()
                            _0 = tensor_args.to_device(js.positions) * 0.0
                            
                            cu_js = JointState(tensor_args.to_device(js.positions),tensor_args.to_device(js.velocities), _0, ctrl_dof_names,_0).get_ordered_joint_state(planner.ordered_j_names)
                            if isinstance(planner, MpcPlanner):
                                planner.update_state(cu_js)
                            elapsed = psw.off()
                            # print(f'debug agent: {a_idx} js update time: {elapsed}')

                            task_space_state_R = a.planner.get_state_in_task_space(cu_js, frame='R')
                            spheres_R = task_space_state_R['spheres']
                            p_R, r_R = spheres_R['p'], spheres_R['r']
                            pr_R = torch.cat((p_R.squeeze(0), r_R.T),dim=1) # S (sphres) x 4 (xyzr)
                            # psw.off()
                            # debug_stime = psw.off()
                            # print(f'debug block 2 time: {debug_stime}')
                            
                            # if a.is_plan_subscriber():
                            if a.is_full_plan_publisher():    
                                # read goal errors for wta conflict resolution
                                psw.on()
                            
                                a_n_links = len(l_name_to_goal_err[a.idx])
                                a_goal_errors = l_name_to_goal_err[a.idx]
                                p_err, q_err = 0.0, 0.0
                                for p_err_link, q_err_link in a_goal_errors.values():
                                    p_err += p_err_link
                                    q_err += q_err_link
                                mean_goal_err[a.idx] = (p_err / a_n_links, q_err / a_n_links)
                                elapsed = psw.off()
                                # print(f'debug: agent {a_idx}, elapsed {elapsed} at goal upadte from others for prioritization in cost function')

                            # *** sense plans ***
                            if a.plan_pub_sub is not None: # everyone that has plan_pub_sub != None are at least subscribers
                                psw.on()
                                if a.idx == 0:
                                    update_col_pred_start_time = time()
                                a.update_col_pred(plans_board, mean_goal_err) # update horizon/naive plans from others
                                if a.idx == 0:
                                    update_col_pred_end_time = time()
                                    read_subscribed_plans_start_time += update_col_pred_end_time - update_col_pred_start_time
                                    read_subscribed_plans_elapsed_steps += 1
                                    if read_subscribed_plans_elapsed_steps % 100 == 0:
                                        print(f'debug: plans reading frequency: {read_subscribed_plans_elapsed_steps / read_subscribed_plans_start_time} hz')
                                elapsed = psw.off()
                                # print(f'debug: agent {a_idx}, elapsed: {elapsed} at update plans of others (naive/over horizon) to use in robot-robot-col cost function')
            
                            # *** sense goals ***
                            goals = link_name_to_target_pose[a.idx]

                            # *** plan ***
                            robot_context = get_topics().get_default_env()[a.idx]
                            robot_context["link_name_to_pose"] = sim_task.get_link_name_to_pose()[a.idx]
                            robot_context["name_link_to_target"] = sim_task.name_link_to_target[a.idx]
                            robot_context["target_name_to_pose"] = sim_task.get_target_name_to_pose()[a.idx]


                            # *** yield action ***
                            
                            if isinstance(planner, CumotionPlanner):
                                psw.on()
                                action = planner.yield_action(goals, cu_js, js.velocities)
                                elapsed = psw.off()
                                # print(f'debug agent {a_idx} solver step time: {elapsed}')
                            elif isinstance(planner, MpcPlanner):
                                psw.on()
                                if a.idx == 0:
                                    solver_time_elapsed_start = time()
                                    # print(f'debug agent {a_idx} solver step time: {elapsed}')
                                action = planner.yield_action(goals)
                                elapsed = psw.off()
                                if a.idx == 0:
                                    solver_time_elapsed_end = time()
                                    solver_time_elapsed_time += solver_time_elapsed_end - solver_time_elapsed_start
                                    solver_time_elapsed_steps += 1
                                    if solver_time_elapsed_steps % 100 == 0:
                                        print(f'debug agent {a_idx} solver step frequency: {solver_time_elapsed_steps / solver_time_elapsed_time}')
                                # print(f'debug agent {a_idx} solver step time: {elapsed}')
                                if viz_mpc_ee_rollouts and t % viz_mpc_ee_rollouts_dt == 0:
                                    pts_debug.append({'points': planner.get_rollouts_in_world_frame(), 'color': a.sim_robot.viz_mpc_ee_rollouts_color})

                            else:
                                raise ValueError(f"Invalid planner type: {planner_type}")
                            
                            # *** act *** 
                            if action is not None: # new: we removed the action in simulator as we are using real robot (simulator now is only for visualization)
                                
                                if real_robot:
                                    # ACTION IN REAL ROBOT:
                                    action_filtered = action.position.cpu().numpy().flatten().tolist() # joint positions only (list in lengths 7 as the arm dofs)
                                    # $$$$$$$$$ WARNING: REAL COMMAND SENDING $$$$$$$$$$$
                                    # uncomment when ready to use real robot!
                                    send_joint_commands(action_filtered, a_idx) # sending also agent index to tell the server which arm we command (0 is left 1 is right)                                
                                else: # 
                                    # act
                                    isaac_action = planner.convert_action_to_isaac(action, ctrl_dof_names, ctrl_dof_indices)
                                    a.sim_robot.articulation_controller.apply_action(isaac_action)
                            
                                a.step_count += 1
                            else:
                                print(f'Warning, action is None')
                            # debug
                            sphere_tensor_W = torch.tensor([])
                            if t % viz_col_spheres_dt == 0:
                                if viz_col_spheres_world:
                                    # task_space_state_W = a.planner.get_state_in_task_space(cu_js, frame='W')                    
                                    # spheres_W = task_space_state_W['spheres']
                                    # p_W, q_W, r_W = spheres_W['p'], spheres_W['q'], spheres_W['r']
                                    # sphere_tensor_W = torch.cat((p_W.squeeze(0), q_W.squeeze(0), r_W.T),dim=1)
                                    sphere_tensor_W = a.get_sphere_tensor_W(cu_js)
                                if viz_col_spheres_robot:
                                    spheres_R = task_space_state_R['spheres']
                                    # p_R, q_R, r_R = spheres_R['p'], spheres_R['q'], spheres_R['r']
                                    p_R, r_R = spheres_R['p'], spheres_R['r']
                                    sphere_viz_tensor_R = torch.cat((p_R.squeeze(0), r_R.T),dim=1)
                                if viz_col_spheres_world and viz_col_spheres_robot:
                                    sphere_viz_tensor = torch.cat((sphere_viz_tensor_R, sphere_tensor_W),dim=0)                    # spheres_tensor_R2 = torch.cat((task_space_state_R2['spheres']['p'].squeeze(0), task_space_state_R2['spheres']['r'].T),dim=1)
                                elif viz_col_spheres_world:
                                    sphere_viz_tensor = sphere_tensor_W
                                elif viz_col_spheres_robot:
                                    sphere_viz_tensor = sphere_viz_tensor_R
                                else:
                                    sphere_viz_tensor = torch.zeros(0,4)
                                
                                a.sim_robot.update_robot_sim_spheres('/curobo', True, a.idx, sphere_viz_tensor,viz_spheres_in_col)

                            if (viz_cpred_own or viz_cpred_obs) and t % viz_cpred_dt == 0:
                                if a.is_full_plan_publisher() and len(cu_agents) > 1:
                                    debug_data = a.planner.get_col_pred_debug()
                                    if debug_data is not None:
                                        p_obs, p_own, r_obs, r_own = debug_data
                                        if a.sim_robot.viz_col_pred_own_mean_only:
                                            p_own = p_own.mean(axis=0)
                                        if viz_cpred_own:
                                            pts_debug.append({'points': p_own, 'color': a.sim_robot.viz_col_pred_own_color})
                                        if viz_cpred_obs:
                                            pts_debug.append({'points': p_obs, 'color': a.sim_robot.viz_col_pred_obs_color})
                            
                            # update agent stats
                            stats_to_update_now = a.stat_man.get_now_update_names(t) # could also pass t
                            # print(f'debug: stats_to_update_now: {stats_to_update_now}')
                            stats = {}
                            for stat_name in stats_to_update_now:
                                

                                if stat_name == 'env_cols':
                                    # print(f'debug: min_esdf_distance: {a.cu_world_wrapper.col_check_wrap.get_min_esdf_distance(pr_R)}')
                                    in_col = a.cu_world_wrapper.col_check_wrap.get_min_esdf_distance(pr_R) < 0.01
                                    if in_col:
                                        print(f"debug: warning robot {a.idx} in col with obstacle!!!")
                                        arm_to_env_col_count += 1
                                    val = in_col  
                                elif stat_name == 'link_target_poses': # link and target poses
                                    val = (robot_context["link_name_to_pose"], robot_context["name_link_to_target"], robot_context["target_name_to_pose"])
                                elif stat_name == 'spheres': # spheres pos and radius (in world frame)
                                    if not len(sphere_tensor_W):
                                        sphere_tensor_W = a.get_sphere_tensor_W(cu_js)
                                    val = sphere_tensor_W
                                    # agents_spheres[a.idx] = sphere_tensor_W
                                elif stat_name == 'total_planning_time': # total planning time
                                    val = psw.total
                                    # print(f'debug: total planning time agent i={a_idx}: {val}') 
                                    # print(f'debug TOTAL CONTROL ITER PLANNING TIME AGENT: {a_idx} = {val/t}')
                                elif stat_name == 'arm_cols': # collisions between arms
                                    viz_spheres_in_col = set()
                                    if not len(sphere_tensor_W):
                                        sphere_tensor_W = a.get_sphere_tensor_W(cu_js)
                                    spheres_per_arm = sphere_tensor_W.shape[0] // n_arms # works only if all robots have same number of spheres
                                    val = False
                                    # val = sphere_tensor_W
                                    if len(cu_agents) > 1: # decentralized
                                        agents_spheres[a.idx] = sphere_tensor_W                                    
                                        collisions = a.check_col_with_others(agents_spheres) # CuAgent.check_collisions_between_agents(agents_spheres)
                                        arm_i_start_sphere = a.idx*spheres_per_arm
                                        for other_idx in range(len(collisions)):
                                            arm_j_start_sphere = other_idx*spheres_per_arm

                                            if len(collisions[other_idx]):
                                                val = True
                                                for k,l in collisions[other_idx]:
                                                    viz_spheres_in_col.add(k+arm_i_start_sphere) # self sphere
                                                    viz_spheres_in_col.add(l+arm_j_start_sphere) # other sphere
                                        
                                    else: # centralized 
                                        arm_tensors_W = a.split_sphere_tensor_W_into_arms(sphere_tensor_W,n_arms) # for centralized robots
                                        for arm_i in range(n_arms):
                                            arm_i_start_sphere = arm_i*spheres_per_arm
                                            if val:
                                                break
                                            for arm_j in range(arm_i+1, n_arms):
                                                if arm_i != arm_j:
                                                    arm_j_start_sphere = arm_j*spheres_per_arm
                                                    collisions = CuAgent.agent_to_agent_colcheck(arm_tensors_W[arm_i], arm_tensors_W[arm_j])
                                                    if len(collisions):
                                                        val = True
                                                        for k,l in collisions:
                                                            print(f'debug kl {k+arm_i_start_sphere},{l+arm_j_start_sphere}')
                                                            viz_spheres_in_col.add(k+arm_i_start_sphere)
                                                            viz_spheres_in_col.add(l+arm_j_start_sphere)                                                        

                                    if val:
                                        arm_to_arm_col_count += 1

                    
                                else:
                                    raise ValueError(f"Invalid stat name: {stat_name}")
                                stats[stat_name] = val
                            a.stat_man.update(stats, t, a.step_count)

                        if len(pts_debug):
                            draw_points(pts_debug)
                    

                    # update task stats
                    task_stats = sim_task.get_stat_vals(sim_task.stat_man.get_now_update_names(t))
                    sim_task.stat_man.update(task_stats,t)
                    # if t % 10 == 0:
                    #     print(f"debug: arm_to_arm_col_count: {arm_to_arm_col_count}, arm_to_env_col_count: {arm_to_env_col_count}")
                    # advance time
                    t += 1                
            
                    # visualize progress          
                    progress.update(task1, advance=1) 
                    progress.update(task2, advance=1) # one time step
                    progress.update(task3, advance= time() - prog_bar_tsys_iter_start) # simulation time
                    progress.update(task4, advance=my_world.current_time - prog_bar_tphys_iter_start) # physics time

                    # Check if reached any of the time limits or got a stop event (ctrl+c)
                    tsto_reached = t > tsto # stop due to time step limit
                    sto_reached = time() - sim_time_start > sto # stop due to simulation time limit
                    pto_reached = my_world.current_time - physics_time_start > pto # stop due to physics time limit
                    
                    debug_ctrl_freq_steps += 1
                    debug_ctrl_freq = debug_ctrl_freq_steps /  (time() - debug_ctrl_freq_time)
                    if t % 100 == 0:
                        print(f"Debug Ctrl Freq (frequency for controlling two arms, per arm is double): {debug_ctrl_freq}")            
                    
                    if tsto_reached or sto_reached or pto_reached or stop_event.is_set() or stop_simulation:
                        if should_capture_frames:
                            frame_capturer.finish(frame_capturing_cfg["to_mp4_cfg"])

                        if meta_cfg["out"]["stats"]: # save states
                            print("Saving stats...")
                            stat_managers:list[StatManager] = [sim_task.stat_man, *[a.stat_man for a in cu_agents]] # [sim_task.stat_man, sim_stat_man, *[a.stat_man for a in cu_agents]] 
                            stats_out = StatManager.save(stat_managers, out_path)
                        if meta_cfg["out"]["meta_cfg"]:
                            with open(os.path.join(out_path, 'meta_cfg.yml'), 'w') as f:
                                yaml.dump(meta_cfg, f)
                        
                        
                        print(f"All Outputs saved to {out_path}")
                    
                        free_memory(cu_agents, sim_task, sim_env, planner, my_world)
                        simulation_app.close()
                        return
                        # return False
                        # else:
                        #     # Thoroughly reset scene and World singleton so next iteration starts clean

                        #     reset_stage(my_world)
                        #     return True

                        
                
                    
                        
            
    def reset_stage(my_world):

        """

        reset stage and world, normally before next simulation
        """
        def _release_viewport_memory():
            """
            Frees the render-targets that the viewport keeps alive.
            Works with both the new (utility) and the legacy viewport.
            Does nothing when running headless.
            """
            import importlib
            for mod_name in (
                "omni.kit.viewport.utility",   # Isaac-Sim ≥ 2023.1
                "omni.kit.viewport_legacy",    # Isaac-Sim 2022.x
            ):
                try:
                    vp = importlib.import_module(mod_name)
                except ImportError:
                    continue

                # 1. New viewport – we get a *window*, then ask it for the interface
                if hasattr(vp, "get_active_viewport_window"):
                    win = vp.get_active_viewport_window()
                    if win is not None and hasattr(win, "get_viewport_interface"):
                        iface = win.get_viewport_interface()
                        if iface is not None and hasattr(iface, "release_resources"):
                            iface.release_resources()
                            return

                # 2. Legacy viewport – static helper already returns the interface
                if hasattr(vp, "get_viewport_interface"):
                    iface = vp.get_viewport_interface()
                    if iface is not None and hasattr(iface, "release_resources"):
                        iface.release_resources()
                        return
        
        try:
            my_world.stop()
        except Exception:
            pass
        try:
            my_world.scene.clear(registry_only=False)
        except Exception:
            pass
        clear_stage()
        try:
            World.clear_instance()
        except Exception:
            pass
        try:    
            from omni.isaac.core.utils.stage import create_new_stage
            create_new_stage()
        except Exception:
            pass
        try:
            simulation_app.update()
        except Exception:
            pass

        _release_viewport_memory()

        import gc, torch
        gc.collect()                # run Python GC
        torch.cuda.empty_cache()    # release cached blocks to driver
        torch.cuda.ipc_collect()    # release CUDA IPC handles (optional)


    # Simple signal handler to set stop flag and let normal flow handle simulation_app.close()
    def signal_handler(signum, _frame):
        print(f"[benchmark_sim] Received signal {signum} - setting stop flags...")
        stop_event.set()  # Set the cooperative stop event
        global stop_simulation
        stop_simulation = True
        print("[benchmark_sim] Stop flags set - normal flow will close simulation_app")

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run main simulation 
    main(meta_cfg, out_path, real_robot)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis_mode", type=str, default="gui", help="vis_mode")
    args = parser.parse_args()
    meta_cfg = load_yaml("projects_root/experiments/benchmarks/cfgs/meta_cfg_arms_benchmark.yml")
    stop_event = mp.Event() # Just a dummy event to pass to the root function (won't be used)
    out_path = os.path.join(meta_cfg["out"]["out_dir_root"], meta_cfg["out"]["batch_dir_name"])
    root(meta_cfg, out_path, stop_event, args.vis_mode)