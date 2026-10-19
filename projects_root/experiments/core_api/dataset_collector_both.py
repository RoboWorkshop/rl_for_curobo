
try:
    import isaacsim
except ImportError:
    pass

import os, shutil, yaml, signal, datetime, argparse
from typing import Union
import time
import multiprocessing
from multiprocessing import Process, Event
import numpy as np

# Set multiprocessing method to spawn (better for Isaac Sim)
multiprocessing.set_start_method('spawn', force=True)
import pickle
from curobo.util_file import load_yaml
from copy import deepcopy
import sys
import os

# Add the root directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from projects_root.experiments.core_api import benchmark_both
from projects_root.experiments.core_api.benchmark_both import PoseUtils
from projects_root.real_world.servers.client_tools import get_joint_states, send_joint_commands

import traceback

alg_to_particle_file = {
    'O': 'O',
    'O-': 'O_zero_trust',
    'SD': 'binary_col_cost',
    # 'others': 'others.yml',
}

def make_meta_cfgs(combo_cfg_path):
    
    def get_default_particle_file(alg='O'):
        particle_files_root = 'projects_root/experiments/benchmarks/cfgs/particle'
        particle_file_name = alg_to_particle_file[alg] if alg in alg_to_particle_file else 'others' # (alg if alg == 'O' else 'others') # should be the same except for the 'prior_rule' field
        return particle_files_root + f'/{particle_file_name}.yml' # auto chosen # projects_root/experiments/benchmarks/cfgs/particle_file_arms.yml 
        
    def make_tmp_particle_options(particle_options,default_particle_file_path):
        import yaml
        ret = []
        
        init_cov_options = particle_options["init_cov"] if "init_cov" in particle_options else [-1]
        wta_trust_options = particle_options["wta_trust"] if "wta_trust" in particle_options else [-1]
        wta_weight_options = particle_options["wta_weight"] if "wta_weight" in particle_options else [-1]

        for i,init_cov in enumerate(init_cov_options):
            for j,wta_trust in enumerate(wta_trust_options):
                for k,wta_weight in enumerate(wta_weight_options):
                    # tmp_out_particle_file_path = f'{out_dir}/tmp_particle_file_{alg}_{i}_{j}_{k}.yml'
                    # if not os.path.exists(tmp_out_particle_file_path):
                    #     shutil.copy(default_particle_file_path, tmp_out_particle_file_path)
                    #     particle_cfg = load_yaml(tmp_out_particle_file_path)
                    particle_cfg = load_yaml(default_particle_file_path)
                    if init_cov != -1:
                        particle_cfg["mppi"]["init_cov"] = init_cov
                    if wta_trust != -1:
                        particle_cfg["cost"]["custom"]["arm_base"]["dynamic_obs_cost"]["wta_trust"] = wta_trust
                    if wta_weight != -1:
                        particle_cfg["cost"]["custom"]["arm_base"]["dynamic_obs_cost"]["weight"] = wta_weight
                    # with open(tmp_out_particle_file_path, 'w') as f:
                    #     yaml.dump(particle_cfg, f)
                    #ret.append(tmp_out_particle_file_path)
                    ret.append(particle_cfg)
                    # print(f'debug: particle_cfg: {particle_cfg}')
            
        return ret
    
    colors = ['orange','blue','green','red','purple','yellow']
    dec_robot_fam_to_cfg = {
        'franka': 'franka.yml',
         'franka_mobile': 'franka_mobile.yml', 
         'ur5e': 'ur5e.yml', 
         'ur10e': 'ur10e.yml', 
         'iiwa': 'iiwa.yml', 
         'kinova_gen3': 
         'kinova_gen3.yml', 
         'jaco7': 'jaco7.yml',
         'jaco7': 'jaco7.yml',
        'tinyDisk': 'simple_disk_tiny.yml',
        'g1': 'g1_feb26/left_arm_only.yml'
        }
    cent_robot_cfgs = {
        'franka':
            {
                '1':'franka.yml',
                2:f'franka_dual_arm.yml',
                3:f'franka_3_arm.yml',
                4:f'franka_4_arm.yml',
            },
                    
            'ur10e': {
                1:f'ur10e.yml',
                2:f'dual_ur10e.yml',
                3: f'tri_ur10e.yml',
                '4_05' :'quad_ur10e.yml'
                },
            'ur5e': {
                1:f'ur5e.yml',
                '2_05':f'dual_ur5e_b.yml',
                '2_05b':f'dual_ur5e_b05b.yml',
                '4_05':'quad_ur5e_05.yml',
                '4_04':'quad_ur5e_04.yml'
                }
                
            }
            
    
    
    # take retract cfg of benchmarks
    alg_to_planner = {
        'CC': 'cumotion', # Cumotion centralized
        'O': 'mpc', # Ours
        'SD' : 'mpc', # Storm decentralized
        'SC': 'mpc', # Storm centralized
        'D': 'drrt*', # todo
        'O-': 'mpc' # Ours minus prioirity (priority ablation)
    }
    
    # [pub, sub] 
    # pub means- centralized planners should not publish (pub=False). For decentralized planners - depends. pub=True means publish full policy, and pub=False means publish current pose as policy (equivalent to static obs)
    # sub means- sub tells if subscribing to other robots as obstalces (either their current state or policy). So if other robots exists (like in all decentralized planners) sub should be true.
    alg_to_pub_sub = { 
        'CC':[False,False], # centralized planners do not publish or subscribe to plans
        'O':[True,True], # ours - publish full policy, and subscribe to others (naturally)
        'SD':[False,True], # storm decentralized - publish current pose as policy (equivalent to static obs), still listening to others
        'SC':[False,False], # storm centralized - centralized planners do not publish or subscribe to plans (naturally)
        'D':[False,False], # drrt* - centralized planners do not publish or subscribe to plans (naturally)
        'O-':[True,True], # ours minus priority - same as O (publish full policy, and subscribe to others)
    }


    ret_pose_cfg = load_yaml(benchmarks_ret_cfg)
    combo_cfg = load_yaml(combo_cfg_path)
    print(f'debug: combo_cfg: {combo_cfg}')

    base_options = combo_cfg["base"] # meta cfg templates
    robot_type = combo_cfg["robot_type"] # number of arms
    robot_fam_options = combo_cfg["robot_fam"] # robot family
    alg_options = combo_cfg["alg"] # algorithm 
    task_to_levels_options = combo_cfg["task_to_levels"] # task to levels
    seed_options = combo_cfg["task_seed"] # seed
    particle_options = combo_cfg["particle"] if "particle" in combo_cfg else {}
    out_names = []
    meta_cfgs = []
    particle_cfgs = []

    obj_configs = [
        {
        'obj_shape': 'cube',
        'obj_lin_vel': [0.3,0.0,0.0],
        'obj_size': [0.5,0.5,0.1],
        'obj_pos': [-0.8, 0, 0.5],
        },
        
        {
        'obj_shape': 'cube',
        'obj_lin_vel': [-0.4,0.0,0.0],
        'obj_size': [0.2,0.2,0.05],
        'obj_pos': [0.8,0.0,0.3],
        },
        {
        'obj_shape': 'cube',
        'obj_lin_vel': [0.3,0.3,0.0], # [0.4,0.4,0.0],
        'obj_size': [0.2,0.2,0.05],
        'obj_pos': [-0.9,-0.9,0.6],
        },
        {
        'obj_shape': 'cube',
        'obj_lin_vel': [-0.4,-0.4,0.0],
        'obj_size': [0.5,0.5,0.1],
        'obj_pos': [0.9,0.9,0.4],
        },        
        ]
    reach_follow_objects_by_level = {
        1: [],
        2: [obj_configs[0]],
        3: [obj_configs[0], obj_configs[1]],
        4: [obj_configs[0],obj_configs[1], obj_configs[2]],
        5: [obj_configs[0],obj_configs[1], obj_configs[2], obj_configs[3]],
    }

    for level in range(6,11): # Currently (currently in reach only, not available in follow!) whats changed is the overlapping target boxes mode (if level > 5, then its overlapping target boxes mode)
        reach_follow_objects_by_level[level] = reach_follow_objects_by_level[level-5] 


    for base_cfg_path in base_options:
        for robot_fam in robot_fam_options: # list
            for robot_type in robot_type: # list
                for alg in alg_options: # list
                    with open(base_cfg_path, 'r') as f:
                        base_cfg = yaml.safe_load(f)
                        if 'override_particle_file' in base_cfg['default']['mpc']['mpc_solver_cfg']:
                            particle_file_path = base_cfg['default']['mpc']['mpc_solver_cfg']['override_particle_file']        
                        else:
                            particle_file_path = get_default_particle_file(alg)
                    if len(particle_options) > 0:
                        particle_paths_alg_options = make_tmp_particle_options(combo_cfg["particle"],particle_file_path)
                    else:
                        default_particle_file_path = particle_file_path
                        particle_cfg = load_yaml(default_particle_file_path)
                        particle_paths_alg_options =  [particle_cfg]
                    
                    
                    for task in task_to_levels_options: # dict
                        for level in task_to_levels_options[task]: # list
                            for task_seed in seed_options: # list
                                for particle_cfg in particle_paths_alg_options:

                                # for particle_cfg_path in particle_paths_alg_options: # list
                                    
                                    # particle_cfg = get_default_particle_file(alg)

                                    meta_cfg = load_yaml(base_cfg_path)

                                    
                                    # set pub sub config by alg type
                                    is_pub = alg_to_pub_sub[alg][0]
                                    is_sub = alg_to_pub_sub[alg][1]
                                    cent = alg in ['CC', 'SC','D'] # is algorithm centralized
                                    meta_cfg["default"]["plan_pub_sub"] = {
                                        'is_on': not cent,
                                        'pub':{'is_on':is_pub,'dt':1,'is_dt_in_sec':False,'pr':1.0}, # is on: should publishe full plan. if false: publish current state as plan (naive plan)
                                        'sub':{'is_on':is_sub,'to':'all'} # should compute costs based on other robots etimated plans (either full or naive)
                                        
                                    }
                                    
                                    

                        
                                    # get num of arms and num of agents (n_cfgs) by alg type    
                                     # is centralized planner        
                                    planner_type = alg_to_planner[alg]
                                    if task == 'CBSMP1':
                                        n_disks = level
                                        n_arms = n_disks
                                    else:
                                        n_arms = ret_pose_cfg[robot_fam][robot_type]["n_arms"]
                                    
                                    if cent:
                                        robot_cfg_path =  cent_robot_cfgs[robot_fam][robot_type] #[n_arms]
                                        n_cfgs = 1
                                    else:
                                        robot_cfg_path =  dec_robot_fam_to_cfg[robot_fam]
                                        n_cfgs = n_arms
                                    
                                    robot_cfg_path = os.path.join(robot_cfgs_dir, robot_cfg_path) # get robot cfg path
                                    ret_root = ret_pose_cfg[robot_fam][robot_type]["retract"] # get retract cfg for all arms
                                    pose_root = ret_pose_cfg[robot_fam][robot_type]["pose"] # get pose cfg for all arms

                                    # Set arm poses (base poses of arms, independent of cent/dec)
                                    meta_cfg["sim_task"]["arm_poses"] = []
                                    for arm_idx in range(n_arms):
                                        if task == 'CBSMP1':
                                            a_to_read = 0 # template
                                        else:
                                            a_to_read = arm_idx
                                        arm_position = pose_root["dec"][a_to_read][:3]
                                        arm_euler = pose_root["dec"][a_to_read][3:]
                                        arm_quat = PoseUtils.rotate_quat([1,0,0,0], arm_euler, q_in_wxyz=True, q_out_wxyz=True)
                                        arm_pose = [*arm_position, *arm_quat]
                                        meta_cfg["sim_task"]["arm_poses"].append(arm_pose)
                                    
                                    # Set sim_task
                                    meta_cfg["sim_task"]["task_type"] = task
                                    meta_cfg["sim_task"]["level"] = level
                                    
                                    if task in ['reach', 'follow']:
                                        meta_cfg["sim_env"]["cfg"]["obj_cfgs"] = reach_follow_objects_by_level[level]
                                    else:
                                        pass
                                        # meta_cfg["sim_env"]["cfg"]["obj_cfgs"] = []
                                            

                                    
                                     

                                    # Set cu_agents
                                    cu_agent_cfgs = []
                                    base_cu_agent_cfgs = meta_cfg["cu_agents"] if "cu_agents" in meta_cfg else []

                                    for a_idx in range(n_cfgs):
                                        if cent: # n_cfgs = 1 (centralized planner)
                                            # ret_cfg = ret_pose_cfg[robot_fam][n_arms]["retract"] # list of lists - retract for each arm
                                            ret_cfg = [item for sublist in ret_root for item in sublist] # flatten the list of lists
                                            base_pose = pose_root["cent"]
                                        else:
                                            if task == 'CBSMP1':
                                                idx_to_read = 0
                                            else:
                                                idx_to_read = a_idx
                                            ret_cfg = ret_root[idx_to_read] # in dec mode: arm index = agent index retract cfg for the robot 
                                            base_pose = pose_root["dec"][idx_to_read] # arm base pose   
                                    
                                        if a_idx < len(base_cu_agent_cfgs):
                                            print(f'warning: reading specifications for agent{a_idx} from meta cfg')
                                            agent_cfg = base_cu_agent_cfgs[a_idx]
                                            # recursive_fill_from_default(agent_cfg, meta_cfg["default"],use_deepcopy=True)
                                            
                                        else:
                                            agent_cfg = {}
                                        
                                        

                                        # Override base values with new values
                                        if robot_fam == 'g1' and robot_type == '2_arms_only':
                                            if a_idx == 1:
                                                final_robot_cfg = robot_cfg_path.replace('left_arm_only.yml', 'right_arm_only.yml')
                                            else:
                                                final_robot_cfg = robot_cfg_path
                                            agent_cfg["robot"] = final_robot_cfg
                                        else:
                                            agent_cfg["robot"] = robot_cfg_path
                                        agent_cfg["planner"] = planner_type
                                        # Ensure each agent gets its own copy to avoid shared-list mutations later
                                        agent_cfg["base_pose"] = deepcopy(base_pose)
                                        agent_cfg["viz_color"] = colors[a_idx%len(colors)]
                                        
                                        if not args.real: # simulation mode
                                            agent_cfg["retract_cfg"] = ret_cfg # fixed retract config from config files
                                        else: # real mode
                                            agent_cfg["retract_cfg"] = get_joint_states()[a_idx][0] # read joint positions from real robot ad set as initial config                                 
                                        
                                        cu_agent_cfgs.append(agent_cfg)

                                    meta_cfg["cu_agents"] = cu_agent_cfgs
                                        # meta_cfg["cu_agents"].append(agent_cfg)


                                
                                    meta_cfg["pose_utils"]["seed"] = task_seed


    
                                    out_name = f'R_{robot_fam}_N{n_arms}_A{alg}_T{task}_s{task_seed}_l{level}'
                                    
                                    
                                    
                                    meta_cfgs.append(meta_cfg)
                                    out_names.append(out_name)
                                    particle_cfgs.append(particle_cfg)
                                
    return meta_cfgs, out_names, particle_cfgs



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
    

_cleanup_in_progress = False

def signal_handler(signum, _frame):
    """Central signal handler: set cooperative stop flag; leave cleanup to worker/root."""
    global _cleanup_in_progress
    
    if _cleanup_in_progress:
        print(f"\nReceived signal {signum} during cleanup - force exiting...")
        os._exit(1)
    
    _cleanup_in_progress = True
    print(f"\nReceived signal {signum} - shutting down gracefully...")
    
    try:
        stop_event.set()
        
        # Quick cleanup of any running subprocess
        if 'current_process' in globals() and current_process is not None:
            print("Terminating Isaac Sim subprocess...")
            if current_process.is_alive():
                current_process.terminate()
                current_process.join(timeout=2)  # Reduced timeout
                if current_process.is_alive():
                    print("Force killing Isaac Sim subprocess...")
                    current_process.kill()
                    current_process.join(timeout=1)  # Wait for cleanup
        
        # Properly close multiprocessing resources
        try:
            import multiprocessing
            import time
            
            # Clean up active children
            active_children = multiprocessing.active_children()
            for child in active_children:
                if child.is_alive():
                    child.terminate()
                    child.join(timeout=1)
                    if child.is_alive():
                        child.kill()
            
            # Close the stop_event properly
            if stop_event is not None:
                stop_event.set()
                time.sleep(0.1)  # Give a moment for processes to see the event
                if hasattr(stop_event, 'close'):
                    stop_event.close()
                
        except Exception as e:
            print(f"Warning: Could not clean up multiprocessing resources: {e}")
            
        print("Exiting...")
        
    except Exception as e:
        print(f"Error during signal cleanup: {e}")
    
    # Use sys.exit() instead of os._exit() to allow proper Python cleanup
    sys.exit(0)
    
def cleanup_isaac_processes():
    """
    Aggressively clean up any remaining Isaac Sim processes and resources
    """
    import psutil
    import gc
    import os
    import subprocess
    
    try:
        print("Starting aggressive process cleanup...")
        current_pid = os.getpid()
        print(f"Current process PID: {current_pid}")
        
        # Method 1: Use psutil to find and kill processes
        killed_processes = []
        print("Scanning for processes to clean up...")
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Skip the current process to avoid self-termination
                if proc.info['pid'] == current_pid:
                    continue
                    
                cmdline = proc.info['cmdline'] or []
                cmdline_str = ' '.join(cmdline).lower()
                
                # Look for Isaac, Omni, and our simulation processes (be more specific)
                if any(keyword in cmdline_str for keyword in [
                    'isaacsim', 'isaac-sim', 'isaac.kit', 'omni.isaac', 
                    'benchmark_both', 'simulation_app', 
                    'curobo/src', 'kit.*isaac', 'omniverse'
                ]) and 'python' in cmdline_str:
                    print(f"Killing process PID {proc.info['pid']}: {proc.info['name']}")
                    proc.kill() 
                    killed_processes.append(proc.info['pid'])
                # Skip dataset_collector processes to avoid killing ourselves
                # (Let the user manually clean up old dataset_collector instances if needed)
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        # Method 2: Use pkill as backup for any missed processes (be very specific)
        # Avoid killing the current dataset_collector process
        patterns = [
            "python.*benchmark_both",  # Only kill benchmark_both subprocesses
            "isaac-sim",
            "isaacsim"
        ]
        
        for pattern in patterns:
            try:
                subprocess.run(['pkill', '-f', pattern], capture_output=True)
            except Exception:
                pass
        
        if killed_processes:
            print(f"Cleaned up {len(killed_processes)} processes: {killed_processes}")
        else:
            print("No processes found to clean up")
                
        # Force cleanup GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
                print("GPU memory cleared")
        except ImportError:
            pass
            
        # Force garbage collection
        gc.collect()
        print("Garbage collection completed")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")

def invalidate(out_path):
    """
    Invalidate sim out path to mark sim as corrupted
    """
    import shutil
    print(f'INVALIDATING {out_path}')
    shutil.move(out_path, out_path + '_failed')


if __name__ == "__main__":

    # Set CUDA arch to speed up compilation
    # if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
    #     os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'
    
    # Parse arguments first to see if we should do cleanup
    args = argparse.ArgumentParser()
    args.add_argument("--combo_cfg_path", type=str, default="projects_root/experiments/benchmarks/cfgs/combo_cfg.yml")
    args.add_argument("--vis_mode", type=str, default="gui", choices=["gui", "livestream", "headless"])
    args.add_argument("--cluster", action="store_true") # if True, will run on cluster
    args.add_argument("--job_id", type=str, default='')
    args.add_argument("--in_process", action="store_true", default=False, help="Run the simulation in the same process as the dataset_collector. Automatically sets the num of meta cfgs to 1 (the first in combo) to avoid issues caused by many isaac-sim processes running at the same time")
    args.add_argument('--ignore_sim_errors',action="store_true", default=False)
    args.add_argument('--cleanup', action="store_true", default=True, help="Clean up zombie processes before starting")
    args.add_argument('--real', action="store_true", default=False, help="Use real robot instead of simulation")
    args = args.parse_args()
    
    # Startup cleanup disabled to prevent self-termination
    # User can manually clean up processes if needed
    if args.cleanup:
        print("Note: --cleanup flag provided but startup cleanup is disabled to prevent self-termination")
        print("If you need to clean up old processes, please do so manually with 'pkill' or 'htop'")
    
    meta_cfgs_dir = "projects_root/experiments/benchmarks/cfgs"
    default_meta_cfg_path = "meta_cfg_arms.yml"
    robot_cfgs_dir = "curobo/src/curobo/content/configs/robot"
    benchmarks_ret_cfg = "projects_root/experiments/benchmarks/retract_and_pose.yml"
    meta_cfgs, initial_out_names, particle_cfgs = make_meta_cfgs(args.combo_cfg_path)
    print(f"debug: Generated {len(meta_cfgs)} simulation configurations")
    if args.in_process:
        meta_cfgs = [meta_cfgs[0]]
        initial_out_names = [initial_out_names[0]] 
        particle_cfgs = [particle_cfgs[0]]

        print(f"debug: Running in in_process mode - reduced to  {len(meta_cfgs)} meta cfgs")
    # Create a shared stop_event before installing signal handlers
    stop_event = Event()
    current_process = None  # Track the current subprocess for proper cleanup
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    # stop_simulation = False
    # stop_event = Event()

    batch_dirname_timestamp = 'BATCH_' + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    for meta_cfg, initial_out_name, particle_cfg in zip(meta_cfgs, initial_out_names, particle_cfgs):
        if stop_event.is_set():
            break
        
        

        # Rename output directory if livestream mode
        if args.cluster:
            meta_cfg["out"]["out_dir_root"] = os.path.expanduser('~/mr_mpc_logs') # '/mnt/new_home/evrond/mr_mpc_logs'
            # if not (args.vis_mode == 'livestream' or args.vis_mode == 'headless'):
            #     raise ValueError(f'invalid vis_mode in cluster: {args.vis_mode}')
    

        if len(meta_cfg["out"]["batch_dir_name"]):
            if meta_cfg["out"]["batch_dir_name"] == 'TIMESTAMP':
                meta_cfg["out"]["batch_dir_name"] = batch_dirname_timestamp 
                if args.job_id != '':
                    meta_cfg["out"]["batch_dir_name"] = f'{meta_cfg["out"]["batch_dir_name"]}_job{args.job_id}'
            
            meta_cfg["out"]["out_dir"] = os.path.join(meta_cfg["out"]["out_dir_root"], f'{meta_cfg["out"]["batch_dir_name"]}')
            os.makedirs(meta_cfg["out"]["out_dir"], exist_ok=True)
            if not 'combo_file.yml' in os.listdir(meta_cfg["out"]["out_dir"]):
                shutil.copy(args.combo_cfg_path, os.path.join(meta_cfg["out"]["out_dir"], 'combo_file.yml'))
        else:
            meta_cfg["out"]["out_dir"] = meta_cfg["out"]["out_dir_root"]
        
        
        
        # Make output directory with timestamp and rename the initial out name
        if len(initial_out_name):
        
            sim_start_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            new_out_name = f'{sim_start_timestamp}_{initial_out_name}'
            out_path = os.path.join(meta_cfg["out"]["out_dir"], new_out_name)
            print(f'out_path: {out_path}')
            # exit()
            os.makedirs(out_path, exist_ok=False)
            
        particle_cfg_path = os.path.join(out_path, 'particle_cfg.yml')
        with open(particle_cfg_path, 'w') as f:
            yaml.dump(particle_cfg, f)
        meta_cfg["default"]["mpc"]["mpc_solver_cfg"]["override_particle_file"] = particle_cfg_path


        as_subprocess = not args.in_process
        stop_event = Event()
        
    
        if as_subprocess:
            
            # Pass arguments positionally rather than by name so that we do not rely on the exact
            # parameter names that the child process sees if an older benchmark_both module is found
            current_process = Process(target=benchmark_both.root, args=(meta_cfg, out_path, stop_event, args.vis_mode, args.real))
            current_process.start()
            time.sleep(1)
        

            while current_process.is_alive():
                # print(f'debug: stop_event.is_set(): {stop_event.is_set()}, current_process.is_alive(): {current_process.is_alive()}')
                if not stop_event.is_set():
                    time.sleep(0.1)
                else:
                    time.sleep(5)
                    if current_process.is_alive():
                        current_process.terminate()
                        if current_process.is_alive():
                            current_process.kill()
                    exit()
            if current_process.exitcode is not None:
                if current_process.exitcode != 0:
                    
                    invalidate(out_path)
                    print(f'SIM FAILED!')
                    print(f'error: sim failed with exit code {current_process.exitcode}')
                    # Light cleanup after failed simulation - aggressive cleanup kills the batch
                    if args.ignore_sim_errors:
                        continue
                    raise Exception("Simualtion Suprocess (benchmark_both.root()) failed and run_mode set to  NOT IGRNORRING SIMULATION ERRORS")
                else:
                    # Light cleanup after successful simulation
                    print("Simulation completed successfully")
                    # Only do light cleanup between simulations, not aggressive cleanup
                        
                
            
        else:

            try:
                benchmark_both.root(meta_cfg, out_path, stop_event, args.vis_mode, args.real)
                print(f'Successful sim!')
                # In-process: on cancellation, invalidate and exit loop
                if stop_event.is_set():
                    break
            
            except Exception as e:
                invalidate(out_path)
                print(f'SIM FAILED!')
                print(f'error: {traceback.format_exc()}')
                print('Continuing simulation...')
                
                if args.ignore_sim_errors:
                    if not stop_event.is_set():
                        continue
                    else:
                        break
                else:
                    print(f'Stopping simulations due to error: {e}')
                    raise e
                
                    
                    
        
    
    print(f'all sims done')
    
    # Final cleanup to prevent semaphore leaks
    try:
        import multiprocessing
        import time
        
        # Clean up active children first
        active_children = multiprocessing.active_children()
        if active_children:
            print(f"Cleaning up {len(active_children)} active child processes...")
            for child in active_children:
                if child.is_alive():
                    child.terminate()
                    child.join(timeout=2)
                    if child.is_alive():
                        child.kill()
                        child.join(timeout=1)
        
        # Properly close the stop_event
        if stop_event is not None:
            try:
                stop_event.set()
                time.sleep(0.1)  # Give time for processes to see the event
                if hasattr(stop_event, 'close'):
                    stop_event.close()
            except Exception as e:
                print(f"Warning: Could not close stop_event: {e}")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print("Enhanced cleanup completed - semaphore leaks prevented")
        
    except Exception as e:
        print(f"Warning during final cleanup: {e}")