#!/usr/bin/env python3
"""
Use the same initialization method that works with isaacsim command
"""
import sys
import os

# Set command line args
sys.argv = [
    'dataset_collector.py',
    '--combo_cfg_path', os.environ.get('MR_STORM_COMBO_CFG', 'projects_root/experiments/benchmarks/cfgs/combo_cfg.yml'),
    '--ignore_sim_errors',
    '--cluster',
    '--vis_mode', 'headless',
    '--in_process'
]

# Execute the main script
exec(open('projects_root/experiments/core_api/dataset_collector.py').read())
