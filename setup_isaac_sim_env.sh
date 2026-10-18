#!/bin/bash
# Setup script for Isaac Sim environment variables in existing conda environment

# Set the Isaac Sim installation path (override by exporting ISAAC_SIM_PATH before sourcing)
export ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-$HOME/isaacsim}"

# Isaac Sim environment variables
export CARB_APP_PATH=$ISAAC_SIM_PATH/kit
export EXP_PATH=$ISAAC_SIM_PATH/apps
export ISAAC_PATH=$ISAAC_SIM_PATH

# Add Isaac Sim Python paths
export PYTHONPATH=$PYTHONPATH:$ISAAC_SIM_PATH/kit/python/lib/python3.10:$ISAAC_SIM_PATH/kit/python/lib/python3.10/site-packages:$ISAAC_SIM_PATH/python_packages:$ISAAC_SIM_PATH/exts/isaacsim.simulation_app:$ISAAC_SIM_PATH/extsDeprecated/omni.isaac.kit:$ISAAC_SIM_PATH/kit/kernel/py:$ISAAC_SIM_PATH/kit/plugins/bindings-python:$ISAAC_SIM_PATH/exts/isaacsim.robot_motion.lula/pip_prebundle:$ISAAC_SIM_PATH/exts/isaacsim.asset.exporter.urdf/pip_prebundle:$ISAAC_SIM_PATH/extscache/omni.kit.pip_archive-0.0.0+d02c707b.lx64.cp310/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.isaac.core_archive/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.isaac.ml_archive/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.pip.compute/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.pip.cloud/pip_prebundle

# Add Isaac Sim library paths
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ISAAC_SIM_PATH/.:$ISAAC_SIM_PATH/exts/omni.usd.schema.isaac/plugins/IsaacSensorSchema/lib:$ISAAC_SIM_PATH/exts/omni.usd.schema.isaac/plugins/RangeSensorSchema/lib:$ISAAC_SIM_PATH/exts/isaacsim.robot_motion.lula/pip_prebundle:$ISAAC_SIM_PATH/exts/isaacsim.asset.exporter.urdf/pip_prebundle:$ISAAC_SIM_PATH/kit:$ISAAC_SIM_PATH/kit/kernel/plugins:$ISAAC_SIM_PATH/kit/libs/iray:$ISAAC_SIM_PATH/kit/plugins:$ISAAC_SIM_PATH/kit/plugins/bindings-python:$ISAAC_SIM_PATH/kit/plugins/carb_gfx:$ISAAC_SIM_PATH/kit/plugins/rtx:$ISAAC_SIM_PATH/kit/plugins/gpu.foundation

# Accept EULA automatically
export OMNI_KIT_ACCEPT_EULA=YES

echo "Isaac Sim environment variables set up for conda environment: $CONDA_DEFAULT_ENV"
echo "Isaac Sim path: $ISAAC_SIM_PATH"

