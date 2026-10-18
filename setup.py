"""
Setup script for MR-STORM: Scalable Multi-Arm Control By Distributed MPC (IEEE/RSJ IROS 2026).

Provides both the modern `mr_storm` package and legacy `rl_for_curobo` / `projects_root`
compatibility imports.
"""

from setuptools import find_packages, setup

setup(
    name="rl_for_curobo",
    version="1.0.0",
    description="MR-STORM: Scalable Multi-Arm Control By Distributed MPC (IEEE/RSJ IROS 2026)",
    author="Dan Evron, Elias Goldsztejn, Dan R. Suissa, Ronen I. Brafman",
    author_email="evrond@post.bgu.ac.il",
    url="https://roboworkshop.github.io/multi-robot-mpc/",
    project_urls={
        "Project Website": "https://roboworkshop.github.io/multi-robot-mpc/",
        "Paper (arXiv)": "https://arxiv.org/abs/2602.10114",
        "Source Code": "https://github.com/RoboWorkshop/rl_for_curobo",
    },
    packages=find_packages(include=["mr_storm", "mr_storm.*", "projects_root", "projects_root.*"]),
    install_requires=[
        "rich",
        "numpy>=1.24.0,<=1.26.4",
        "scipy>=1.10.0",
        "tqdm",
        "pyyaml",
    ],
    extras_require={
        "real_ur": ["ur-rtde"],
        "distributed_zmq": ["pyzmq", "msgpack"],
    },
    entry_points={
        "console_scripts": [
            "mr-storm-ros2-node=mr_storm.ros2.distributed_arm_node:main",
        ],
    },
    python_requires=">=3.10",
)