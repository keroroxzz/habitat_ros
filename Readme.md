# Habitat_ROS

A ROS package connecting the Habitat and providing some common sensors (Laser, Lidar) not supported in the Habitat Sim.
You can easily build a robot and set the topic, link and sensors with config files.

# Requirments

The package is currently tested with:

Ubuntu 20.04 + ROS noetic

Conda + habitat-sim 0.2.1

Install habitat: https://github.com/facebookresearch/habitat-sim


# Installation

$ cd ~/catkin_ws/src

$ git clone https://github.com/keroroxzz/habitat_ros.git

$ cd ..

& catkin_make

# Setting Dataset

It currently supports the HM3D-semantic v1.0 dataset.

Please prepare and set the path to the dataset config file.

$ export HABITAT_DATASET_PATH=your/path/to/dataset.scene_dataset_config.json

https://github.com/matterport/habitat-matterport-3dresearch/tree/hm3d-semantics-v0.1-fix

# Usage

$ conda activate {your habitat env}

$ roslaunch habitat_ros oreo_sim.launch