# Habitat_ROS

A ROS package connecting the Habitat and providing some common sensors (Laser, Lidar) not supported in the Habitat Sim.
You can easily build a robot and set the topic, link and sensors with config files.

# Requirments

The package is currently tested with:

    1. Ubuntu: 20.04
    2. ROS: noetic
    3. Conda: 4.12.0
    4. habitat-sim: 0.2.1 & 0.2.2

# Installation

1. Initialize the conda env:

        conda create -n habitat python=3.7 cmake=3.14.0
        conda activate habitat

2. Install the Habitat-Sim package:

        conda install habitat-sim withbullet -c conda-forge -c aihabitat

3. Install the dependencies:

        conda install -c conda-forge pyyaml rospkg defusedxml
        conda install opencv

4. Install Habitat-ROS package:

        $ cd ~/catkin_ws/src
        $ git clone https://github.com/keroroxzz/habitat_ros.git
        $ cd ..
        & catkin_make

# Setting Dataset

It currently supports the HM3D-semantic v1.0 dataset. Please prepare and set the path to the dataset config file by setting the environment variable "HABITAT_DATASET_PATH".

    $ export HABITAT_DATASET_PATH=your/path/to/dataset.scene_dataset_config.json

https://github.com/matterport/habitat-matterport-3dresearch

# Usage

Activate a simulation environment (without control panel):

    $ conda activate {your habitat env}
    $ roslaunch habitat_ros oreo_sim.launch

Activate a simulation environment (with control panel):

    $ conda activate {your habitat env}
    $ roslaunch habitat_ros oreo_sim.launch control_panel:=true
