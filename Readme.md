![.](https://live.staticflickr.com/65535/53019503002_8f0fb1524c_o_d.png)
# Habitat_ROS

A ROS package connecting the Habitat and providing some common sensors (Laser, Lidar) not supported in the Habitat Sim.
You can easily build a robot and set the topic, static links and sensors with config files.

# Requirments

The package is currently tested with:

    - Ubuntu: 20.04
    - ROS: noetic
    - Conda: 4.12.0
    - habitat-sim: 0.2.1 & 0.2.2

# Installation

Note: You may refer to [habitat-sim](https://github.com/facebookresearch/habitat-sim) for step 1 and 2.

1. Initialize the conda env:

        conda create -n habitat python=3.7 cmake=3.14.0
        conda activate habitat

2. Install the Habitat-Sim package:

        conda install habitat-sim withbullet -c conda-forge -c aihabitat

3. Install the dependencies:

        conda install -c anaconda pyqt
        conda install -c conda-forge pyyaml rospkg defusedxml opencv numba
        conda install -c esrf-bcu simple-pid

4. Install Habitat-ROS package:

        $ cd ~/catkin_ws/src
        $ git clone https://github.com/keroroxzz/habitat_ros.git
        $ cd ..
        & catkin_make

## Trouble Shooting

- loading fail due to pyqt: try different installation order in step 3.

# Setting Dataset

It currently tested with the [HM3D-semantic v0.1 and v0.2 dataset](https://github.com/matterport/habitat-matterport-3dresearch).

Please prepare and set the path to the dataset config file by setting the environment variable "HABITAT_DATASET_PATH".

    $ export HABITAT_DATASET_PATH=your/path/to/dataset.scene_dataset_config.json

Note: We recommend adding this line to the ~/.bashrc or your bash file.

# Startup

Always remenber to activate your conda env before starting this.

    $ conda activate {your habitat env}

Activate a simulation environment (with control panel in default):

    $ roslaunch habitat_ros oreo.launch

Activate a simulation environment (without control panel):

    $ roslaunch habitat_ros oreo.launch control_panel:=false

You can also specify a scene by argument "scene":

    $ roslaunch habitat_ros oreo_sim.launch scene:=00009

# Usage

The control panel defines a few keys to control the robot.
        
    W/S/A/D: move forward/backward/left/right
    Q/E: rotate left/right
    SPACE: accelerate

    O: restart the control panel (sometime it stucks.)
    esc: turn off the control panel

# How to build a custom robot?

We recommend creating a new package to organize the config and model of your custom robot. 

The package should contains the following files, and properly configured these files accroding to your package name, sensor settup, and so on.

    catkin_ws/src/{your project name}
        CMakeLists.txt
        package.xml
        config/
            robot.yaml  <= Should specify the "object_config.json" file within your package
            sensor1.yaml
            sensor2.yaml
            ...
        launch/
            launch_my_robot.launch  <= should launch the yaml files within your package
        robot/
            my_robot/
                collision.glb
                model.glb
                mybot.object_config.json

We also provide a lot of instructions in the comments. 

1. download robot_example.zip in the [latest release](https://github.com/keroroxzz/habitat_ros/releases) and decompress it to your {catkin_ws}/src/
2. catkin_make your workspace
3. Run  the following comamnd to use the example robot in a isolated package.
    
    roslaunch robot_example robot_example.launch

You may also refer to the [releases](https://github.com/keroroxzz/habitat_ros/releases/tag/release).

# Reference

```bibtex
@misc{habitat_ros,
      title={Habitat ROS: An Habitat Package for ROS 1}, 
      author={Chih-Hung Tu},
      year={2023},
      url={https://github.com/keroroxzz/habitat_ros}, 
      journal={GitHub}
}
```

# License

The original default robot "OREO" belongs to Advanced Control Lab (ACL), National Taiwan University.

Please create your robot and prevent using OREO in your publishment, thanks!

Contact the lab for more information about OREO: https://www.ntueeacl.com/

If you have any problem with this package, please open a new issue on Github or find me through my email.

The code is distributed under the CC BY-NC-SA 4.0 License.


