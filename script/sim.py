#! /usr/bin/env python3
"""
The Habitat_ROS package - Simulation runner
by Brian, Ariel, and James.
"""

# habitat
import habitat_sim

# ros libs
import rospy
import rospkg
from habitat_ros.robot import *

# ros messages
from cv_bridge import CvBridge

bridge = CvBridge()
pkg_path = rospkg.RosPack().get_path("habitat_ros")

default_scene = "/home/rtu/dataset/habitat/hm3d/hm3d/00009-vLpv2VX547B/vLpv2VX547B.basis.glb"
default_dataset = "/home/rtu/dataset/habitat/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
enable_physic = True


class HabitatSimROS:
    def __init__(self, rate) -> None:

        # add ros node, pub, sub
        rospy.init_node('habitat_ros', anonymous=True)

        self.rate = rospy.Rate(rate)
        self.fps: float = rate
        robot_name = rospy.get_param("/robot_name", default="oreo")
        self.robot = Robot(robot_name)
        self.robot.loadSensors()
        # input()
        # Init settings
        self.sim_settings = self.make_sim_settings()
        self.agent_id: int = self.sim_settings["default_agent"]
        self.cfg = self.make_cfg(self.sim_settings)
        self.cfg.agents.append(self.robot.agent_config())
        self.sim = habitat_sim.Simulator(self.cfg)

        self.active_scene_graph = self.sim.get_active_scene_graph()
        self.default_agent = self.sim.get_agent(self.agent_id)
        self.agent_body_node = self.default_agent.scene_node

        self.robot.bindSimulator(self.sim)
        self.robot.setAgentNode(self.agent_body_node)
        self.robot.loadModel(self.sim)


    ## Simulation configuratrion and setting functions ##
    def make_sim_settings(self):
        """
        This function initialize the simulator setting for configureing the scene
        """

        dataset = rospy.get_param("habitat_dataset", default=default_dataset)
        scene = rospy.get_param("habitat_scene", default=default_scene)
        enable_physic = rospy.get_param("enable_physic", default=True)
        
        return {

            # size of the window
            "viewport_width": 500,
            "viewport_height": 500,

            # must specify the dataset config to include the semantic data
            "scene": scene,
            "scene_dataset_config_file": dataset,

            "default_agent": 0,
            "silent": True,  # do not print log info (default: OFF)

            "enable_physics": enable_physic,
        }

    def make_cfg(self, settings) -> habitat_sim.Configuration:
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene_id = settings["scene"]
        sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]
        sim_cfg.enable_physics = settings["enable_physics"]

        return habitat_sim.Configuration(sim_cfg, [])
        
    def draw(self) -> None:
        self.sim.step_world(1.0 / self.fps)

        self.robot.update(rospy.Time.now())
        self.rate.sleep()

if __name__ == "__main__":

    simulator = HabitatSimROS(60)
    while not rospy.is_shutdown():
        simulator.draw()