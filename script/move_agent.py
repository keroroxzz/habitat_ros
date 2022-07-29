#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Twist

import math
import os
import random
import sys

import numpy as np

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut

import cv2
from std_msgs.msg import String
from examples.settings import default_sim_settings, make_cfg
from habitat_sim import physics
from habitat_sim.logging import LoggingContext, logger
from habitat_sim.utils.common import quat_from_angle_axis

import keyboard

class simulator: 
    def __init__(self, sim_settings) -> None:

        rospy.init_node('move_agent', anonymous=True)
        pub = rospy.Publisher('observer', Twist, queue_size=10)
        sub = rospy.Subscriber('cmd_vel',Twist, callback=self.callback)

        # configuration = self.Configuration()
        # configuration.title = "Habitat Sim Interactive Viewer"
        self.sim_settings = sim_settings
        # draw Bullet debug line visualizations (e.g. collision meshes)
        self.debug_bullet_draw = False
        # draw active contact point debug line visualizations
        self.contact_debug_draw = False
        # cache most recently loaded URDF file for quick-reload
        self.cached_urdf = ""

        # set up our movement map
        
        # self.pressed = {
        #     key.UP: False,
        #     key.DOWN: False,
        #     key.LEFT: False,
        #     key.RIGHT: False,
        #     key.A: False,
        #     key.D: False,
        #     key.S: False,
        #     key.W: False,
        #     key.X: False,
        #     key.Z: False,
        # }

        # set up our movement key bindings map
        # self.key_to_action = {
        #     key.UP: "look_up",
        #     key.DOWN: "look_down",
        #     key.LEFT: "turn_left",
        #     key.RIGHT: "turn_right",
        #     key.A: "move_left",
        #     key.D: "move_right",
        #     key.S: "move_backward",
        #     key.W: "move_forward",
        #     key.X: "move_down",
        #     key.Z: "move_up",
        # }

        # toggle physics simulation on/off
        self.simulating = True

        # toggle a single simulation step at the next opportunity if not
        # simulating continuously.
        self.simulate_single_step = False

        # configure our simulator
        self.cfg = None
        self.sim = None
        self.reconfigure_sim()

        # compute NavMesh if not already loaded by the scene.
        if not self.sim.pathfinder.is_loaded and self.cfg.sim_cfg.scene_id != "NONE":
            self.navmesh_config_and_recompute()
                                          
        self.time_since_last_simulation = 0.0
        LoggingContext.reinitialize_from_env()
        logger.setLevel("INFO")                                  
        # self.print_help_text()

    def callback(self, msg):
        action = "no_action"
        if msg.linear.x > 0:
            action =  "move_forward"
        elif msg.linear.x < 0:
            action =  "move_backward"
        elif msg.linear.y > 0:
            action =  "move_left"
        elif msg.linear.y < 0:
            action =  "move_right"
        
        agent = self.sim.agents[self.agent_id]
        # print(agent)
        self.sim.step_world(1.0 / 60)
        agent.act(action)
        observations = self.sim.get_sensor_observations()
        # observations = self.sim.step(action)
        print(observations)
        print(action)
        

    def default_agent_config(self) -> habitat_sim.agent.AgentConfiguration:
        """
        Set up our own agent and agent controls
        """
        make_action_spec = habitat_sim.agent.ActionSpec
        make_actuation_spec = habitat_sim.agent.ActuationSpec
        MOVE, LOOK = 0.07, 1.5

        # all of our possible actions' names
        action_list = [
            "move_left",
            "turn_left",
            "move_right",
            "turn_right",
            "move_backward",
            "look_up",
            "move_forward",
            "look_down",
            "move_down",
            "move_up",
        ]

        action_space = {}

        # build our action space map
        for action in action_list:
            actuation_spec_amt = MOVE if "move" in action else LOOK
            action_spec = make_action_spec(
                action, make_actuation_spec(actuation_spec_amt)
            )
            action_space[action] = action_spec

        sensor_spec = self.cfg.agents[
            self.agent_id
        ].sensor_specifications

        agent_config = habitat_sim.agent.AgentConfiguration(
            height=1.5,
            radius=0.1,
            sensor_specifications=sensor_spec,
            action_space=action_space,
            body_type="cylinder",
        )
        return agent_config


    def make_simple_cfg(self, settings):
        # simulator backend
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = settings["scene"]
        

        # agent
        agent_cfg = habitat_sim.agent.AgentConfiguration()

        # In the 1st example, we attach only one sensor,
        # a RGB visual sensor, to the agent
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
        rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

        agent_cfg.sensor_specifications = [rgb_sensor_spec]

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def reconfigure_sim(self) -> None:
        # configure our sim_settings but then set the agent to our default
        self.cfg = self.make_simple_cfg(self.sim_settings)
        self.agent_id: int = self.sim_settings["default_agent"]
        self.cfg.agents[self.agent_id] = self.default_agent_config()
        
        if self.sim is None:
            self.sim = habitat_sim.Simulator(self.cfg)

        # else:  # edge case
        #     if self.sim.config.sim_cfg.scene_id == self.cfg.sim_cfg.scene_id:
        #         # we need to force a reset, so change the internal config scene name
        #         self.sim.config.sim_cfg.scene_id = "NONE"
        #     self.sim.reconfigure(self.cfg)
        # # post reconfigure
        # self.active_scene_graph = self.sim.get_active_scene_graph()
        # self.default_agent = self.sim.get_agent(self.agent_id)
        # self.agent_body_node = self.default_agent.scene_node
        # self.render_camera = self.agent_body_node.node_sensor_suite.get("color_sensor")
        # # set sim_settings scene name as actual loaded scene
        # self.sim_settings["scene"] = self.sim.curr_scene_name

        # self.step = -1
        agent = self.sim.initialize_agent(self.sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([-0.6, 0.0, 0.0])  # in world space
        agent.set_state(agent_state)


if __name__ == "__main__":

    # Setting up sim_settings
    # sim_settings = default_sim_settings
    # sim_settings["scene"] = "/home/ariel/Datasets/Habitat/3D_assets/hm3d-example-habitat/00337-CFVBbU9Rsyb/CFVBbU9Rsyb.basis.glb"
    # sim_settings["scene_dataset_config_file"] = ""
    # sim_settings["enable_physics"] = False

    sim_settings = {
    "scene": "/home/ariel/Datasets/Habitat/3D_assets/hm3d-example-habitat/00337-CFVBbU9Rsyb/CFVBbU9Rsyb.basis.glb",  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
}

    sim = simulator(sim_settings)
    rospy.spin()