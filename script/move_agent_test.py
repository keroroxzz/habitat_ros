#! /usr/bin/env python
import math
import os
import random
import sys
import cv2

import numpy as np

import rospy
from geometry_msgs.msg import Twist

import habitat_sim


def navigateAndSee(action):
    global sim
    observations = sim.step(action)
    print("action: ", action)


def render(sim, action):
    global actions, next_action, text_features, room_type, clip_pub

    obs = sim.step(action)

    bgr = obs['sensor'][..., 0:3][..., ::-1]
    depth = obs['depth']

    return next_action, bgr, depth

def callback(msg):
    global sim
    action = "no_action"
    if msg.linear.x > 0:
        action =  "move_forward"
    elif msg.linear.y > 0:
        action =  "turn_left"
    elif msg.linear.y < 0:
        action =  "turn_right"

    if action != "no_action":
        render(sim, action)
    

def prepareHabitat():

    cv2.namedWindow("stereo_pair")

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = (
        "/home/rtu/habitat-lab/data/scene_datasets/HM3D/00808-y9hTuugGdiq/y9hTuugGdiq.basis.glb"
    )

    # RGB camera
    rgb_sensor = habitat_sim.bindings.CameraSensorSpec()
    rgb_sensor.uuid = "sensor"
    rgb_sensor.resolution = [512, 512]
    rgb_sensor.hfov = 72

    # Depth camera
    depth_sensor = habitat_sim.CameraSensorSpec()
    depth_sensor.uuid = "depth"
    depth_sensor.resolution = [50, 50]
    depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor.hfov = 72

    agent_config = habitat_sim.AgentConfiguration()
    agent_config.sensor_specifications = [rgb_sensor, depth_sensor]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_config]))
    return sim

if __name__ == "__main__":
    global sim

    rospy.init_node('move_agent', anonymous=True)
    pub = rospy.Publisher('observer', Twist, queue_size=10)
    sub = rospy.Subscriber('cmd_vel',Twist, callback=callback)

    # test_scene = "/home/rtu/habitat-lab/data/scene_datasets/HM3D/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"

    # sim_settings = {
    #     "scene": test_scene,  # Scene path
    #     "default_agent": 0,  # Index of the default agent
    #     "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    #     "width": 256,  # Spatial resolution of the observations
    #     "height": 256,
    # }

    # cfg = make_simple_cfg(sim_settings)
    # backend_cfg = habitat_sim.SimulatorConfiguration()
    # backend_cfg.scene_id = (
    #     "/home/rtu/habitat-lab/data/scene_datasets/HM3D/00808-y9hTuugGdiq/y9hTuugGdiq.basis.glb"
    # )
    # sim = habitat_sim.Simulator(backend_cfg)
    
    sim = prepareHabitat()

    # # initialize an agent
    # agent = sim.initialize_agent(sim_settings["default_agent"])

    # # Set agent state
    # agent_state = habitat_sim.AgentState()
    # agent_state.position = np.array([-0.6, 0.0, 0.0])  # in world space
    # agent.set_state(agent_state)

    # # Get agent state
    # agent_state = agent.get_state()
    # print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)



    # default action space contains 3 actions: move_forward, turn_left, and turn_right
    # action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
    # print("Discrete action space: ", action_names)

    rospy.spin()

