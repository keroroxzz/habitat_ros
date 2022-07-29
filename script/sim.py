#! /usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import time
import math
import ctypes
import PIL.Image as PILImage
from enum import Enum
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Optional, Tuple

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

import numpy as np

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_angle_axis, quat_to_coeffs, quat_from_magnum
from habitat_sim.utils.common import d3_40_colors_rgb

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, PointCloud2, CameraInfo

import cv2
from cv_bridge import CvBridge
# import open3d as o3d
import tf
import yaml

class HabitatSimInteractiveViewer:
    def __init__(self, sim_settings: Dict[str, Any]) -> None:
        self.action=""
        # add ros node, pub, sub
        rospy.init_node('move_agent', anonymous=True)
        rospy.Subscriber('cmd_vel',Twist, callback=self.callback)
        self.br = tf.TransformBroadcaster()

        self.image_pub = rospy.Publisher("image",Image)
        self.image_full_pub = rospy.Publisher("image_full",Image)
        self.depth_pub = rospy.Publisher("depth",Image)
        self.range_pub = rospy.Publisher("range",Image)
        self.semantic_pub = rospy.Publisher("semantic",Image)
        self.camera_pub = rospy.Publisher("camera_info",CameraInfo, latch=True)

        fname = "/home/rtu/catkin_ws/src/habitat_robot/config/camera.yaml"
        with open(fname, "r") as fp:
            calib_data = yaml.load(fp, yaml.Loader)
        self.camera_info = CameraInfo()
        self.camera_info.width = sim_settings["width"]
        self.camera_info.height = sim_settings["height"]
        self.camera_info.D = calib_data["distortion_coefficients"]["data"]
        self.camera_info.R = calib_data["rectification_matrix"]["data"]
        self.camera_info.distortion_model = calib_data["distortion_model"]
        self.bridge = CvBridge()
        
        self.sim_settings: Dict[str:Any] = sim_settings
        
        # configure our simulator
        self.cfg: Optional[habitat_sim.simulator.Configuration] = None
        self.sim: Optional[habitat_sim.simulator.Simulator] = None
        self.reconfigure_sim()

    ## Utils
    def y_up2z_up(self, position, rotation):
        position = position[(2,0,1),]
        position[:2] *= -1.0
        theta, w = quat_to_angle_axis(rotation)
        w = w[(2,0,1),]
        w[1] *= -1.0
        rotation = quat_from_angle_axis(theta, w)
        return position, rotation

    def nodeTranslationToNumpy(self, translation):
        return np.asarray([translation.x, translation.y, translation.z])

    def getCameraMatrix(self, camera):
        focal = sim_settings["width"] / 2.0 * camera.projection_matrix[0,0]
        K = [focal, 0, sim_settings["width"]/2.0, 0, focal, sim_settings["height"]/2.0, 0, 0, 1.0]
        P = [focal, 0, sim_settings["width"]/2.0, 0, 0, focal, sim_settings["height"]/2.0, 0, 0, 0, 1.0, 0.0]
        return K, P

    def act(self):
        if self.action!="":
            
            agent = self.sim.agents[self.agent_id]
            agent.act(self.action) 
            self.sim.step_world(1.0)
            self.draw_event()

            self.action=""


    def callback(self, msg):
        if msg.linear.x > 0:
            self.action = "move_forward"
        elif msg.linear.x < 0:
            self.action = "move_backward"
        elif msg.linear.y > 0:
            self.action = "move_left"
        elif msg.linear.y < 0:
            self.action = "move_right"


    def toNumpy(self, semantic_obs):
        semantic_img = PILImage.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        return np.array(semantic_img)

    def draw_event(self) -> None:
        """
        Calls continuously to re-render frames and swap the two frame buffers
        at a fixed rate.
        """
        self.observations = self.sim.get_sensor_observations()

        ## use the same stamp time to prevent jiggling
        msg_time = rospy.Time.now()

        ## Get agent state
        agent_state = self.default_agent.get_state()
        position, rotation = self.y_up2z_up(agent_state.position, agent_state.rotation)
        q_coeff = quat_to_coeffs(rotation)
        self.br.sendTransform(position, q_coeff, msg_time, 'robot', 'map')

        # translate the position and orientation to desired format
        position = self.nodeTranslationToNumpy(self.sensor_camera.node.translation)
        rotation = quat_from_magnum(self.sensor_camera.node.rotation)
        position, rotation = self.y_up2z_up(position, rotation)
        q_coeff = quat_to_coeffs(rotation)
        self.br.sendTransform(position, q_coeff, msg_time, 'camera_link', 'robot')

        ## add observations
        self.bgr = self.observations['color_sensor'][...,0:3][...,::-1]
        self.bgr_full = self.observations['color_render'][...,0:3][...,::-1]
        self.depth = self.observations['depth_sensor']
        self.range = self.observations['range_sensor']
        self.semantic = self.toNumpy(self.observations['semantic_sensor'])
        
        cv_bgr = self.bridge.cv2_to_imgmsg(self.bgr, encoding="bgr8")
        cv_bgr_full = self.bridge.cv2_to_imgmsg(self.bgr_full, encoding="bgr8")
        cv_depth = self.bridge.cv2_to_imgmsg(self.depth, encoding="passthrough")
        cv_range = self.bridge.cv2_to_imgmsg(self.range, encoding="passthrough")
        cv_semantic = self.bridge.cv2_to_imgmsg(self.semantic, encoding="8UC4")

        cv_bgr.header.stamp = msg_time
        cv_bgr.header.frame_id = "camera"
        cv_bgr_full.header = cv_bgr.header
        cv_depth.header = cv_bgr.header
        cv_range.header = cv_bgr.header
        cv_semantic.header = cv_bgr.header

        self.camera_info.K, self.camera_info.P = self.getCameraMatrix(self.sensor_camera.render_camera)
        self.camera_info.header = cv_bgr.header

        self.image_pub.publish(cv_bgr)
        self.image_full_pub.publish(cv_bgr_full)
        self.depth_pub.publish(cv_depth)
        self.range_pub.publish(cv_range)
        self.semantic_pub.publish(cv_semantic)
        self.camera_pub.publish(self.camera_info)

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

        action_space: Dict[str, habitat_sim.agent.ActionSpec] = {}

        # build our action space map
        for action in action_list:
            actuation_spec_amt = MOVE if "move" in action else LOOK
            action_spec = make_action_spec(
                action, make_actuation_spec(actuation_spec_amt)
            )
            action_space[action] = action_spec

        sensor_spec: List[habitat_sim.sensor.SensorSpec] = self.cfg.agents[
            self.agent_id
        ].sensor_specifications

        agent_config = habitat_sim.agent.AgentConfiguration(
            height=sim_settings['robot_height'],
            radius=0.1,
            sensor_specifications=sensor_spec,
            action_space=action_space,
            body_type="cylinder",
        )
        return agent_config

    def make_cfg(self, settings):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene_id = settings["scene"]
        sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]
        sim_cfg.enable_physics = settings["enable_physics"]

        # Note: all sensors must have the same resolution
        sensor_specs = []

        # The camera used for rendering
        color_render_spec = habitat_sim.CameraSensorSpec()
        color_render_spec.uuid = "color_render"
        color_render_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_render_spec.resolution = [settings["viewport_height"], settings["viewport_width"]]
        color_render_spec.position = [0.0, settings["sensor_height"], 0.0]
        color_render_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_render_spec)

        ## The sensors sending messages to ROS
        color_sensor_spec = habitat_sim.CameraSensorSpec()
        color_sensor_spec.uuid = "color_sensor"
        color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_spec.resolution = [settings["height"], settings["width"]]
        color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_sensor_spec)

        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [settings["height"], settings["width"]]
        depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(depth_sensor_spec)

        semantic_sensor_spec = habitat_sim.CameraSensorSpec()
        semantic_sensor_spec.uuid = "semantic_sensor"
        semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
        semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(semantic_sensor_spec)

        range_sensor_spec = habitat_sim.EquirectangularSensorSpec()
        range_sensor_spec.uuid = "range_sensor"
        range_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        range_sensor_spec.resolution = [360, 360]
        range_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        range_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR
        sensor_specs.append(range_sensor_spec)

        lidar_sensor_spec = habitat_sim.EquirectangularSensorSpec()
        lidar_sensor_spec.uuid = "lidar_sensor"
        lidar_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        lidar_sensor_spec.resolution = [360, 360]
        lidar_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        lidar_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR
        sensor_specs.append(lidar_sensor_spec)

        # Here you can specify the amount of displacement in a forward action and the turn angle
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def reconfigure_sim(self) -> None:
        """
        Utilizes the current `self.sim_settings` to configure and set up a new
        `habitat_sim.Simulator`, and then either starts a simulation instance, or replaces
        the current simulator instance, reloading the most recently loaded scene
        """
        # configure our sim_settings but then set the agent to our default
        self.agent_id: int = self.sim_settings["default_agent"]
        self.cfg = self.make_cfg(self.sim_settings)
        self.cfg.agents[self.agent_id] = self.default_agent_config()

        if self.sim is None:
            self.sim = habitat_sim.Simulator(self.cfg)

        else:  # edge case
            if self.sim.config.sim_cfg.scene_id == self.cfg.sim_cfg.scene_id:
                # we need to force a reset, so change the internal config scene name
                self.sim.config.sim_cfg.scene_id = "NONE"
            self.sim.reconfigure(self.cfg)

        # post reconfigure
        self.active_scene_graph = self.sim.get_active_scene_graph()
        self.default_agent = self.sim.get_agent(self.agent_id)
        self.agent_body_node = self.default_agent.scene_node
        self.render_camera = self.agent_body_node.node_sensor_suite.get("color_render")
        self.sensor_camera = self.agent_body_node.node_sensor_suite.get("color_sensor")
        self.sensor_depth = self.agent_body_node.node_sensor_suite.get("depth_sensor")
        self.sensor_semantic = self.agent_body_node.node_sensor_suite.get("semantic_sensor")
        self.sensor_range = self.agent_body_node.node_sensor_suite.get("range_sensor")
        self.sensor_lidar = self.agent_body_node.node_sensor_suite.get("lidar_sensor")

        self.tiltable_sensors = [self.render_camera, self.sensor_camera, self.sensor_depth, self.sensor_semantic]

        # set sim_settings scene name as actual loaded scene
        self.sim_settings["scene"] = self.sim.curr_scene_name

        self.step = -1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # optional arguments
    parser.add_argument(
        "--scene",
        default="/home/rtu/dataset/habitat/hm3d/hm3d/00009-vLpv2VX547B/vLpv2VX547B.basis.glb",
        type=str,
        help='scene/stage file to load (default: "NONE")',
    )
    parser.add_argument(
        "--dataset",
        default="default",
        type=str,
        metavar="DATASET",
        help="dataset configuration file to use (default: default)",
    )
    parser.add_argument(
        "--disable_physics",
        action="store_true",
        help="disable physics simulation (default: False)",
    )

    args = parser.parse_args()

    # Setting up sim_settings
    sim_settings: Dict[str, Any] = {
        
        # size of the window
        "viewport_width": 500,
        "viewport_height": 500,

        # size of the sensed image for ROS
        "width": 224,
        "height": 224,

        "scene": args.scene,
        # must specify the dataset config to include the semantic data
        "scene_dataset_config_file": "/home/rtu/dataset/habitat/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
        "test_scene_data_url": "http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip",

        "default_agent": 0,
        "robot_height": 1.0,
        "sensor_height": 1.0,
        "hfov": 90,
        "color_sensor": True,  # RGB sensor (default: ON)
        "semantic_sensor": True,  # semantic sensor (default: OFF)
        "depth_sensor": True,  # depth sensor (default: OFF)
        "ortho_rgba_sensor": False,  # Orthographic RGB sensor (default: OFF)
        "ortho_depth_sensor": False,  # Orthographic depth sensor (default: OFF)
        "ortho_semantic_sensor": False,  # Orthographic semantic sensor (default: OFF)
        "fisheye_rgba_sensor": False,
        "fisheye_depth_sensor": False,
        "fisheye_semantic_sensor": False,
        "equirect_rgba_sensor": False,
        "equirect_depth_sensor": False,
        "equirect_semantic_sensor": False,
        "seed": 1,
        "silent": False,  # do not print log info (default: OFF)

        # settings exclusive to example.py
        "save_png": False,  # save the pngs to disk (default: OFF)
        "print_semantic_scene": False,
        "print_semantic_mask_stats": False,
        "compute_shortest_path": False,
        "compute_action_shortest_path": False,
        "goal_position": [5.047, 0.199, 11.145],
        "enable_physics": not args.disable_physics,
        "enable_gfx_replay_save": False,
        "physics_config_file": "./data/default.physics_config.json",
        "num_objects": 10,
        "test_object_index": 0,
        "frustum_culling": True,
    }

    viewer = HabitatSimInteractiveViewer(sim_settings)

    while not rospy.is_shutdown():

        viewer.act()
        rospy.sleep(0.1)
