#! /usr/bin/env python

import numpy as np
import rospy
import tf
import cv2
import clip

import torch
import torchvision

import habitat_sim
from habitat_sim.utils.common import quat_to_coeffs, quat_to_angle_axis, quat_from_angle_axis

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String

from cv_bridge import CvBridge
cvBridge = CvBridge()

# initialize
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

room_type = ["bedroom", "dining room", "hall", "bathroom", "living room", "laundry room", "kitchen", "stairs", "door", "corridor", "wall", "unknown"]
text_inputs = torch.cat([clip.tokenize(f"{c}") for c in room_type]).to(device)
text_features = None
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
resize = torchvision.transforms.Resize((224,224))

actions = ["move_forward", "turn_right", "turn_left"]
next_action = actions[0]

sim_settings = {
    
    # size of the window
    "viewport_width": 500,
    "viewport_height": 500,

    # size of the sensed image for ROS
    "width": 224,
    "height": 224,

    "scene": "/home/rtu/dataset/habitat/hm3d/hm3d/00009-vLpv2VX547B/vLpv2VX547B.basis.glb",
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
    "enable_physics": False,
    "enable_gfx_replay_save": False,
    "physics_config_file": "./data/default.physics_config.json",
    "num_objects": 10,
    "test_object_index": 0,
    "frustum_culling": True,
}

def doClip(img):

    #normalization
    img = cv2.resize(img,(224,224))
    img = (2.0*img.copy().astype(float)/255.0-1.0)

    with torch.no_grad():

        tensor = torch.from_numpy(img).to(device)
        tensor = tensor.permute(2,0,1)[None,:,:,:]
        
        image_features = model.encode_image(tensor)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T)
        similarity = similarity.softmax(dim=-1)
    
    values, indices = similarity[0].topk(5)
    print("//==============detection result==============//")
    print(similarity)
    for v, i in zip(values, indices):
        print(f'{room_type[i]}: {100.0*v.item()}')

    return image_features.cpu().numpy()[0], values, indices


def render(sim, action):
    global actions, next_action, text_features, room_type, clip_pub

    agent = sim.agents[0]
    agent.act(action) 
    sim.step_world(1.0)
    obs = sim.get_sensor_observations()

    bgr = obs['color_render'][..., 0:3][..., ::-1]
    depth = obs['depth_sensor']
    semantic = obs['semantic_sensor']
    print(semantic)

    embedding, values, indices = doClip(bgr)

    # Visualization
    # bgr = bgr.astype(np.uint8)
    # bgr = cv2.putText(bgr, room_type[indices[0]], (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    # bgr = cv2.putText(bgr, f'{100.0*values[0].item():.2f}%', (50,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

    embarr = Float32MultiArray()
    embarr.data = embedding
    embarr.layout.data_offset=0
    clip_pub.publish(embarr)

    return next_action, bgr, depth

def default_agent_config(cfg, agent_id) -> habitat_sim.agent.AgentConfiguration:
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

    sensor_spec = cfg.agents[
        agent_id
    ].sensor_specifications

    agent_config = habitat_sim.agent.AgentConfiguration(
        height=sim_settings['robot_height'],
        radius=0.1,
        sensor_specifications=sensor_spec,
        action_space=action_space,
        body_type="cylinder",
    )
    return agent_config

def make_cfg(settings):
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

def prepareHabitat():

    # cv2.namedWindow("stereo_pair")

    agent_id: int = sim_settings["default_agent"]
    cfg = make_cfg(sim_settings)
    cfg.agents[agent_id] = default_agent_config(cfg, agent_id)

    sim = habitat_sim.Simulator(cfg)

    # post reconfigure
    active_scene_graph = sim.get_active_scene_graph()
    default_agent = sim.get_agent(agent_id)
    agent_body_node = default_agent.scene_node
    render_camera = agent_body_node.node_sensor_suite.get("color_render")
    sensor_camera = agent_body_node.node_sensor_suite.get("color_sensor")
    sensor_depth = agent_body_node.node_sensor_suite.get("depth_sensor")
    sensor_semantic = agent_body_node.node_sensor_suite.get("semantic_sensor")
    sensor_range = agent_body_node.node_sensor_suite.get("range_sensor")
    sensor_lidar = agent_body_node.node_sensor_suite.get("lidar_sensor")

    return sim

def instruction_callback(msg):
    
    instruction = [msg.data,]
    text_inputs = torch.cat([clip.tokenize(f"{c}") for c in instruction]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    text_feature_arr = Float32MultiArray()
    text_feature_arr.data = text_features[0].cpu().numpy()
    text_feature_arr.layout.data_offset=0
    clip_text_pub.publish(text_feature_arr)


def y_up2z_up(position, rotation):
    position = position[(2,0,1),]
    position[:2] *= -1.0
    theta, w = quat_to_angle_axis(rotation)
    w = w[(2,0,1),]
    rotation = quat_from_angle_axis(theta, w)
    return position, rotation

def sendTransform(position, rotation, time):
    position, rotation = y_up2z_up(position, rotation)
    q_coeff = quat_to_coeffs(rotation)
    br.sendTransform(position, q_coeff, rospy.Time.now() if time is None else time, 'robot', 'map' )


if __name__ == "__main__":

    rospy.init_node('habitat_robot')
    rgb_pub = rospy.Publisher('camera', Image, queue_size=1)
    depth_pub = rospy.Publisher('depth', Image, queue_size=1)
    clip_pub = rospy.Publisher('clip', Float32MultiArray, queue_size=1)
    clip_text_pub = rospy.Publisher('clip_text_feature', Float32MultiArray, queue_size=1)
    rospy.Subscriber("instruction", String, instruction_callback, queue_size=1)

    br = tf.TransformBroadcaster()

    sim = prepareHabitat()

    bgr = np.zeros((100,100))
    depth = np.zeros((100,100))
    while not rospy.is_shutdown():

        if len(next_action)>0:
            next_action, bgr, depth = render(sim, next_action)
            time = rospy.Time.now()

            sendTransform(sim.agents[0].state.position, sim.agents[0].state.rotation, time=time)

            rbg_msg = cvBridge.cv2_to_imgmsg(bgr, encoding='bgr8')
            depth_msg = cvBridge.cv2_to_imgmsg(depth, encoding='passthrough')
            rbg_msg.header.stamp = time
            depth_msg.header.stamp = time

            rgb_pub.publish(rbg_msg)
            depth_pub.publish(depth_msg)

        else:
            sendTransform(sim.agents[0].state.position, sim.agents[0].state.rotation, time=rospy.Time.now())
            

        cv2.imshow("stereo_pair", bgr)
        k = cv2.waitKey(100)

        if k == ord("w"):
            next_action = actions[0]
        elif k == ord("d"):
            next_action = actions[1]
        elif k == ord("a"):
            next_action = actions[2]
        else:
            next_action = ''


        
