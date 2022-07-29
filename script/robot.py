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

    obs = sim.step(action)

    bgr = obs['sensor'][..., 0:3][..., ::-1]
    depth = obs['depth']
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


def prepareHabitat():

    cv2.namedWindow("stereo_pair")

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = (
        "/home/rtu/dataset/habitat/hm3d/hm3d/00009-vLpv2VX547B/vLpv2VX547B.basis.glb"
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

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [512, 512]
    semantic_sensor_spec.position = [0.0, 0.0, 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_config = habitat_sim.AgentConfiguration()
    agent_config.sensor_specifications = [rgb_sensor, depth_sensor, semantic_sensor_spec]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_config]))
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
    default_agent = sim.get_agent(0)
    agent_body_node = default_agent.scene_node
    render_camera = agent_body_node.node_sensor_suite.get("sensor")

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

            print(render_camera.render_camera.camera_matrix)

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


        
