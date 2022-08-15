import os
import math
import yaml
import numpy as np
import PIL.Image as PILImage
from abc import ABC, abstractmethod
from typing import Dict, List

# habitat
import magnum as mn
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_angle_axis, quat_to_coeffs, quat_from_magnum
from habitat_sim.utils.common import d3_40_colors_rgb

# ros libs
import rospy
import rospkg
import tf
import tf.transformations as tfs

# ros messages
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

bridge = CvBridge()
tfBroadcaster = tf.TransformBroadcaster()
pkg_path = rospkg.RosPack().get_path('habitat_ros')

## Utils
def y_up2z_up(position=None, rotation=None):
    """
    Transfrom the coordinate to z-up
    """

    if position is not None:
        position = position[(0,2,1),]
        position[1] *= -1.0
    
    if rotation is not None:
        theta, w = quat_to_angle_axis(rotation)
        w = w[(0,2,1),]
        w[1] *= -1.0
        rotation = quat_from_angle_axis(theta, w)
    return position, rotation

def z_up2y_up(position=None, rotation=None):
    """
    Transfrom the coordinate back to y-up
    """

    if position is not None:
        position = position[(0,2,1),]
        position[2] *= -1.0
    
    if rotation is not None:
        theta, w = quat_to_angle_axis(rotation)
        w = w[(0,2,1),]
        w[2] *= -1.0
        rotation = quat_from_angle_axis(theta, w)

    return position, rotation

def nodeTranslationToNumpy(translation):
    """
    Translate the translation of node into nparray
    """

    return np.asarray([translation.x, translation.y, translation.z])

class Transformation():
    def __init__(self, frame, parent_frame, position, orientation):

        self.frame = frame
        self.parent_frame = parent_frame
        self.position = position
        self.orientation = orientation

        if len(self.orientation)==3:
            self.orientation = tfs.quaternion_from_euler(self.orientation[0], self.orientation[1], self.orientation[2], "rzyx")

        # rospy.logwarn(f'Init a link tf {frame} from {parent_frame}, {self.position}, {self.orientation}')

    def publish(self, msg_time=None):

        if msg_time is None:
            msg_time=rospy.Time.now()

        tfBroadcaster.sendTransform(self.position, self.orientation, msg_time, self.frame, self.parent_frame)

### The classes handling nodes (robot, sensors, etc.s)
class ControllableObject(ABC):
    def __init__(self, name, parent_frame, yaml_file=None):
        self.node = None
        self.subnodes = []
        self.subtfs = []

        # Load sensor infromation
        self.data = self.__loadParam__(name, parent_frame, filename=yaml_file)

    def __setNode__(self, node):
        self.node = node

    @abstractmethod
    def __loadSpec__(self, data):
        pass

    def extendTopic(self, topic: str):
        extended = topic if topic[0]=='/' else f'/{self.uuid}/{topic}'
        print(f'{topic} ==> {extended}')
        return extended[1:]

    ## Param loading
    def __loadParam__(self, name, parent_frame, filename=None):

        rospy.loginfo(f"Loading sensor parameters of {name} from ROS parameters...")
        if filename:
            with open(filename, "r") as fp:
                data = yaml.load(fp, yaml.Loader)[name]
        else:
            data = rospy.get_param(name)

        self.uuid = data["name"]
        self.type = data["type"]

        self.frame = self.extendTopic(data["frame"])
        self.parent_frame = parent_frame

        keys = data.keys()
        self.is_pub_tf = True
        if 'publish_transfromation' in keys:
            self.is_pub_tf = data["publish_transfromation"]

        # Dynamically add tf link
        if 'tf_links' in keys:
            for tf in data['tf_links'].values():
                frame = self.extendTopic(tf['frame'])
                self.subtfs.append(
                    Transformation(
                        frame,
                        self.frame,
                        tf['position'],
                        tf['orientation']))

        # Dynamically register actuation topics
        if 'actuation' in keys:
            actuations = data['actuation']
            topics = actuations['topics']
            actions = actuations['actions']

            for topic, action in zip(topics, actions):
                def temp_callback(msg: Float32):
                    self.act(action, msg.data)

                topic = self.extendTopic(topic)
                rospy.Subscriber(topic, Float32, temp_callback, queue_size=10)
                rospy.loginfo(f'{self.uuid}: Subscribe {topic} for {action} action.')

        # Loading the specs of child class
        self.__loadSpec__(data)

        return data

    def __tfCorrection__(self, q_coeff):
        return q_coeff

    def act(self, action, amount):

        if self.node is not None:
            control = habitat_sim.agent.ObjectControls()
            act_spec = habitat_sim.agent.ActuationSpec

            # left/right on agent scene node
            control(self.node, action, act_spec(amount))

    def getPose(self):
        position = nodeTranslationToNumpy(self.node.translation)
        rotation = quat_from_magnum(self.node.rotation)
        position, rotation = y_up2z_up(position, rotation)
        q_coeff = quat_to_coeffs(rotation)
        q_coeff = self.__tfCorrection__(q_coeff)
        return position, q_coeff

    def publishTF(self, msg_time=None):

        for s in self.subnodes:
            s.publishTF(msg_time)

        for tf in self.subtfs:
            tf.publish(msg_time)

        if not self.is_pub_tf:
            return

        if msg_time is None:
            msg_time=rospy.Time.now()

        # translate the position and orientation to desired format
        position, q_coeff = self.getPose()
        tfBroadcaster.sendTransform(position, q_coeff, msg_time, self.frame, self.parent_frame)


class Robot(ControllableObject):
    def __init__(self, name, yaml_file=None):
        """
        The name is the namespace in the yaml file.
        """
        super().__init__(name, 'map', yaml_file=None)

        self.cmd_sub = rospy.Subscriber(self.cmd_topic, Twist, self.cmd_cb, queue_size=10)
        self.odom_pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=10)


    def __loadSpec__(self, data):
        self.model_path =  os.path.join(pkg_path, data["model_path"])
        self.model_translation = np.asarray(data["translation"])[(1,2,0),]
        self.model_rotation = data["rotation"]

        self.cmd_topic = self.extendTopic(data["cmd_topic"])

        odom = data["odom"]
        self.odom = Odometry()
        self.odom_topic = self.extendTopic(odom["topic"])
        self.odom.header.frame_id = self.extendTopic(odom["frame"])
        self.odom.child_frame_id = self.extendTopic(odom["child_frame"])

    def getVelocity(self):

        lin_vel = self.vel_control.linear_velocity
        ang_vel = self.vel_control.angular_velocity
        lin_vel,_ = y_up2z_up(position=np.asarray([lin_vel[0], lin_vel[1], lin_vel[2]]))
        ang_vel,_ = y_up2z_up(position=np.asarray([ang_vel[0], ang_vel[1], ang_vel[2]]))
        return lin_vel, ang_vel

    def updateOdom(self):

        position, q_coeff = self.getPose()

        self.odom.pose.pose.position.x = position[0]
        self.odom.pose.pose.position.y = position[1]
        self.odom.pose.pose.position.z = position[2]
        self.odom.pose.pose.orientation.x = q_coeff[0]
        self.odom.pose.pose.orientation.y = q_coeff[1]
        self.odom.pose.pose.orientation.z = q_coeff[2]
        self.odom.pose.pose.orientation.w = q_coeff[3]

        lin_vel, ang_vel = self.getVelocity()
        self.odom.twist.twist.linear.x = lin_vel[0]
        self.odom.twist.twist.linear.y = lin_vel[1]
        self.odom.twist.twist.linear.z = lin_vel[2]
        self.odom.twist.twist.angular.x = ang_vel[0]
        self.odom.twist.twist.angular.y = ang_vel[1]
        self.odom.twist.twist.angular.z = ang_vel[2]

    ## Agent config
    def agent_config(self) -> habitat_sim.agent.AgentConfiguration:
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

        sensor_spec: List[habitat_sim.sensor.SensorSpec] = []

        for sensor in self.subnodes:
            sensor.getSensorSpec(sensor_spec)

        agent_config = habitat_sim.agent.AgentConfiguration(
            sensor_specifications=sensor_spec,
            action_space=action_space,
        )
        return agent_config

    def setAgentNode(self, agent_node):
        self.__setNode__(agent_node)
        for sensor in self.subnodes:
            sensor.setSensorNode(self.node)

    ## Load model asset
    def loadModel(self, sim):
        rigid_obj_mgr = sim.get_rigid_object_manager()
        obj_templates_mgr = sim.get_object_template_manager()

        # load the robot asset
        template_id = obj_templates_mgr.load_configs(self.model_path)[0]
        self.model = rigid_obj_mgr.add_object_by_template_id(template_id, self.node)
        self.model.translation = self.model_translation
        # model.rotation = self.model_rotation
        
        # using the vel controller 
        self.vel_control = self.model.velocity_control
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True

    ## Sensors
    def loadSensors(self):
        self.subnodes = []
        for s in self.data['sensors']:

            type=rospy.get_param(s+'/type')
            if type == "RGB_Camera":
                self.subnodes.append(RGBCamera(s, self.frame))
            elif type == "3D_LiDAR":
                self.subnodes.append(LiDAR(s, self.frame))
            elif type == "2D_Laser":
                self.subnodes.append(Laser(s, self.frame))
            elif type == "Depth_Camera":
                self.subnodes.append(DepthCamera(s, self.frame))
            elif type == "Semantic_Camera":
                self.subnodes.append(SemanticCamera(s, self.frame))
            else:
                rospy.logdebug(f"Unknown sensor type {type}.")

    def setVel(self, lin_vel, ang_vel):

        lin_vel,_  = z_up2y_up(position=lin_vel)
        ang_vel,_  = z_up2y_up(position=ang_vel)

        self.vel_control.linear_velocity = lin_vel
        self.vel_control.angular_velocity = ang_vel

        # experimental
        # lin_vel = self.model.transformation.rotation()*mn.Vector3(lin_vel[0],lin_vel[1],lin_vel[2])
        # lin_vel.y = self.model.linear_velocity.y
        # ang_vel[0] = self.model.angular_velocity.x
        # ang_vel[2] = self.model.angular_velocity.z
        # self.model.linear_velocity = lin_vel
        # self.model.angular_velocity = ang_vel

        # Neet to transform to the local coordinate
        # lin_vel *= 50.0
        # self.model.apply_force(force=mn.Vector3(lin_vel[0], lin_vel[1], lin_vel[2]), relative_position=mn.Vector3(0.0, 0.0, 0.0))
        # self.model.apply_torque(torque=mn.Vector3(ang_vel[0], ang_vel[1], ang_vel[2]))

    def cmd_cb(self, msg: Twist):
        self.setVel(
            np.asarray([msg.linear.x,msg.linear.y,msg.linear.z]),
            np.asarray([msg.angular.x,msg.angular.y,msg.angular.z]))

    def publish(self, observation, msg_time=None):
        for s in self.subnodes:
            s.publish(observation, msg_time)

    def publishOdom(self, msg_time=None):
        self.updateOdom()
        self.odom.header.stamp = rospy.Time.now() if msg_time is None else msg_time
        self.odom_pub.publish(self.odom)

class Sensor(ControllableObject):
    def __init__(self, name, parent_frame, yaml_file=None):
        """
        The name is the namespace in the yaml file.
        """
        super().__init__(name, parent_frame, yaml_file=yaml_file)
        self.pub = rospy.Publisher(self.topic, Image, queue_size=1)
        
        # Used to correct the -z direction of camera projection
        self.correction_rotation = [0.0,-1.5707963267,0.0]
        self.correction_matrix = tfs.rotation_matrix(math.pi/2.0, np.asarray([0.0,0.0,1.0]))

    def __loadSpec__(self, data):
        self.topic = self.extendTopic(data["topic"])
        self.position = np.asarray(data["position"])
        self.position,_  = z_up2y_up(position=self.position)

        keys = data.keys()
        if 'topic_frame' in keys:
            self.msg_frame = self.extendTopic(data['topic_frame'])
        else:
            self.msg_frame = self.frame

    def setSensorNode(self, agent_node):
        node = agent_node.node_sensor_suite.get(self.uuid).node
        self.__setNode__(node)

    def uuid(self):
        return self.uuid

    def getObservation(self, observation):
        return observation[self.uuid]

    def __tfCorrection__(self, q_coeff):
        """
        We want to correct the x axis of the sensor's frame to the front
        """
        matrix = tfs.quaternion_matrix(q_coeff)
        matrix = np.matmul(matrix, self.correction_matrix)
        return tfs.quaternion_from_matrix(matrix)

    def __publish__(self, msg, msg_time=None):
        """
        This function 
        """

        msg.header.stamp = msg_time if msg_time is not None else rospy.Time.now()
        msg.header.frame_id = self.msg_frame
        self.pub.publish(msg)

    @abstractmethod
    def publish(self, observation, msg_time = None):
        pass


class LiDAR(Sensor):
    def __init__(self, name, parent_frame, yaml_file=None):
        super().__init__(name, parent_frame, yaml_file)

        # Load sensor infromation
        self._init_params(self.data)
        self.pub = rospy.Publisher(self.topic, Image, queue_size=1)

    def _init_params(self, data):
        sensor_info = data["sensor_info"]
        self.far = sensor_info["far"]
        self.near = sensor_info["near"]
        self.vfov = sensor_info["vfov"]
        self.hfov = sensor_info["hfov"]
        self.vres = sensor_info["resolution"]["vertical"]
        self.hres = sensor_info["resolution"]["horizontal"]
        self.accuracy = sensor_info["accuracy"]
        self.rate = sensor_info["rate"]

        self.res_v = int(self.vres/self.vfov*180)
        self.res_h = int(self.hres/self.hfov*360)

        self.v_bound = int((self.res_v - self.vres)/2)
        self.h_bound = int((self.res_h - self.hres)/2)

    def uuid(self):

        return self.uuid

    def getEquRectResolution(self):

        return [self.res_v, self.res_h]

    def getSensorSpec(self, sensor_list):

        spec = habitat_sim.EquirectangularSensorSpec()
        spec.uuid = self.uuid
        spec.far = self.far
        spec.near = 0.01
        spec.resolution = self.getEquRectResolution()
        spec.position = self.position
        spec.orientation = self.correction_rotation
        spec.sensor_type = habitat_sim.SensorType.DEPTH
        spec.sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR

        sensor_list.append(spec)

    def publish(self, observation, msg_time = None):
        obs_sensor = self.getObservation(observation)
        crop = obs_sensor[self.v_bound:self.v_bound+self.vres, self.h_bound:self.h_bound+self.hres]
        msg = bridge.cv2_to_imgmsg(crop, encoding="passthrough")
        self.__publish__(msg, msg_time)


class Laser(Sensor):
    def __init__(self, name, parent_frame, yaml_file=None):
        super().__init__(name, parent_frame, yaml_file)

        # Load sensor infromation
        self._init_params(self.data)
        self.pub = rospy.Publisher(self.topic, Image, queue_size=1)

    def _init_params(self, data):

        sensor_info = data["sensor_info"]
        self.far = sensor_info["far"]
        self.near = sensor_info["near"]
        self.hfov = sensor_info["hfov"]
        self.accuracy = sensor_info["accuracy"]
        self.rate = sensor_info["rate"]
        self.hres = sensor_info["resolution"]["horizontal"]
        self.vres = sensor_info["resolution"]["vertical"]

        self.res_h = int(self.hres/self.hfov*360)
        self.h_bound = int((self.res_h - self.hres)/2)

    def uuid(self):

        return self.uuid

    def getResolution(self):

        return [self.vres, self.res_h]

    def getSensorSpec(self, sensor_list):

        spec = habitat_sim.EquirectangularSensorSpec()
        spec.uuid = self.uuid
        spec.far = self.far
        spec.near = 0.01
        spec.resolution = self.getResolution()
        spec.position = self.position
        spec.orientation = self.correction_rotation
        spec.sensor_type = habitat_sim.SensorType.DEPTH
        spec.sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR

        sensor_list.append(spec)

    def publish(self, observation, msg_time = None):
        obs_sensor = self.getObservation(observation)
        crop = obs_sensor[:, self.h_bound:self.h_bound+self.hres]
        msg = bridge.cv2_to_imgmsg(crop, encoding="passthrough")
        self.__publish__(msg, msg_time)


class Camera(Sensor):
    def __init__(self, name, parent_frame, yaml_file=None):
        super().__init__(name, parent_frame, yaml_file)

        self.camera_info = None
        self.cam_info_pub = None

        # Loading sensor infromation
        self._init_params(self.data)
        self.pub = rospy.Publisher(self.topic, Image, queue_size=1)

    def _init_params(self, data):
        sensor_info = data["sensor_info"]
        self.far = sensor_info["far"]
        self.near = sensor_info["near"]
        self.hfov = sensor_info["hfov"]
        self.width = sensor_info["image_width"]
        self.height = sensor_info["image_width"]

        if 'camera_info_topic' in self.data.keys():
            camera_info = CameraInfo()
            camera_info.header.frame_id = self.frame
            camera_info.width = self.width
            camera_info.height = self.height
            camera_info.D = sensor_info["distortion_coefficients"]["data"]
            camera_info.R = sensor_info["rectification_matrix"]["data"]
            camera_info.distortion_model = sensor_info["distortion_model"]
            camera_info.K, camera_info.P = self.getCameraMatrix(camera_info)
            self.camera_info = camera_info
            self.cam_info_pub = rospy.Publisher(
                self.extendTopic(self.data["camera_info_topic"]),
                CameraInfo,
                queue_size=10)

    def getCameraMatrix(self, camera_info):
        """
        Calculate the camera metrix for projection
        """

        focal = camera_info.width / 2.0 / np.tan(self.hfov * np.pi/180.0 / 2.0)
        K = [focal, 0, camera_info.width/2.0, 0, focal, camera_info.height/2.0, 0, 0, 1.0]
        P = [focal, 0, camera_info.width/2.0, 0, 0, focal, camera_info.height/2.0, 0, 0, 0, 1.0, 0.0]
        return K, P

    def uuid(self):

        return self.uuid

    def getSensorSpec(self, sensor_list):

        spec = habitat_sim.CameraSensorSpec()

        if self.type == "RGB_Camera":
            spec.sensor_type = habitat_sim.SensorType.COLOR
        elif self.type == "Depth_Camera":
            spec.sensor_type = habitat_sim.SensorType.DEPTH
        elif self.type == "Semantic_Camera":
            spec.sensor_type = habitat_sim.SensorType.SEMANTIC

        spec.uuid = self.uuid
        spec.resolution = [self.height, self.width]
        spec.far = self.far
        spec.near = self.near
        spec.hfov = self.hfov
        spec.position = self.position
        spec.orientation = self.correction_rotation
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        sensor_list.append(spec)

    def publishCameraInfo(self, msg_time=None):
        if self.camera_info is not None:
            self.camera_info.header.stamp = rospy.Time.now() if msg_time is None else msg_time
            self.cam_info_pub.publish(self.camera_info)

    @abstractmethod
    def publish(self, observation, msg_time = None):
        pass


class RGBCamera(Camera):
    def __init__(self, name, parent_frame, yaml_file=None):
        super().__init__(name, parent_frame, yaml_file)

    def publish(self, observation, msg_time = None):

        obs_sensor = self.getObservation(observation)
        obs_sensor = obs_sensor[...,0:3][...,::-1]
        msg = bridge.cv2_to_imgmsg(obs_sensor, encoding="bgr8")

        self.__publish__(msg, msg_time)
        self.publishCameraInfo(msg_time)


class DepthCamera(Camera):
    def __init__(self, name, parent_frame, yaml_file=None):
        super().__init__(name, parent_frame, yaml_file)

    def publish(self, observation, msg_time = None):

        obs_sensor = self.getObservation(observation)
        msg = bridge.cv2_to_imgmsg(obs_sensor, encoding="32FC1")

        self.__publish__(msg, msg_time)
        self.publishCameraInfo(msg_time)


class SemanticCamera(Camera):
    def __init__(self, name, parent_frame, yaml_file=None):
        super().__init__(name, parent_frame, yaml_file)

    def toNumpy(self, semantic_obs):
        """
        Translate the semantic label into RGB image for publishing
        """
        semantic_img = PILImage.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        return np.array(semantic_img)

    def publish(self, observation, msg_time = None):

        obs_sensor = self.getObservation(observation)
        obs_sensor = self.toNumpy(obs_sensor)
        msg = bridge.cv2_to_imgmsg(obs_sensor, encoding="8UC4")

        self.__publish__(msg, msg_time)
        self.publishCameraInfo(msg_time)