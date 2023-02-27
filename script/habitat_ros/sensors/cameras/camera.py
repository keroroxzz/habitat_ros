import numpy as np
from abc import abstractmethod

# habitat
import habitat_sim

# ros libs
import rospy

# ros messages
from sensor_msgs.msg import Image, CompressedImage, CameraInfo

from habitat_ros.utils import *
from habitat_ros.transformation import *
from habitat_ros.controllable_object import *
from habitat_ros.sensors.sensor import *

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
        self.height = sensor_info["image_height"]

        keys = data.keys()

        self.pub_compressed = None
        if 'pub_compressed' in keys:
            if data['pub_compressed']:
                self.pub_compressed = rospy.Publisher(self.topic+'/compressed', CompressedImage, queue_size=1)

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