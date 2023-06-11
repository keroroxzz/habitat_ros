"""
Habitat-ROS LiDAR Simulator

Date: 2023/3/7

Fix the LiDAR resolution issue
"""

# habitat
import habitat_sim

# ros libs
import rospy

# ros messages
from sensor_msgs.msg import Image

from habitat_ros.utils import *
from habitat_ros.transformation import *
from habitat_ros.controllable_object import *
from habitat_ros.sensors.sensor import *

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
        self.meanerror = sensor_info["mean_error"]
        self.maxerror = sensor_info["max_error"]
        self.rate = sensor_info["rate"]

        self.stride_v = 1
        self.res_h = int(self.hres/self.hfov*360)
        self.res_v = int(self.vres/self.vfov*180)

        # habitat renders in equal resolution and stretch to the required resolution only
        if self.res_h/self.res_v>=2:
            self.stride_v = int(self.res_h/self.res_v)
            self.res_v *= self.stride_v
            self.vres *= self.stride_v
            rospy.logwarn(f"The LiDAR renders in {self.res_v} x {self.res_h}, corrected by {self.stride_v}.")

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
        crop = obs_sensor[self.v_bound:self.v_bound+self.vres:self.stride_v, self.h_bound:self.h_bound+self.hres]
        
        crop = self.noise(crop)
        msg = bridge.cv2_to_imgmsg(crop, encoding="passthrough")
        self.__publish__(self.pub, msg, msg_time)
