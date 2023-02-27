import numpy as np

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

        # self.hfov = sensor_info["hfov"]
        self.ang_min = sensor_info["ang_min"]
        self.ang_max = sensor_info["ang_max"]
        self.ang_increment = sensor_info["ang_increment"]

        self.meanerror = sensor_info["mean_error"]
        self.maxerror = sensor_info["max_error"]
        # self.rate = sensor_info["rate"]
        # self.hres = sensor_info["resolution"]["horizontal"]
        # self.vres = sensor_info["resolution"]["vertical"]

        if sensor_info["unit"] == "deg":
            self.ang_min = np.deg2rad(self.ang_min)
            self.ang_max = np.deg2rad(self.ang_max)
            self.ang_increment = np.deg2rad(self.ang_increment)

        self.res_h = 2.0*np.pi/self.ang_increment 
        self.res_v = self.res_h # must be the same for full resolution rendering
        self.h_min = int((self.ang_min + np.pi)/self.ang_increment)
        self.h_max = int((self.ang_max + np.pi)/self.ang_increment)
        # self.res_h = int(self.hres/self.hfov*360)
        # self.h_bound = int((self.res_h - self.hres)/2)

    def uuid(self):

        return self.uuid

    def getResolution(self):

        return [self.res_v, self.res_h]

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
        mid = int(obs_sensor.shape[0]/2)
        crop = obs_sensor[mid:mid+1, self.h_min:self.h_max+1] # plus one to include the last scan

        crop = crop + self.noise(crop.shape)
        msg = bridge.cv2_to_imgmsg(crop, encoding="passthrough")
        self.__publish__(self.pub, msg, msg_time)