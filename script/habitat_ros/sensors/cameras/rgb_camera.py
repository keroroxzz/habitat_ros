from habitat_ros.utils import *
from habitat_ros.transformation import *
from habitat_ros.controllable_object import *
from habitat_ros.sensors.cameras.camera import *

class RGBCamera(Camera):
    def __init__(self, name, parent_frame, yaml_file=None):
        super().__init__(name, parent_frame, yaml_file)

    def getSensorType(self):
        return habitat_sim.SensorType.COLOR

    def publish(self, observation, msg_time = None):

        obs_sensor = self.getObservation(observation)
        obs_sensor = obs_sensor[...,0:3][...,::-1]
        msg = bridge.cv2_to_imgmsg(obs_sensor, encoding="bgr8")

        if not self.pub_compressed is None:
            msg_cp = bridge.cv2_to_compressed_imgmsg(obs_sensor, dst_format="bmp")
            self.__publish__(self.pub_compressed, msg_cp, msg_time)

        self.__publish__(self.pub, msg, msg_time)
        self.publishCameraInfo(msg_time)