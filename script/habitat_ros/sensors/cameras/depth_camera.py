from habitat_ros.utils import *
from habitat_ros.transformation import *
from habitat_ros.controllable_object import *
from habitat_ros.sensors.cameras.camera import *

class DepthCamera(Camera):
    def __init__(self, name, parent_frame, yaml_file=None):
        super().__init__(name, parent_frame, yaml_file)

    def getSensorType(self):
        return habitat_sim.SensorType.DEPTH

    def publish(self, observation, msg_time = None):

        obs_sensor = self.getObservation(observation)
        msg = bridge.cv2_to_imgmsg(obs_sensor, encoding="32FC1")

        self.__publish__(self.pub, msg, msg_time)
        self.publishCameraInfo(msg_time)