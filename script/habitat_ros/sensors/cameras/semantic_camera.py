import numpy as np
import PIL.Image as PILImage

# habitat
from habitat_sim.utils.common import d3_40_colors_rgb

from habitat_ros.utils import *
from habitat_ros.transformation import *
from habitat_ros.controllable_object import *
from habitat_ros.sensors.cameras.camera import *

class SemanticCamera(Camera):
    def __init__(self, name, parent_frame, yaml_file=None):
        super().__init__(name, parent_frame, yaml_file)

    def getSensorType(self):
        return habitat_sim.SensorType.SEMANTIC
    
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

        self.__publish__(self.pub, msg, msg_time)
        self.publishCameraInfo(msg_time)