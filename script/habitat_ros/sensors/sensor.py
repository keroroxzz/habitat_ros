import math
import numpy as np
from abc import abstractmethod

# ros libs
import rospy
import tf.transformations as tfs

# ros messages
from sensor_msgs.msg import Image

from habitat_ros.utils import *
from habitat_ros.transformation import *
from habitat_ros.controllable_object import *

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

    def noise(self, shape):
        return np.clip(np.random.randn(shape[0], shape[1])*self.meanerror, -self.maxerror, self.maxerror)

    def __tfCorrection__(self, q_coeff):
        """
        We want to correct the x axis of the sensor's frame to the front
        """
        matrix = tfs.quaternion_matrix(q_coeff)
        matrix = np.matmul(matrix, self.correction_matrix)
        return tfs.quaternion_from_matrix(matrix)

    def __publish__(self, pub, msg, msg_time=None):
        """
        This function 
        """

        msg.header.stamp = msg_time if msg_time is not None else rospy.Time.now()
        msg.header.frame_id = self.msg_frame
        pub.publish(msg)

    @abstractmethod
    def publish(self, observation, msg_time = None):
        pass