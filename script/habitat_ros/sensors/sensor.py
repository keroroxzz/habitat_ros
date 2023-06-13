import math
import numpy as np
import numba as nb
import threading
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
        
        # for debugging
        # self.obs_time = 0.0
        # self.pub_time = 0.0
        # self.hz = 0.0
        # self.prev_pub = rospy.Time.now()
        # self.hz_ = 0.0
        # self.prev_pub_ = rospy.Time.now()

        # Used to correct the -z direction of camera projection
        self.correction_rotation = np.asarray([0.0,-1.5707963267,0.0])
        self.correction_matrix = tfs.rotation_matrix(math.pi/2.0, np.asarray([0.0,0.0,1.0]))

        # individual thread publishing the updated observations
        self.mutex = threading.Lock()
        self.mutex.acquire()
        self.observation = None
        self.observation_time = rospy.Time.now()
        self.thread = threading.Thread(target=self.update)
        self.thread.start()

    def __loadSpec__(self, data):
        self.topic = self.extendTopic(data["topic"])
        self.position = np.asarray(data["position"])
        self.position,_  = z_up2y_up(position=self.position)

        self.Rate = rospy.Rate(data['sensor_info']['rate'])
        self.duration = 1.0/data['sensor_info']['rate']

        keys = data.keys()
        if 'orientation' in keys:
            self.orientation = np.asarray(data["orientation"])
        else:
            self.orientation = np.zeros(3, dtype=float)
            
        if 'topic_frame' in keys:
            self.msg_frame = self.extendTopic(data['topic_frame'])
        else:
            self.msg_frame = self.frame

    def __setSensor__(self, sensor:habitat_sim.simulator.Sensor):
        self.sensor = sensor

    def setSensorNode(self, agent_node, sensor):
        node = agent_node.node_sensor_suite.get(self.uuid).node
        self.__setNode__(node)
        self.__setSensor__(sensor)

    def uuid(self):
        return self.uuid

    def getObservation(self, none):
        return self.observation
    
    def noise(self, src: np.ndarray):
        return noise_numba(src, self.meanerror, self.maxerror)

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
        """
        observation is deprecated...
        """
        pass

    def updateObservation(self, msg_time):
        
        if self.observation is not None:
            return False
        
        #t1 = rospy.Time.now()

        self.sensor.draw_observation()
        self.observation =  self.sensor.get_observation()
        self.observation_time = rospy.Time.now() if msg_time is None else msg_time

        # for debugging
        # self.obs_time = (rospy.Time.now()-t1).to_sec()*0.1 + self.obs_time*0.9
        # self.hz_ = 1.0/((rospy.Time.now()-self.prev_pub_).to_sec())*0.1 + self.hz_*0.9
        # self.prev_pub_ = rospy.Time.now()

        self.mutex.release()
        return True

    def update(self):

        while not rospy.is_shutdown():

            self.mutex.acquire()

            #t1 = rospy.Time.now()

            self.publish(None, self.observation_time)
            self.publishTF(self.observation_time)
            dt = (self.duration-((rospy.Time.now()-self.observation_time).to_sec()))

            # for debugging
            # self.pub_time = (rospy.Time.now()-t1).to_sec()*0.1 + self.pub_time*0.9
            # self.hz = 1.0/((rospy.Time.now()-self.prev_pub).to_sec())*0.1 + self.hz*0.9
            # self.prev_pub = rospy.Time.now()

            if dt>0.0:
                rospy.Rate(1.0/dt).sleep()
            self.observation = None
            
