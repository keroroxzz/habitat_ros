# ros libs
import rospy
import tf.transformations as tfs

from habitat_ros.utils import *

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
