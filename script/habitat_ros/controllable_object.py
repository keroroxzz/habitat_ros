import yaml
from abc import ABC, abstractmethod

# habitat
import habitat_sim
from habitat_sim.utils.common import quat_to_coeffs, quat_from_magnum

# ros libs
import rospy

# ros messages
from std_msgs.msg import Float32

from habitat_ros.utils import *
from habitat_ros.transformation import *

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

        if not topic[0]=='/':
            extended = f'/{self.uuid}/{topic}'
            rospy.logdebug(f'{self.uuid}: Extend topic \"{topic}\" as \"{extended}\".')
            return extended[1:]

        return topic[1:]

    ## Param loading
    def __loadParam__(self, data, parent_frame, filename=None):

        rospy.logdebug(f"Loading parameters of {data['name']} from ROS parameters...")
        if filename:
            with open(filename, "r") as fp:
                data = yaml.load(fp, yaml.Loader).values()[0]

        self.uuid = data["name"]
        self.type = data["type"]

        self.frame = self.extendTopic(data["frame"])


        keys = data.keys()
        self.is_pub_tf = True
        if 'publish_transfromation' in keys:
            self.is_pub_tf = data["publish_transfromation"]
            
        self.parent_frame = parent_frame
        self.inverse_parent_frame = False
        if 'transform' in keys:
            if 'frame' in data['transform'].keys():
                self.parent_frame = data['transform']['frame']

            if 'inverse' in data['transform'].keys():
                self.inverse_parent_frame = data['transform']['inverse']

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
                rospy.logdebug(f'{self.uuid}: Subscribe \"{topic}\" for \"{action}\" action.')

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

    def getPose(self, inverse=False):
        if inverse:
            inv_rot = self.node.rotation.inverted()
            position = nodeTranslationToNumpy(-inv_rot.transform_vector(self.node.translation))
            rotation = quat_from_magnum(inv_rot)
        else:
            position = nodeTranslationToNumpy(self.node.translation)
            rotation = quat_from_magnum(self.node.rotation)
        position, rotation = y_up2z_up(position, rotation)
        q_coeff = quat_to_coeffs(rotation)
        q_coeff = self.__tfCorrection__(q_coeff)
        return position, q_coeff

    def publishTF(self, msg_time=None):

        # for s in self.subnodes:
        #     s.publishTF(msg_time)

        for tf in self.subtfs:
            tf.publish(msg_time)

        if not self.is_pub_tf:
            return

        if msg_time is None:
            msg_time=rospy.Time.now()

        # translate the position and orientation to desired format
        position, q_coeff = self.getPose(inverse=self.inverse_parent_frame)
        if self.inverse_parent_frame:
            tfBroadcaster.sendTransform(position, q_coeff, msg_time, self.parent_frame, self.frame)
        else:
            tfBroadcaster.sendTransform(position, q_coeff, msg_time, self.frame, self.parent_frame)

