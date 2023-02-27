import os
import numpy as np
from typing import Dict, List

# habitat
import magnum as mn
import habitat_sim
from habitat_sim.utils.common import quat_to_magnum, quat_from_coeffs

# ros libs
import rospy
import rospkg

# ros messages
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from habitat_ros.utils import *
from habitat_ros.transformation import *
from habitat_ros.controllable_object import *
from habitat_ros.sensors.cameras.rgb_camera import *
from habitat_ros.sensors.cameras.depth_camera import *
from habitat_ros.sensors.cameras.semantic_camera import * 
from habitat_ros.sensors.lidar.lidar import *
from habitat_ros.sensors.lidar.laser import * 

class Robot(ControllableObject):
    def __init__(self, name, yaml_file=None):
        """
        The name is the namespace in the yaml file.
        """
        super().__init__(name, 'map', yaml_file=None)
        
        self.sim = None
        self.cmd_sub = rospy.Subscriber(self.cmd_topic, Twist, self.cmd_cb, queue_size=10)
        self.odom_pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=10)

    def __loadSpec__(self, data):

        path = data["model_path"].split('/')
        pkg_path = rospkg.RosPack().get_path(path[0])
        model_path = data["model_path"][len(path[0])+1:]
        self.model_path =  os.path.join(pkg_path, model_path)
        rospy.logdebug("Load robot model from: " + self.model_path)
        
        self.model_translation = np.asarray(data["translation"])[(1,2,0),]
        self.model_rotation = data["rotation"]

        self.standing_force = data["dynamic"]["standing_force"]
        self.mode = data["dynamic"]["mode"]
        self.friction_coefficient = data["dynamic"]["friction_coefficient"]
        self.angular_damping = data["dynamic"]["angular_damping"]
        self.linear_damping = data["dynamic"]["linear_damping"]

        self.cmd_topic = self.extendTopic(data["cmd_topic"])

        odom = data["odom"]
        self.odom = Odometry()
        self.odom_topic = self.extendTopic(odom["topic"])
        self.odom.header.frame_id = self.extendTopic(odom["frame"])
        self.odom.child_frame_id = self.extendTopic(odom["child_frame"])

    def getVelocity(self):

        lin_vel = self.vel_control.linear_velocity
        ang_vel = self.vel_control.angular_velocity
        lin_vel,_ = y_up2z_up(position=np.asarray([lin_vel[0], lin_vel[1], lin_vel[2]]))
        ang_vel,_ = y_up2z_up(position=np.asarray([ang_vel[0], ang_vel[1], ang_vel[2]]))
        return lin_vel, ang_vel

    def updateOdom(self):

        position, q_coeff = self.getPose()

        self.odom.pose.pose.position.x = position[0]
        self.odom.pose.pose.position.y = position[1]
        self.odom.pose.pose.position.z = position[2]
        self.odom.pose.pose.orientation.x = q_coeff[0]
        self.odom.pose.pose.orientation.y = q_coeff[1]
        self.odom.pose.pose.orientation.z = q_coeff[2]
        self.odom.pose.pose.orientation.w = q_coeff[3]

        lin_vel, ang_vel = self.getVelocity()
        self.odom.twist.twist.linear.x = lin_vel[0]
        self.odom.twist.twist.linear.y = lin_vel[1]
        self.odom.twist.twist.linear.z = lin_vel[2]
        self.odom.twist.twist.angular.x = ang_vel[0]
        self.odom.twist.twist.angular.y = ang_vel[1]
        self.odom.twist.twist.angular.z = ang_vel[2]

    ## Agent config
    def agent_config(self) -> habitat_sim.agent.AgentConfiguration:
        """
        Set up our own agent and agent controls
        """
        make_action_spec = habitat_sim.agent.ActionSpec
        make_actuation_spec = habitat_sim.agent.ActuationSpec
        MOVE, LOOK = 0.07, 1.5

        # all of our possible actions' names
        action_list = [
            "move_left",
            "turn_left",
            "move_right",
            "turn_right",
            "move_backward",
            "look_up",
            "move_forward",
            "look_down",
            "move_down",
            "move_up",
        ]

        action_space: Dict[str, habitat_sim.agent.ActionSpec] = {}

        # build our action space map
        for action in action_list:
            actuation_spec_amt = MOVE if "move" in action else LOOK
            action_spec = make_action_spec(
                action, make_actuation_spec(actuation_spec_amt)
            )
            action_space[action] = action_spec

        sensor_spec: List[habitat_sim.sensor.SensorSpec] = []

        for sensor in self.subnodes:
            sensor.getSensorSpec(sensor_spec)

        agent_config = habitat_sim.agent.AgentConfiguration(
            sensor_specifications=sensor_spec,
            action_space=action_space,
        )
        return agent_config

    def bindSimulator(self, sim):
        self.sim = sim

    def setAgentNode(self, agent_node):
        self.__setNode__(agent_node)
        for sensor in self.subnodes:
            sensor.setSensorNode(self.node)

    ## Load model asset
    def loadModel(self, sim):
        rigid_obj_mgr = sim.get_rigid_object_manager()
        obj_templates_mgr = sim.get_object_template_manager()

        # load the robot asset
        template_id = obj_templates_mgr.load_configs(self.model_path)[0]
        self.model = rigid_obj_mgr.add_object_by_template_id(template_id, self.node)
        self.model.translation = self.model_translation
        self.model.rotation = quat_to_magnum(quat_from_coeffs(self.model_rotation))
        
        self.model.friction_coefficient = self.friction_coefficient
        self.model.angular_damping = self.angular_damping
        self.model.linear_damping = self.linear_damping
        self.vel_control = self.model.velocity_control
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True

        # using the vel controller 
        if self.mode=='legacy':
            self.vel_control.controlling_lin_vel = True
            self.vel_control.controlling_ang_vel = True

    ## Sensors
    def loadSensors(self):
        self.subnodes = []
        for s in self.data['sensors']:

            type=rospy.get_param(s+'/type')
            if type == "RGB_Camera":
                self.subnodes.append(RGBCamera(s, self.frame))
            elif type == "3D_LiDAR":
                self.subnodes.append(LiDAR(s, self.frame))
            elif type == "2D_Laser":
                self.subnodes.append(Laser(s, self.frame))
            elif type == "Depth_Camera":
                self.subnodes.append(DepthCamera(s, self.frame))
            elif type == "Semantic_Camera":
                self.subnodes.append(SemanticCamera(s, self.frame))
            else:
                rospy.logdebug(f"Unknown sensor type {type}.")

    def setVel(self, lin_vel, ang_vel):

        self.model.apply_force(force=mn.Vector3(0.0, -self.standing_force, 0.0), relative_position=mn.Vector3(0.0, 0.0, 0.0))

        if self.mode=='legacy':
            lin_vel,_  = z_up2y_up(position=lin_vel)
            ang_vel,_  = z_up2y_up(position=ang_vel)
            self.vel_control.linear_velocity = lin_vel
            self.vel_control.angular_velocity = ang_vel

        elif self.mode=='dynamic':
            
            lin,ang = self.getVelocity()

            lin_vel = lin_vel-lin
            ang_vel = ang_vel-ang

            vel = self.model.rotation.transform_vector(mn.Vector3(lin_vel[0], lin_vel[2], -lin_vel[1]))*50.0
            self.model.apply_force(force=vel, relative_position=mn.Vector3(0.0, 0.0, 0.0))
            self.model.apply_force(force=mn.Vector3(ang_vel[2], 0.0, 0.0), relative_position=mn.Vector3(0.0, 0.0, 1.0))
            self.model.apply_force(force=mn.Vector3(-ang_vel[2], 0.0, 0.0), relative_position=mn.Vector3(0.0, 0.0, -1.0))


    def cmd_cb(self, msg: Twist):
        self.setVel(
            np.asarray([msg.linear.x,msg.linear.y,msg.linear.z]),
            np.asarray([msg.angular.x,msg.angular.y,msg.angular.z]))

    def publish(self, msg_time=None):
        observation = self.sim.get_sensor_observations()
        for s in self.subnodes:
            s.publish(observation, msg_time)

    def publishOdom(self, msg_time=None):
        self.updateOdom()
        self.odom.header.stamp = rospy.Time.now() if msg_time is None else msg_time
        self.odom_pub.publish(self.odom)

    def update(self, time):
        self.publishTF(time)
        self.publish(time)
        self.publishOdom(time)