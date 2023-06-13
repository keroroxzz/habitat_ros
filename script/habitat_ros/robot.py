import os
import time
import threading
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
        self.target_vel = Twist()
        self.last_cmd_time = rospy.Time.now()
        self.cmd_sub = rospy.Subscriber(self.cmd_topic, Twist, self.cmd_cb, queue_size=10)
        self.odom_pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=10)

    def __loadSpec__(self, data):

        path = data["model"]["model_path"].split('/')
        pkg_path = rospkg.RosPack().get_path(path[0])
        model_path = data["model"]["model_path"][len(path[0])+1:]
        self.model_path =  os.path.join(pkg_path, model_path)
        rospy.logdebug("Load robot model from: " + self.model_path)

        self.collidable =  data["model"]["collidable"] if 'collidable' in data['model'].keys() else True
        
        self.model_translation = np.asarray(data["translation"])[(1,2,0),]
        self.model_rotation = data["rotation"]

        self.mode = data["dynamic"]["mode"]
        self.navmesh = data["dynamic"]["navmesh"]
        self.navmesh_offset = mn.Vector3(0.0,data["dynamic"]["navmesh_offset"],0.0)
        self.lock_rotation = data["dynamic"]["lock_rotation"]
        self.friction_coefficient = data["dynamic"]["friction_coefficient"]
        self.angular_damping = data["dynamic"]["angular_damping"]
        self.linear_damping = data["dynamic"]["linear_damping"]

        self.height = data["geometric"]["height"]
        self.radius = data["geometric"]["radius"]

        self.cmd_topic = self.extendTopic(data["cmd_topic"])

        odom = data["odom"]
        self.odom = Odometry()
        self.odom_topic = self.extendTopic(odom["topic"])
        self.odom.header.frame_id = self.extendTopic(odom["frame"])
        self.odom.child_frame_id = self.extendTopic(odom["child_frame"])

    def getVelocity(self):

        lin_vel = self.model.linear_velocity
        ang_vel = self.model.angular_velocity
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
            height=self.height,
            radius=self.radius
        )
        return agent_config

    def bindSimulator(self, sim):
        self.sim = sim

    def setAgentNode(self, agent_node):
        self.__setNode__(agent_node)
        for sensor in self.subnodes:
            sensor.setSensorNode(self.node, self.sim._sensors[sensor.uuid])

    ## Load model asset
    def loadModel(self, sim):
        rigid_obj_mgr = sim.get_rigid_object_manager()
        obj_templates_mgr = sim.get_object_template_manager()

        # load the robot asset
        template_id = obj_templates_mgr.load_configs(self.model_path)[0]
        self.model = rigid_obj_mgr.add_object_by_template_id(template_id, self.node)
        self.model.translation = self.model_translation
        self.model.rotation = quat_to_magnum(quat_from_coeffs(self.model_rotation))
        
        # physical parameters
        self.model.collidable = self.collidable
        self.model.restitution_coefficient = 0.1
        self.model.friction_coefficient = self.friction_coefficient
        self.model.angular_damping = self.angular_damping
        self.model.linear_damping = self.linear_damping

        # init model state
        self.prev_state = self.model.rigid_state

    ## Sensors
    def loadSensors(self):
        self.subnodes = []
        for s in self.data['sensors'].values():

            type=s['type']
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

    def updateDynamic(self):

        lin_vel = np.asarray([self.target_vel.linear.x,self.target_vel.linear.y,self.target_vel.linear.z])
        ang_vel = np.asarray([self.target_vel.angular.x,self.target_vel.angular.y,self.target_vel.angular.z])

        # stop exert force if the robot tilts.
        if not self.lock_rotation:
            robot_z = self.model.rotation.transform_vector(mn.Vector3(0.0, 1.0, 0.0)).y
            if np.arccos(robot_z)>0.075:
                return

        if self.mode=='legacy':
            self.model.linear_velocity = self.model.rotation.transform_vector(mn.Vector3(lin_vel[0], lin_vel[2], -lin_vel[1]))
            self.model.angular_velocity = self.model.rotation.transform_vector(mn.Vector3(0.0, ang_vel[2], 0.0))

        elif self.mode=='dynamic':
            
            vel_loc = self.model.rotation.inverted().transform_vector(self.model.linear_velocity)
            vel = self.model.rotation.transform_vector(mn.Vector3(lin_vel[0]-vel_loc.x, 0.0, -lin_vel[1]-vel_loc.z))*50.0

            ang_loc = self.model.rotation.inverted().transform_vector(self.model.angular_velocity)*1.5
            torque = self.model.rotation.transform_vector(mn.Vector3(0.0, ang_vel[2]-ang_loc.y, 0.0))*3.0

            self.model.apply_force(force=vel, relative_position=mn.Vector3(0.0, 0.0, 0.0))
            self.model.apply_torque(torque=torque)

        # constrain the position on the navmesh
        if self.navmesh:
            new_state = self.model.rigid_state
            self.model.translation = self.sim.step_filter(self.prev_state.translation, new_state.translation) + self.navmesh_offset
            self.prev_state = new_state

        # constrain the rotation to be one dimensional
        if self.lock_rotation:
            rot_mat_z = self.model.rotation.to_matrix()
            norm = np.linalg.norm(np.array((rot_mat_z[0,0], rot_mat_z[0,2])))
            rot_mat_z[0,1]=rot_mat_z[1,0]=rot_mat_z[2,1]=rot_mat_z[1,2]=0.0
            rot_mat_z[1,1]=1.0
            rot_mat_z[0,0]/=norm
            rot_mat_z[0,2]/=norm
            rot_mat_z[2,2]=rot_mat_z[0,0]
            rot_mat_z[2,0]=-rot_mat_z[0,2]
            self.model.rotation = mn.Quaternion.from_matrix(rot_mat_z)

    def cmd_cb(self, msg: Twist):
        self.target_vel=msg
        self.last_cmd_time = rospy.Time.now()

    def publish(self):
        # log="\n"
        for s in self.subnodes:
            msg_time = rospy.Time.now()
            # log += f"{s.uuid}: \t\t {1000*s.obs_time:.2f} \t {1000*s.pub_time:.2f} \t {s.hz:.2f} \t {s.hz_:.2f}\n"
            if s.updateObservation(msg_time):
                yield msg_time

        # rospy.logdebug(log)

        if (rospy.Time.now() - self.last_cmd_time).to_sec()>0.2:
            self.target_vel=Twist()

    def publishOdom(self, msg_time=None):
        self.updateOdom()
        self.odom.header.stamp = rospy.Time.now() if msg_time is None else msg_time
        self.odom_pub.publish(self.odom)