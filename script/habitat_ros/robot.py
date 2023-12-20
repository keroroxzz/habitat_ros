import os
import time
import threading
import numpy as np
from simple_pid import PID
from typing import Dict, List

# habitat
import magnum as mn
import habitat_sim
from habitat_sim.utils.common import quat_to_magnum, quat_from_coeffs

# ros libs
import rospy
import rospkg
import tf2_ros

# ros messages
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from habitat_ros.utils import *
from habitat_ros.transformation import *
from habitat_ros.controllable_object import *
from habitat_ros.sensors import *   # import all sensors in sensors/__init__.py

class Robot(ControllableObject):
    def __init__(self, name, yaml_file=None):
        """
        The name is the namespace in the yaml file.
        """
        super().__init__(name, 'map', yaml_file=None)
        
        self.sim = None
        self.target_vel = Twist()
        self.last_odom_time = 0.0
        self.last_cmd_time = rospy.Time.now()

        self.pid_x = PID(setpoint=0, output_limits=(-self.maximun_velocity['x'], self.maximun_velocity['x']), **self.linear_x_pid)
        self.pid_y = PID(setpoint=0, output_limits=(-self.maximun_velocity['y'], self.maximun_velocity['y']), **self.linear_y_pid)
        self.pid_z = PID(setpoint=0, output_limits=(-self.maximun_velocity['a'], self.maximun_velocity['a']), **self.angular_pid)

        # tf
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.cmd_sub = rospy.Subscriber(self.cmd_topic, Twist, self.cmd_cb, queue_size=10)
        self.odom_pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=10)

    def __loadSpec__(self, data):

        path = data["model"]["model_path"].split('/')
        pkg_path = rospkg.RosPack().get_path(path[0])
        model_path = data["model"]["model_path"][len(path[0])+1:]
        self.model_path =  os.path.join(pkg_path, model_path)

        if os.path.exists(self.model_path):
            rospy.logdebug("Load robot model from: " + self.model_path)
        else:
            rospy.logerr("Path to robot model not found: " + self.model_path)

        self.collidable =  data["model"]["collidable"] if 'collidable' in data['model'].keys() else True
        
        self.model_translation = np.asarray(data["position"])[(1,2,0),]
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

        self.cmd_topic = self.extendTopic(data["control"]["cmd_topic"])
        self.angular_pid = data["control"]["angular_pid"]
        self.linear_x_pid = data["control"]["linear_x_pid"]
        self.linear_y_pid = data["control"]["linear_y_pid"]
        self.maximun_velocity = data["control"]["maximun_velocity"]

        odom = data["odom"]
        self.odom = Odometry()
        self.odom_topic = self.extendTopic(odom["topic"])
        self.odom.header.frame_id = self.extendTopic(odom["frame"])
        self.odom.child_frame_id = self.extendTopic(odom["child_frame"])
        self.odom_ang = np.zeros(3, dtype=float)

        if 'vel_cov' in odom.keys():
            self.cov = np.array(odom['vel_cov']).reshape(6,6)
        else:
            self.cov = np.identity(6, dtype=float) + np.clip(0.1*np.random.randn(6,6)*(1.0-np.identity(6, dtype=float)), -0.1,0.1)

        self.odom.twist.covariance = self.cov.reshape(-1)

    def getVelocity(self):

        ang_vel = self.model.angular_velocity

        # ground truth local velocity
        vel_loc = self.model.rotation.inverted().transform_vector(self.model.linear_velocity)

        # note that habitat is y-up
        state = np.array((vel_loc.x, -vel_loc.z, 0.0, 0.0, 0.0, ang_vel.y), dtype=float)
        new_state = np.matmul(state, self.cov)

        return new_state

    def updateOdom(self):

        dt = self.sim.get_world_time()-self.last_odom_time

        self.odom_ang[0] += self.odom.twist.twist.angular.x*dt
        self.odom_ang[1] += self.odom.twist.twist.angular.y*dt
        self.odom_ang[2] += self.odom.twist.twist.angular.z*dt
        q_coeff = tfs.quaternion_from_euler(self.odom_ang[0], self.odom_ang[1], self.odom_ang[2], "sxyz")
        
        dx = dt*(np.cos(self.odom_ang[2])*self.odom.twist.twist.linear.x - np.sin(self.odom_ang[2])*self.odom.twist.twist.linear.y)
        dy = dt*(np.sin(self.odom_ang[2])*self.odom.twist.twist.linear.x + np.cos(self.odom_ang[2])*self.odom.twist.twist.linear.y)

        self.odom.pose.pose.position.x += dx
        self.odom.pose.pose.position.y += dy
        self.odom.pose.pose.position.z = 0.0
        self.odom.pose.pose.orientation.x = q_coeff[0]
        self.odom.pose.pose.orientation.y = q_coeff[1]
        self.odom.pose.pose.orientation.z = q_coeff[2]
        self.odom.pose.pose.orientation.w = q_coeff[3]

        v = self.getVelocity()
        self.odom.twist.twist.linear.x = v[0]
        self.odom.twist.twist.linear.y = v[1]
        self.odom.twist.twist.linear.z = 0.0
        self.odom.twist.twist.angular.x = 0.0
        self.odom.twist.twist.angular.y = 0.0
        self.odom.twist.twist.angular.z = v[5]

        self.last_odom_time = self.sim.get_world_time()

        if not self.is_pub_tf:
            tfBroadcaster.sendTransform(
                (self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z), 
                q_coeff, 
                rospy.Time.now(), 
                self.frame, 
                self.odom.header.frame_id)

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
        self.last_odom_time = self.sim.get_world_time()

    def setAgentNode(self, agent_node):
        self.__setNode__(agent_node)
        for sensor in self.subnodes:
            sensor.setSensorNode(self.node, self.sim._sensors[sensor.uuid])

    ## Load model asset
    def loadModel(self, sim):


        rigid_obj_mgr = sim.get_rigid_object_manager()
        obj_templates_mgr = sim.get_object_template_manager()

        # load the robot asset
        template_id = obj_templates_mgr.load_configs(self.model_path)
        
        if len(template_id) == 0:
            rospy.logerr(f"Failed to load model from {self.model_path}.")
            exit()

        template_id = template_id[0]

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
            
            # dynamic loading of sensor class
            if type in globals().keys():
                self.subnodes.append(globals()[type](s, self.frame))
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
            
            # set the target
            self.pid_x.setpoint = lin_vel[0]
            self.pid_y.setpoint = -lin_vel[1]
            self.pid_z.setpoint = ang_vel[2]

            vel_loc = self.model.rotation.inverted().transform_vector(self.model.linear_velocity)
            vel = self.model.rotation.transform_vector(
                mn.Vector3(
                self.pid_x(vel_loc.x), 
                0.0, 
                self.pid_y(vel_loc.z)))

            ang_loc = self.model.rotation.inverted().transform_vector(self.model.angular_velocity)
            torque = self.model.rotation.transform_vector(
                mn.Vector3(
                0.0, 
                self.pid_z(ang_loc.y), 
                0.0))
            
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