# Habitat_ROS Configuration

This documentation is about the definition of configuring sensors and robot.

## Basic Structure

The configurarion use the yaml format, the indent represents the hierachy:

    fruit:
      name: "apple"
      size:
        height: 0.07
        horizontal_radius: 0.04

    water:
        (more stuff)...

Each item is either a namespace or property.

The "fruit" and "size" are namespaces, while the "name", "height", and "horizontal_radius" are property.

To specify a property/namespace, we need to include the whole hierachy.

For example:

the param name of the property "name" in rosparam system is :

    fruit/name

the param name of the property "height" in rosparam system is :

    fruit/size/height

## How to load a Robot?

### robot_name
The root property "robot_name" specifies the namespace defining the robot spec.

For example, habitat_ros will find all the parameters belonging to the namespace "oreo" with the following yaml:
    
    robot_name: "oreo"

    oreo:
        type: ...
        name: ...

## How to attach a sensor?

The system automatically scan all the sub-namespaces belonging to {robot_name}/sensors/.

So you can either directly declare sensor information in the robot yaml file.

Or, you can wrtie the sensor information seperatly and loaded to a specific namespace in launch file as following:

    # load robot setting
    <rosparam command="load" file="$(find habitat_ros)/config/oreo.yaml"/>

    # attach VLP_16 to the robot named oreo by setting the namespace as "oreo/sensors/".
    <rosparam command="load" file="$(find habitat_ros)/config/VLP_16.yaml" ns="oreo/sensors/"/>


## Controllable Object

A Controllable Object could be the robot or a sensor attached to the robot.

Every object must have the following properties:

### 1. type
The "type" defines the python class to initiate the object instance.

A robot must be the type "Robot":

    type: "Robot" 

Every sensor also has the property "type" to decide using which class to initaite the sensor instance.

The code segment to dynamically load the sensor is at 
script/habitat_ros/robot.py#Robot.loadSensors()

Currently, there are: "Robot", "DepthCamera", "RGBCamera", "SemanticCamera", "LiDAR", "Laser"

You can also define your own sensor class in habitat_ros/script/habitat_ros/sensors/... 
and import it in habitat_ros/script/habitat_ros/sensors/__init__.py.

The habitat_ros will automatically load them as available types.

### 2. Basic properties

    name: "oreo"  # name of the instance, must be unique

### 3. TF Information

The name of the coordinate frame of the object:

    frame: "/base_link"

The pose of the object in the parent frame. For a robot, this is its initail pose.

    position: [0.0, 0.0, 0.5]
    rotation: [0.0, 0.0, 0.0, 1.0]

Whether the object should publish the ground truth TF between the parent and object frame. For a robot type, switch to true to publish ground-truth tf between map and base_link.

    publish_transfromation: false

Extra static chiled frames to be published, note that their parent is this object's frame.

    tf_links:
        base_footprint:
            frame: "/base_footprint"
            position: [0.0, 0.0, 0.0]
            orientation: [0.0, 0.0, 0.0, 1.0]

## Robot 

This section is about the robot-specifiec properties.

### 1. Dynamic

Robot dynamic properties defined the physical behavior of the robot. They all belong to the namespace "dynamic".

#### Dynamic Mode
"dynamic" : using pysical simulation
"legacy" : overwrite velocity 

    mode: "dynamic"/"legacy"

#### Navmesh

Define the properties of the interaction between robot and navmesh.

If turned on, the robot root position will be limited to move on the navemesh only.
You can turn on lock_rotation to lock the rotation in x and y axis to prevent falling, 

    navmesh: true
    navmesh_offset: -0.05
    lock_rotation: true 

#### Phyical Properties

friction_coefficient: friction  of the robot wheel, note that the robot wheel can not spin so that this should be small enough to enable moving for dynamic mode.

angular_damping: the damping factor for angular movement, a value that is too small could lead to simulation error.

linear_damping: this could prevent the simulation from exploding, a value that is too small could lead to simulation error.

    friction_coefficient: 0.05
    angular_damping: 0.2
    linear_damping: 0.2

robot dimension properties for navmesh generation and collision, will be overwritten if collidable = True

### 2. Geometry

#### Geometry
The default geometry of the robot, if not import 3D model as collision model, this will be the collision body for the robot.

    geometric:
        height: 1.0
        radius: 0.24

#### 3D model

model_path: folder path of where your robot model is, {pkg_name}/robot/{robot_name}

collidable: use this model as the collision model.

    model:
        model_path: "habitat_ros/robot/oreo/oreo.object_config.json" 
        collidable: false  # recomment false if navmesh = true

### 3. Control

_pid: PID controller gain for torque control

maximun_velocity: max force and torque to apply on the robot [newton, newton, newton*meter]

    control:
        cmd_topic: "/cmd_vel" 
        angular_pid: {"Kp":1.0, "Ki":0.1, "Kd":0.05} 
        linear_x_pid: {"Kp":40.0, "Ki":10.0, "Kd":0.5}
        linear_y_pid: {"Kp":40.0, "Ki":10.0, "Kd":0.5}
        maximun_velocity: {"x":50.0, "y":50.0, "a":1.0} 

### 4. Odometry

The odometry configuration.

vel_cov: the covariance matrix of robot state, currently randomly biased, can be modified if you want to simulate real imperfection of the robot. The robot state is [vx vy vz wx wy wz].

    odom:
        topic: "/odometry_odm"
        frame: "/odom"
        child_frame: "/base_link"

        vel_cov: [
        [1.0,0.03,0.0,0.0,0.0,-0.02],
        [0.03,1.0,0.0,0.0,0.0,0.06],
        [0.0,0.0,1.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,1.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,1.0,0.0],
        [-0.02,0.06,0.0,0.0,0.0,1.0],]

## Sensor

### 1. Actuator

Define the actuation topics and corresponding actions in two corresponding lists.

The following sensor will listen to the ROS topic "/camera/tilt" to control the action "looks_up" in habitat sim.

You can define multiple actions and topics.

    actuation:
        topics: ["/camera/tilt", "another topic",...]
        actions: ["look_up", "another action",...]


### 2. Publish Topic & Frame

topic: the topic of the current sensor to be published

topic_frame: (optional) if setted, overwrite the topic frame name.

    topic: "/zed2/zed_node/depth/depth_registered"
    topic_frame: "/camera" # explicitly set the msg's frame

### 3. Sensor Info

Sensor specific information to be used by the sensor class.

It depends on each type of sensor or your own implementation.