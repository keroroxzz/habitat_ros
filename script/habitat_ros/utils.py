import numpy as np
import numba as nb

# habitat
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_angle_axis

# ros libs
import rospkg
import tf

# ros messages
from cv_bridge import CvBridge

bridge = CvBridge()
tfBroadcaster = tf.TransformBroadcaster()
pkg_path = rospkg.RosPack().get_path("habitat_ros")

## Utils
def y_up2z_up(position=None, rotation=None):
    """
    Transfrom the coordinate to z-up
    """

    if position is not None:
        position = position[(0,2,1),]
        position[1] *= -1.0
    
    if rotation is not None:
        theta, w = quat_to_angle_axis(rotation)
        w = w[(0,2,1),]
        w[1] *= -1.0
        rotation = quat_from_angle_axis(theta, w)
    return position, rotation

def z_up2y_up(position=None, rotation=None):
    """
    Transfrom the coordinate back to y-up
    """

    if position is not None:
        position = position[(0,2,1),]
        position[2] *= -1.0
    
    if rotation is not None:
        theta, w = quat_to_angle_axis(rotation)
        w = w[(0,2,1),]
        w[2] *= -1.0
        rotation = quat_from_angle_axis(theta, w)

    return position, rotation

def nodeTranslationToNumpy(translation):
    """
    Translate the translation of node into nparray
    """

    return np.asarray([translation.x, translation.y, translation.z])

@nb.jit(nopython=True)
def noise_numba(src: np.ndarray, mean, max):
    return np.clip(np.random.randn(*src.shape).astype(src.dtype)*mean, -max, max) + src

@nb.jit(nopython=True)
def raw_to_laser_numba(raw: np.ndarray, cor: np.ndarray, min, max):

    d = raw/cor
    d[d<min] = 0.0
    d[d>max] = 0.0

    return d

@nb.jit(nopython=True)
def raw_to_lidar_numba(raw: np.ndarray = np.array([[]]), vec: np.ndarray = np.array([[[]]]), cos:np.ndarray = np.array([[]]), min=0.0, max=100.0):

    p = raw.reshape(raw.shape[0],raw.shape[1],1)*vec
    d = raw/cos
    id = np.bitwise_and(d>=min,d<=max).nonzero()

    return p, id

@nb.jit()
def lidar_correction(hfov, vfov, hres, vres):

    x = np.float32(np.mgrid[0:vres, 0:hres])
    x[1] = -(x[1]/hres - 0.5)*hfov
    x[0] = -(x[0]/vres - 0.5)*vfov
    x *= np.pi/180.0

    rx = np.cos(x[1])*np.cos(x[0])
    ry = np.sin(x[1])*np.cos(x[0])
    rz = np.sin(x[0])

    ax = np.abs(rx)
    ay = np.abs(ry)
    az = np.abs(rz)

    dot = np.max(np.stack((ax, ay, az), axis=2), axis=2)
    vector = np.stack((rx, ry, rz), axis=2)/dot.reshape(dot.shape[0],dot.shape[1],1)

    return dot, vector