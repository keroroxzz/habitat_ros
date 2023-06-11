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
    return np.clip(np.random.randn(src.shape[0], src.shape[1])*mean, -max, max) + src