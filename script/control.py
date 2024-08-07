#! /usr/bin/env python3

import sys
import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QGridLayout, QLabel

# ros libs
import rospy
from cv_bridge import CvBridge

# ros messages
from std_msgs.msg import Float32
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Twist
# from habitat_ros.utils import *

bridge = CvBridge()

class MouseController(QDialog):
    def __init__(self, rate, app, point) -> None:
        self.QApp = app
        super().__init__()
        self.initUI()

        if point is not None:
            self.move(point.x(), point.y())
        
        # add ros node, pub, sub
        rospy.init_node('MouseController', anonymous=True)
        self.reset = False
        self.rate = rospy.Rate(rate)
        self.img = None
        self.pmouse = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.loop)
        self.timer.start(1000)

        self.cmd_pub = CmdPub(rate)

        topic = rospy.get_param("camera_topic", default="fake_camera")
        rospy.Subscriber(topic+'/compressed', CompressedImage, self.img_cb, queue_size=1)
        rospy.Subscriber(topic, Image, self.img_cb, queue_size=1)

        self.cmd_pub.start()

    def initUI(self):
        self.resize(400, 300)
        
        self.label = QLabel(self)
        layout = QGridLayout(self)
        layout.addWidget(self.label, 0, 0, 0, 0, alignment=Qt.AlignCenter)

    def exec(self):
        return self.QApp.exec_()

    def showImage(self, img):
        height, width, channel = img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(self.qImg))

    def loop(self):
        if rospy.is_shutdown():
            print("ROS is shut down! Exit...")
            self.reset=False
            self.QApp.closeAllWindows()

    def img_cb(self, msg):
        if isinstance(msg, Image):
            self.img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        elif isinstance(msg, CompressedImage):
            self.img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.img = cv2.resize(self.img, (int(self.width()*0.9), int(self.height()*0.9)))
        self.showImage(self.img)
        self.update()
        self.QApp.processEvents()

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        self.pmouse = a0.pos()
        return super().mousePressEvent(a0)

    def mouseMoveEvent(self, a0: QtGui.QMouseEvent) -> None:
        # print(a0.pos())
        if self.pmouse is not None:
            mm = a0.pos()-self.pmouse
            self.cmd_pub.setMouseMovement(mm.x(), mm.y())
        self.pmouse = a0.pos()
        return super().mouseMoveEvent(a0)

    def restartApplication(self):
        print("Restart Application...")
        self.reset=True
        self.QApp.closeAllWindows()

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        self.cmd_pub.setKey(a0.key(), 1)

        if a0.key()==79: # press 'q' to restart the application if stuck
            self.restartApplication()

        return super().keyPressEvent(a0)
    
    def keyReleaseEvent(self, a0: QtGui.QKeyEvent) -> None:
        self.cmd_pub.setKey(a0.key(), 0)
        return super().keyReleaseEvent(a0)

class CmdPub(QThread):
    def __init__(self, rate):
        super().__init__()
        self.cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.tilt_pub = rospy.Publisher("camera/tilt", Float32, queue_size=10)
        self.rate = rospy.Rate(rate)

        self.mx = 0.0
        self.my = 0.0
        self.keys = np.zeros(256, dtype=int)
        self.cmd_vel = Twist()
        self.reset = False

    def setMouseMovement(self, x, y):
        self.mx = x*2.0
        self.my = -y

    def setKey(self, key, status):
        if key>-1 and key<self.keys.shape[0]:
            self.keys[key]=status

    def valid(self, cmd):
        v = np.linalg.norm(np.asarray([cmd.linear.x,
        cmd.linear.y,
        cmd.linear.z]))

        a = np.linalg.norm(np.asarray([cmd.angular.x,
        cmd.angular.y,
        cmd.angular.z]))
        return v+a>0.1

    def pub_cmd(self):
        cmd = Twist()
        if self.keys[87] == 1: # W
            cmd.linear.x += 0.8
        if self.keys[83] == 1: # S
            cmd.linear.x += -0.8
        if self.keys[65] == 1: # A
            cmd.linear.y += 0.5
        if self.keys[68] == 1: # D
            cmd.linear.y += -0.5
        if self.keys[81] == 1: # Q
            cmd.angular.z += 1.0
        if self.keys[69] == 1: # E
            cmd.angular.z -= 1.0

        cmd.angular.z += -self.mx

        # boost the speed by pressing space
        if self.keys[32] == 1:
            cmd.linear.x *= 2.0
            cmd.linear.y *= 2.0
            cmd.angular.z *= 3.0

        self.cmd_vel.linear.x = self.cmd_vel.linear.x*0.75 + cmd.linear.x*0.25
        self.cmd_vel.linear.y = self.cmd_vel.linear.y*0.75 + cmd.linear.y*0.25
        self.cmd_vel.linear.z = self.cmd_vel.linear.z*0.75 + cmd.linear.z*0.25

        self.cmd_vel.angular.x = self.cmd_vel.angular.x*0.5 + cmd.angular.x*0.5
        self.cmd_vel.angular.y = self.cmd_vel.angular.y*0.5 + cmd.angular.y*0.5
        self.cmd_vel.angular.z = self.cmd_vel.angular.z*0.5 + cmd.angular.z*0.5

        if self.valid(self.cmd_vel):
            self.cmd_pub.publish(self.cmd_vel)
            self.reset = False
        elif not self.reset:
            self.cmd_vel = Twist()
            self.cmd_pub.publish(self.cmd_vel)
            self.reset = True

        self.tilt_pub.publish(self.my)
        self.mx = 0.0
        self.my *= 0.2

    def run(self):
        while not rospy.is_shutdown():
            self.pub_cmd()
            self.rate.sleep()

if __name__ == "__main__":

    app = QApplication(sys.argv)
    point = None
    controller = None
    while controller is None or controller.reset:
        controller = MouseController(60, app, point)
        controller.show()
        controller.exec()
        point = controller.pos()
