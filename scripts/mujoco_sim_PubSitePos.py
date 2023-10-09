#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np

import rospy
from rospkg import RosPack
from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import JointState, Image, CameraInfo
from geometry_msgs.msg import PointStamped

import mujoco_py

"""
publish each site position of tensegrity robot.
change site position to calculated value later.
"""

class MujocoSim():
    def __init__(self):

        # ROSの初期化
        rospy.init_node('mujoco_sim_PubSitePos')
        self.rospack = RosPack()

        # initialize of model, data, viewer
        model_path = self.rospack.get_path('tensegrity_slam_sim') + '/models/scene_old.xml'
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        viewer = mujoco_py.MjViewer(self.sim)

        self.sim.step()

        # construct site id
        site_id_1t = self.model.site_name2id('link1_top')
        site_id_1b = self.model.site_name2id('link1_bottom')
        site_id_2t = self.model.site_name2id('link2_top')
        site_id_2b = self.model.site_name2id('link2_bottom')
        site_id_3t = self.model.site_name2id('link3_top')
        site_id_3b = self.model.site_name2id('link3_bottom')
        site_id_4t = self.model.site_name2id('link4_top')
        site_id_4b = self.model.site_name2id('link4_bottom')
        site_id_5t = self.model.site_name2id('link5_top')
        site_id_5b = self.model.site_name2id('link5_bottom')
        site_id_6t = self.model.site_name2id('link6_top')
        site_id_6b = self.model.site_name2id('link6_bottom')

        # construct site position message
        site_pos_1t_msg = PointStamped()
        site_pos_1b_msg = PointStamped()
        site_pos_2t_msg = PointStamped()
        site_pos_2b_msg = PointStamped()
        site_pos_3t_msg = PointStamped()
        site_pos_3b_msg = PointStamped()
        site_pos_4t_msg = PointStamped()
        site_pos_4b_msg = PointStamped()
        site_pos_5t_msg = PointStamped()
        site_pos_5b_msg = PointStamped()
        site_pos_6t_msg = PointStamped()
        site_pos_6b_msg = PointStamped()

        # setup publisher
        site_pos_1t_pub = rospy.Publisher('/site_pos_1t', PointStamped, queue_size=1)
        site_pos_1b_pub = rospy.Publisher('/site_pos_1b', PointStamped, queue_size=1)
        site_pos_2t_pub = rospy.Publisher('/site_pos_2t', PointStamped, queue_size=1)
        site_pos_2b_pub = rospy.Publisher('/site_pos_2b', PointStamped, queue_size=1)
        site_pos_3t_pub = rospy.Publisher('/site_pos_3t', PointStamped, queue_size=1)
        site_pos_3b_pub = rospy.Publisher('/site_pos_3b', PointStamped, queue_size=1)
        site_pos_4t_pub = rospy.Publisher('/site_pos_4t', PointStamped, queue_size=1)
        site_pos_4b_pub = rospy.Publisher('/site_pos_4b', PointStamped, queue_size=1)
        site_pos_5t_pub = rospy.Publisher('/site_pos_5t', PointStamped, queue_size=1)
        site_pos_5b_pub = rospy.Publisher('/site_pos_5b', PointStamped, queue_size=1)
        site_pos_6t_pub = rospy.Publisher('/site_pos_6t', PointStamped, queue_size=1)
        site_pos_6b_pub = rospy.Publisher('/site_pos_6b', PointStamped, queue_size=1)
        
        # main loop
        while not rospy.is_shutdown():
            self.sim.step() # step the physics

            # publish site position
            current_time = rospy.Time.now()
            site_pos_1t_msg.header.stamp = current_time
            site_pos_1b_msg.header.stamp = current_time
            site_pos_2t_msg.header.stamp = current_time
            site_pos_2b_msg.header.stamp = current_time
            site_pos_3t_msg.header.stamp = current_time
            site_pos_3b_msg.header.stamp = current_time
            site_pos_4t_msg.header.stamp = current_time
            site_pos_4b_msg.header.stamp = current_time
            site_pos_5t_msg.header.stamp = current_time
            site_pos_5b_msg.header.stamp = current_time
            site_pos_6t_msg.header.stamp = current_time
            site_pos_6b_msg.header.stamp = current_time

            site_pos_1b_msg.header.frame_id = "world"
            site_pos_1t_msg.header.frame_id = "world"
            site_pos_2b_msg.header.frame_id = "world"
            site_pos_2t_msg.header.frame_id = "world"
            site_pos_3b_msg.header.frame_id = "world"
            site_pos_3t_msg.header.frame_id = "world"
            site_pos_4b_msg.header.frame_id = "world"
            site_pos_4t_msg.header.frame_id = "world"
            site_pos_5b_msg.header.frame_id = "world"
            site_pos_5t_msg.header.frame_id = "world"
            site_pos_6b_msg.header.frame_id = "world"
            site_pos_6t_msg.header.frame_id = "world"           

            site_pos_1t_msg.point.x = self.sim.data.site_xpos[site_id_1t][0] 
            site_pos_1t_msg.point.y = self.sim.data.site_xpos[site_id_1t][1] 
            site_pos_1t_msg.point.z = self.sim.data.site_xpos[site_id_1t][2] 
            site_pos_1b_msg.point.x = self.sim.data.site_xpos[site_id_1b][0]
            site_pos_1b_msg.point.y = self.sim.data.site_xpos[site_id_1b][1]
            site_pos_1b_msg.point.z = self.sim.data.site_xpos[site_id_1b][2]
            site_pos_2t_msg.point.x = self.sim.data.site_xpos[site_id_2t][0]
            site_pos_2t_msg.point.y = self.sim.data.site_xpos[site_id_2t][1]
            site_pos_2t_msg.point.z = self.sim.data.site_xpos[site_id_2t][2]
            site_pos_2b_msg.point.x = self.sim.data.site_xpos[site_id_2b][0]
            site_pos_2b_msg.point.y = self.sim.data.site_xpos[site_id_2b][1]
            site_pos_2b_msg.point.z = self.sim.data.site_xpos[site_id_2b][2]
            site_pos_3t_msg.point.x = self.sim.data.site_xpos[site_id_3t][0]
            site_pos_3t_msg.point.y = self.sim.data.site_xpos[site_id_3t][1]
            site_pos_3t_msg.point.z = self.sim.data.site_xpos[site_id_3t][2]
            site_pos_3b_msg.point.x = self.sim.data.site_xpos[site_id_3b][0]
            site_pos_3b_msg.point.y = self.sim.data.site_xpos[site_id_3b][1]
            site_pos_3b_msg.point.z = self.sim.data.site_xpos[site_id_3b][2]
            site_pos_4t_msg.point.x = self.sim.data.site_xpos[site_id_4t][0]
            site_pos_4t_msg.point.y = self.sim.data.site_xpos[site_id_4t][1]
            site_pos_4t_msg.point.z = self.sim.data.site_xpos[site_id_4t][2]
            site_pos_4b_msg.point.x = self.sim.data.site_xpos[site_id_4b][0]
            site_pos_4b_msg.point.y = self.sim.data.site_xpos[site_id_4b][1]
            site_pos_4b_msg.point.z = self.sim.data.site_xpos[site_id_4b][2]
            site_pos_5t_msg.point.x = self.sim.data.site_xpos[site_id_5t][0]
            site_pos_5t_msg.point.y = self.sim.data.site_xpos[site_id_5t][1]
            site_pos_5t_msg.point.z = self.sim.data.site_xpos[site_id_5t][2]
            site_pos_5b_msg.point.x = self.sim.data.site_xpos[site_id_5b][0]
            site_pos_5b_msg.point.y = self.sim.data.site_xpos[site_id_5b][1]
            site_pos_5b_msg.point.z = self.sim.data.site_xpos[site_id_5b][2]
            site_pos_6t_msg.point.x = self.sim.data.site_xpos[site_id_6t][0]
            site_pos_6t_msg.point.y = self.sim.data.site_xpos[site_id_6t][1]
            site_pos_6t_msg.point.z = self.sim.data.site_xpos[site_id_6t][2]
            site_pos_6b_msg.point.x = self.sim.data.site_xpos[site_id_6b][0]
            site_pos_6b_msg.point.y = self.sim.data.site_xpos[site_id_6b][1]
            site_pos_6b_msg.point.z = self.sim.data.site_xpos[site_id_6b][2]

            ##print(site_pos_1t_msg.point)

            site_pos_1t_pub.publish(site_pos_1t_msg)
            site_pos_1b_pub.publish(site_pos_1b_msg)
            site_pos_2t_pub.publish(site_pos_2t_msg)
            site_pos_2b_pub.publish(site_pos_2b_msg)
            site_pos_3t_pub.publish(site_pos_3t_msg)
            site_pos_3b_pub.publish(site_pos_3b_msg)
            site_pos_4t_pub.publish(site_pos_4t_msg)
            site_pos_4b_pub.publish(site_pos_4b_msg)
            site_pos_5t_pub.publish(site_pos_5t_msg)
            site_pos_5b_pub.publish(site_pos_5b_msg)
            site_pos_6t_pub.publish(site_pos_6t_msg)
            site_pos_6b_pub.publish(site_pos_6b_msg)

            viewer.render() # render the scene

if __name__ == '__main__':
    try:
        MujocoSim()
    except rospy.ROSInterruptException:
        pass






