#!/usr/bin/env python

import rospy
import tf
from geometry_msgs.msg import TransformStamped

def broadcast_world_frame():
    rospy.init_node('world_frame_broadcaster')
    
    # TransformBroadcasterのインスタンスを作成
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10)  # 10Hz

    while not rospy.is_shutdown():
        # 現在のROS時刻を取得
        current_time = rospy.Time.now()

        # "world"から"base_link"への変換をbroadcastします。
        # 位置は(0, 0, 0)、回転は四元数(0, 0, 0, 1)で、実質的に変換はありません。
        br.sendTransform(
            (0.0, 0.0, 0.0),  # 位置 (x, y, z)
            (0.0, 0.0, 0.0, 1.0),  # 四元数 (x, y, z, w)
            current_time,   # タイムスタンプ
            "world",    # child frame (target frame of the transform)
            "map"         # parent frame (source frame of the transform)
        )

        rate.sleep()

if __name__ == '__main__':
    broadcast_world_frame()
