import rospy
from std_msgs.msg import Float64
import mujoco_py

def mujoco_ros_publisher():
    # ROSの初期化
    rospy.init_node('mujoco_tendon_length_publisher', anonymous=True)

    # トピックを設定 (例: /tendon_force)
    pub = rospy.Publisher('/tendon_length', Float64, queue_size=10)

    # MuJoCoのモデルとシミュレーションの初期化
    model = mujoco_py.load_model_from_path("../models/scene.xml")
    sim = mujoco_py.MjSim(model)

    rate = rospy.Rate(10)  # 10Hzの更新レート

    while not rospy.is_shutdown():
        # シミュレーションを1ステップ実行
        sim.step()

        # テンドンの張力を取得
        tendon_id = model.tendon_name2id("link1t_3b")
        tendon_length = sim.data.ten_length[tendon_id]
        print(f"Length of link1t_3b: {tendon_length}")

        # ROSトピックとして発行
        pub.publish(tendon_length)

        rate.sleep()

if __name__ == '__main__':
    try:
        mujoco_ros_publisher()
    except rospy.ROSInterruptException:
        pass
