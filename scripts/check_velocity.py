import mujoco
import numpy as np
from rospkg import RosPack
import time
from mujoco import viewer


# MJCFファイルのパス
rospack = RosPack()
#model_path = rospack.get_path('tensegrity_slam_sim') + '/models/scene_real_model.xml'  
model_path = rospack.get_path('tensegrity_slam_sim') + '/models/scene_real_model_fullactuator.xml'  

# モデルとシミュレーションのロード
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

data.qpos += np.array([0, 0, 0.5, 0, 0, 0, 0,
                0, 0, 0.5, 0, 0, 0, 0,
                0, 0, 0.5, 0, 0, 0, 0,
                0, 0, 0.5, 0, 0, 0, 0,
                0, 0, 0.5, 0, 0, 0, 0,
                0, 0, 0.5, 0, 0, 0, 0
                ])

model.opt.gravity = [1, 0, -9.8]
# Viewerの初期化
viewer.launch_passive(model, data)

#data.xfrc_applied[:] = [0.1*np.random.randn(len(data.xfrc_applied[0])) for _ in range(len(data.xfrc_applied))]

while True:
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.01)
    #print(data.actuator_force)
    #print("tendon_length: ", data.ten_length)