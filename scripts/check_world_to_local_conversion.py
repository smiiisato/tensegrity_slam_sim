import os
import mujoco
import mujoco_viewer
import numpy as np
from rospkg import RosPack
import time
from scipy.spatial.transform import Rotation

# MJCFファイルのパス
rospack = RosPack()
#model_path = rospack.get_path('tensegrity_slam_sim') + '/models/scene_real_model.xml'  
model_path = rospack.get_path('tensegrity_slam_sim') + '/models/scene_real_model_fullactuator.xml'  

# モデルファイルのロード
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

def vector2mat(vector):
    mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mat[0, :] = vector
    mat[1, :] = np.cross(vector, np.array([0, 0, 1]))
    mat[2, :] = np.cross(mat[0, :], mat[1, :])
    return mat

# ビューアの作成
viewer = mujoco_viewer.MujocoViewer(model, data)
mujoco.mj_step(model, data)

while True:

    # 矢印を描画
    viewer.add_marker(
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        rgba=[1, 0, 0, 1],
        size=[0.01, 0.01, 1],
        pos=data.xpos[0],
        mat=np.array([0., 0., 1., 0., 1., 0., -1., 0., 0.])
    )
    viewer.add_marker(
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        rgba=[0, 1, 0, 1],
        size=[0.01, 0.01, 1],
        pos=data.xpos[0],
        mat=np.dot(data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link1")].reshape(3, 3).T, np.array([0., 0., 1., 0., 1., 0., -1., 0., 0.]).reshape(3,3))
    )
    # ビューアの更新
    mujoco.mj_step(model, data)
    viewer.render()