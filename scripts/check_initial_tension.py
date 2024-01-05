import mujoco
import mujoco_viewer
import numpy as np
from rospkg import RosPack
import time

# MJCFファイルのパス
rospack = RosPack()
#model_path = rospack.get_path('tensegrity_slam_sim') + '/models/scene_real_model.xml'  
model_path = rospack.get_path('tensegrity_slam_sim') + '/models/scene_real_model_fullactuator_no_stiffness.xml'  

current_tension = -5  # アクチュエータの初期張力

# モデルとシミュレーションのロード
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

for i in range(model.nu):
    data.ctrl[i] = current_tension

# Viewerの初期化
viewer = mujoco_viewer.MujocoViewer(model, data)

model.opt.gravity = [0.0, 0, -9.8]
data.ten_length[:] = 0.30
data.qpos = np.array([0.14717668,  0.14711882,  0.15701801,  0.86432397, -0.40548401,  0.2194443,
                        -0.20092532,  0.350647,    0.11930152,  0.06542414,  0.79409071, -0.2563381,
                        0.54759233,  0.06207542,  0.22993135,  0.20179415,  0.06074503,  0.50408641,
                        -0.1424163,   0.77721656, -0.34863865,  0.27766309,  0.00355943,  0.15893443,
                        0.39771177, -0.1131317,   0.89793995, -0.1507661,   0.35460463,  0.19562937,
                        0.15674695,  0.86554097,  0.36059424,  0.23697669, -0.25426887,  0.19753333,
                        0.03760321,  0.07496286,  0.74249165,  0.51360453,  0.28749698, -0.31978435,
                    ])
    
count = 0

while True:
    mujoco.mj_step(model, data)
    data.ten_length[:] = [0.30] * 24
    viewer.render()
    #time.sleep(0.01)
    #print(data.actuator_force)
    #print("tendon_length: ", data.ten_length)
    #print("imu: ", data.sensordata)
    #print("ctrl: ", data.ctrl)
    print("pos: ", data.qpos)
    count += 1
    time.sleep(0.01)

"""
## PID control
class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.desired_tendon_length_min = 0.28
        self.desired_tendon_length_max = 0.29
        self.reset()
    
    def reset(self):
        self.integral = 0
        self.previous_error = 0
    
    def step(self, tendon_length):
        if tendon_length < self.desired_tendon_length_max and tendon_length > self.desired_tendon_length_min:
            error = 0
        elif tendon_length > self.desired_tendon_length_max:
            error = tendon_length - self.desired_tendon_length_max
        else:
            error = tendon_length - self.desired_tendon_length_min
        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

pid = PIDController(20.0, 0.1, 0.05, 0.002)


while True:
    for i in range(model.nu):
        data.ctrl[i] = pid.step(data.ten_length[i])
        data.ctrl[i] = max(data.ctrl[i], model.actuator_ctrlrange[i, 0])  # 下限値を超えないようにする
        data.ctrl[i] = min(data.ctrl[i], model.actuator_ctrlrange[i, 1])  # 上限値を超えないようにする
    mujoco.mj_step(model, data)
    viewer.render()
    #time.sleep(0.01)
    #print(data.actuator_force)
    print("tendon_length: ", data.ten_length)
    #print("ctrl: ", data.ctrl)
    """