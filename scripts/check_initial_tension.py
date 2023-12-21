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

model.opt.gravity = [0.0, 0, -0.5]
count = 0

while True:
    if count > 1000:
        model.opt.gravity = [0.0, 0, -9.8]
    mujoco.mj_step(model, data)
    viewer.render()
    #time.sleep(0.01)
    #print(data.actuator_force)
    #print("tendon_length: ", data.ten_length)
    print("ctrl: ", data.ctrl)
    count += 1

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