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
data.ten_length[:] = [0.30]*24
qpos_addition = np.random.uniform(-0.02, 0.02, 42)   # TODO:BUG    
""" data.qpos = np.array([-1.18984625e-01,  4.63494792e-04,  2.47213290e-01,  9.82661423e-01,
 -2.74916764e-03,  1.11122860e-02, -1.85055361e-01,  1.37937407e-01,
 -1.15811175e-03,  2.46882063e-01,  9.99695948e-01,  2.19814322e-03,
  2.45588049e-02,  2.10299991e-04,  6.66250341e-03,  1.10618851e-01,
  2.18362927e-01,  6.99977926e-01,  3.64139513e-03,  7.14153931e-01,
  1.34407546e-03,  8.56161190e-03, -1.09258606e-01,  2.21433970e-01,
  6.94595037e-01,  5.56256183e-02,  7.16151973e-01,  3.96216596e-02,
  8.47204181e-03, -1.07714591e-03,  3.47549673e-01,  7.04564028e-01,
  7.09531554e-01, -9.15995917e-03,  8.40233416e-03,  2.45486510e-03,
  3.33814398e-04,  7.05319175e-02,  7.09541174e-01,  7.04166615e-01,
  1.63291203e-02, -2.08341113e-02,
                    ]) + qpos_addition
count = 0
 """
data.ctrl[:] = [-0.0] * 24

while True:
    mujoco.mj_step(model, data)
    #data.ten_length[:] = [0.30] * 24
    viewer.render()
    #time.sleep(0.01)
    print(data.actuator_force)
    #print("tendon_length: ", data.ten_length)
    #print("imu: ", data.sensordata)
    #print("ctrl: ", data.ctrl)
    #print("pos: ", data.qpos)
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