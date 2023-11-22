import mujoco
import mujoco_viewer
import numpy as np
from rospkg import RosPack
import time

# MJCFファイルのパス
rospack = RosPack()
model_path = rospack.get_path('tensegrity_slam_sim') + '/models/scene_real_model.xml'  
#model_path = rospack.get_path('tensegrity_slam_sim') + '/models/scene_real_model_fullactuator.xml'  

tension_threshold = -50  # 張力のしきい値
current_tension = 0  # アクチュエータの初期張力

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

#for i in range(model.nu):
#    data.ctrl[i] = current_tension

# Viewerの初期化
viewer = mujoco_viewer.MujocoViewer(model, data)

#data.xfrc_applied[:] = [0.1*np.random.randn(len(data.xfrc_applied[0])) for _ in range(len(data.xfrc_applied))]

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

pid = PIDController(1.0, 0.1, 0.05, 0.002)


while True:
    for i in range(model.nu):
        data.ctrl[i] = pid.step(data.ten_length[i])
        data.ctrl[i] = max(data.ctrl[i], model.actuator_ctrlrange[i, 0])  # 下限値を超えないようにする
        data.ctrl[i] = min(data.ctrl[i], model.actuator_ctrlrange[i, 1])  # 上限値を超えないようにする
    mujoco.mj_step(model, data)
    viewer.render()
    #time.sleep(0.01)
    #print(data.actuator_force)
    #print("tendon_length: ", data.ten_length)
    print("ctrl: ", data.ctrl)

"""
# シミュレーションループ
while True:
    mujoco.mj_step(model, data)  # シミュレーションステップの実行
    viewer.render()  # Viewerでの描画

    # テンドンの張力を取得
    tendon_tensions = data.actuator_force
    print(tendon_tensions)  # デバッグ用に出力

    # 張力のしきい値を設定
    if all(tension < tension_threshold for tension in tendon_tensions):
        # シミュレーション終了後の処理
        print("シミュレーション終了")
        break

    # アクチュエータの制御（ここで張力を徐々に減少させる）
    # ここでは例として全てのアクチュエータの張力を減少させる
    for i in range(model.nu):
        data.ctrl[i] -= 0.001  # 0.1ずつ減少させる
        data.ctrl[i] = max(data.ctrl[i], model.actuator_ctrlrange[i, 0])  # 下限値を超えないようにする

    time.sleep(0.00125)  
"""
