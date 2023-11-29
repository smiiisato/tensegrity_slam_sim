import mujoco
import numpy as np
from rospkg import RosPack
import time
import mujoco.viewer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

average_velocity = np.zeros(6)
angular_momentum = np.zeros(3)
angular_momentum_2 = np.zeros(3)

fig1, ax1 = plt.subplots()
xdata1, ydata1 = [], []
ln1, = plt.plot([], [], 'r-', animated=True)

def init1():
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-4, 4)
    ax1.set_ylabel("roll_velocity")
    return ln1,

def update1(frame):
    xdata1.append(frame)
    ydata1.append(average_velocity[3])
    ln1.set_data(xdata1, ydata1)
    return ln1,

ani1 = FuncAnimation(fig1, update1, frames=np.linspace(0, 100, 1000),
                    init_func=init1, blit=True)

fig2, ax2 = plt.subplots()
xdata2, ydata2 = [], []
ln2, = plt.plot([], [], 'r-', animated=True)

def init2():
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-4, 4)
    ax2.set_ylabel("pitch_velocity")
    return ln2,

def update2(frame):
    xdata2.append(frame)
    ydata2.append(average_velocity[4])
    ln2.set_data(xdata2, ydata2)
    return ln2,

ani2 = FuncAnimation(fig2, update2, frames=np.linspace(0, 100, 1000),
                    init_func=init2, blit=True)

fig3, ax3 = plt.subplots()
xdata3, ydata3 = [], []
ln3, = plt.plot([], [], 'r-', animated=True)

def init3():
    ax3.set_xlim(0, 100)
    ax3.set_ylim(-4, 4)
    ax3.set_ylabel("yaw_velocity")
    return ln3,

def update3(frame):
    xdata3.append(frame)
    ydata3.append(average_velocity[5])
    ln3.set_data(xdata3, ydata3)
    return ln3,

ani3 = FuncAnimation(fig3, update3, frames=np.linspace(0, 100, 1000),
                    init_func=init3, blit=True)

"""
fig4, ax4 = plt.subplots()
xdata4, ydata4 = [], []
ln4, = plt.plot([], [], 'r-', animated=True)

def init4():
    ax4.set_xlim(0, 100)
    ax4.set_ylim(-0.1, 0.1)
    ax4.set_xlabel("time [s]")
    ax4.set_ylabel("angular momentum[x] [kg m^2/s]")
    return ln4,

def update4(frame):
    xdata4.append(frame)
    ydata4.append(angular_momentum[0])
    ln4.set_data(xdata4, ydata4)
    return ln4,

ani4 = FuncAnimation(fig4, update4, frames=np.linspace(0, 100, 1000),
                    init_func=init4, blit=True)
"""
fig5, ax5 = plt.subplots()
xdata5, ydata5 = [], []
ln5, = plt.plot([], [], 'r-', animated=True)

def init5():
    ax5.set_xlim(0, 100)
    ax5.set_ylim(-0.1, 0.1)
    ax5.set_xlabel("time [s]")
    ax5.set_ylabel("angular momentum[y] [kg m^2/s]")
    return ln5,

def update5(frame):
    xdata5.append(frame)
    ydata5.append(angular_momentum[1])
    ln5.set_data(xdata5, ydata5)
    return ln5,

ani5 = FuncAnimation(fig5, update5, frames=np.linspace(0, 100, 1000),
                    init_func=init5, blit=True)
"""
fig6, ax6 = plt.subplots()
xdata6, ydata6 = [], []
ln6, = plt.plot([], [], 'r-', animated=True)

def init6():
    ax6.set_xlim(0, 100)
    ax6.set_ylim(-0.1, 0.1)
    ax6.set_xlabel("time [s]")
    ax6.set_ylabel("angular momentum[z] [kg m^2/s]")
    return ln6,

def update6(frame):
    xdata6.append(frame)
    ydata6.append(angular_momentum[2])
    ln6.set_data(xdata6, ydata6)
    return ln6,

ani6 = FuncAnimation(fig6, update6, frames=np.linspace(0, 100, 1000),
                    init_func=init6, blit=True)
"""
fig7, ax7 = plt.subplots()
xdata7, ydata7 = [], []
ln7, = plt.plot([], [], 'r-', animated=True)

def init7():
    ax7.set_xlim(0, 100)
    ax7.set_ylim(-0.1, 0.1)
    ax7.set_xlabel("time [s]")
    ax7.set_ylabel("angular momentum 2[y] [kg m^2/s]")
    return ln7,

def update7(frame):
    xdata7.append(frame)
    ydata7.append(angular_momentum_2[1])
    ln7.set_data(xdata7, ydata7)
    return ln7,

ani7 = FuncAnimation(fig7, update7, frames=np.linspace(0, 100, 1000),
                    init_func=init7, blit=True)


# MJCFファイルのパス
rospack = RosPack()
#model_path = rospack.get_path('tensegrity_slam_sim') + '/models/scene_real_model.xml'  
model_path = rospack.get_path('tensegrity_slam_sim') + '/models/scene_real_model_fullactuator.xml'  

# モデルとシミュレーションのロード
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

data.qpos += np.array([0, 0, 0.025, 0, 0, 0, 0,
                0, 0, 0.025, 0, 0, 0, 0,
                0, 0, 0.025, 0, 0, 0, 0,
                0, 0, 0.025, 0, 0, 0, 0,
                0, 0, 0.025, 0, 0, 0, 0,
                0, 0, 0.025, 0, 0, 0, 0
                ])

model.opt.gravity = [2.4, 0, -5.0]
# Viewerの初期化
viewer = mujoco.viewer.launch_passive(model, data)

#data.xfrc_applied[:] = [0.1*np.random.randn(len(data.xfrc_applied[0])) for _ in range(len(data.xfrc_applied))]

plt.show(block=False)

while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.sync()
    qvel = np.vstack([data.qvel[0:6],
                     data.qvel[6:12],
                     data.qvel[12:18],
                     data.qvel[18:24],
                     data.qvel[24:30],
                     data.qvel[30:36]])
    average_velocity = np.mean(qvel, axis=0)
    print("average_velocity: ", average_velocity)
    fig1.canvas.draw()
    fig1.canvas.flush_events()

    ## calculate angular momentum
    body_xpos = np.vstack((
                    data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "link1")],
                    data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "link2")],
                    data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "link3")],
                    data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "link4")],
                    data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "link5")],
                    data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "link6")],
                    ))
    current_body_xpos = np.mean(body_xpos, axis=0) ## (3,)
    angular_momentum = np.zeros(3)
    angular_momentum_2 = np.zeros(3)
    com_qvel = average_velocity[0:3]
    com_xpos = current_body_xpos

    for i in range(6):
        body_mass = model.body_mass[i]
        body_com_xpos = body_xpos[i]
        body_vel = qvel[i][0:3]
        angular_momentum += body_mass*np.cross(body_com_xpos-com_xpos, body_vel-com_qvel) # m_i * (x_i-x_com) x (v_i-v_com)
        angular_momentum_2 += body_mass*np.cross(body_com_xpos, body_vel) # m_i * (x_i) x 
    angular_momentum_2 -= 6*body_mass*np.cross(com_xpos, com_qvel)
    print("angular_momentum: ", angular_momentum)
        
    """
    fig2.canvas.draw()
    fig2.canvas.flush_events()
    fig3.canvas.draw()
    fig3.canvas.flush_events()
    """
    #time.sleep(0.01)

plt.show(block=True)