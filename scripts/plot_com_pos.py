import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from rospkg import RosPack

""" FILE_NAME_1 = 'com_pos_terrain_0.csv'
FILE_NAME_2 = 'com_pos_terrain_05.csv'
FILE_NAME_3 = 'com_pos_terrain_1.csv'
FILE_NAME_4 = 'com_pos_terrain_15.csv' """
""" FILE_NAME_1 = 'com_pos_friction_0.csv'
FILE_NAME_2 = 'com_pos_friction_1.csv'
FILE_NAME_3 = 'com_pos_terrain_0.csv'
FILE_NAME_4 = 'com_pos_friction_10.csv'
NUM = [7,8,9,10] """
FILE_NAME_1 = 'com_pos_slope_3.csv'
FILE_NAME_2 = 'com_pos_slope_5.csv'
FILE_NAME_3 = 'com_pos_slope_3.csv'
FILE_NAME_4 = 'com_pos_slope_5.csv'
NUM = [5,6,5,6]

# CSVファイルを読み込む
rospack = RosPack()
csv_file_path = rospack.get_path('tensegrity_slam_sim') + '/logs/' + FILE_NAME_1
data_1 = pd.read_csv(csv_file_path, header=None)
csv_file_path = rospack.get_path('tensegrity_slam_sim') + '/logs/' + FILE_NAME_2
data_2 = pd.read_csv(csv_file_path, header=None)
csv_file_path = rospack.get_path('tensegrity_slam_sim') + '/logs/' + FILE_NAME_3
data_3 = pd.read_csv(csv_file_path, header=None)
csv_file_path = rospack.get_path('tensegrity_slam_sim') + '/logs/' + FILE_NAME_4
data_4 = pd.read_csv(csv_file_path, header=None)

# データのステップ数を取得
steps = 100*22
x_steps = range(steps)  # ステップ数を適宜設定

# 各列のデータに対してグラフを描画
plt.figure(figsize=(5, 3.5))
#plt.plot(x_steps, np.array(data_1[1][0:steps]), label=f'実験{NUM[0]}')
#plt.plot(x_steps, np.array(data_2[1][0:steps]), label=f'実験{NUM[1]}')
plt.plot(x_steps, np.array(data_3[1][0:steps]), label=f'実験{NUM[2]}')
plt.plot(x_steps, np.array(data_4[1][0:steps]), label=f'実験{NUM[3]}')

# グラフのタイトルと軸ラベルを設定
plt.title('重心位置',fontname = 'IPAPGothic')
plt.xlabel('Step')
plt.ylabel('body com pos[m]')
plt.legend(prop={"family":"IPAexGothic"})
plt.show()
