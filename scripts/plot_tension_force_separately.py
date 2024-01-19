import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from rospkg import RosPack

FILE_NAME = 'com_vel_forward_0118.csv'

LINK_NUM = 1
VALUE = 0

# CSVファイルを読み込む
rospack = RosPack()
csv_file_path = rospack.get_path('tensegrity_slam_sim') + '/logs/' + FILE_NAME
data = pd.read_csv(csv_file_path, header=None)
print(data)

# データのステップ数を取得
steps = 100*10
x_steps = range(steps)  # ステップ数を適宜設定

# 各列のデータに対してグラフを描画
plt.figure(figsize=(5, 3.5))
#plt.plot(x_steps, np.array(data[(LINK_NUM-1)*3+VALUE+1][0:steps]), label='link{} value{}'.format(LINK_NUM, VALUE))
plt.plot(x_steps, np.array(data[1][0:steps]), label='forward reward')

# グラフのタイトルと軸ラベルを設定
plt.title('The shift of the body com vel')
plt.xlabel('step')
plt.ylabel('body com vel')
plt.legend()
plt.show()
