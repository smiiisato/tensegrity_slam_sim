import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from rospkg import RosPack

FILE_NAME = 'tension_0115.csv'

LINK_NUM = 1
VALUE = 0

# CSVファイルを読み込む
rospack = RosPack()
csv_file_path = rospack.get_path('tensegrity_slam_sim') + '/logs/' + FILE_NAME
data = pd.read_csv(csv_file_path, header=None)
print(data)

# データのステップ数を取得
steps = 100*1
x_steps = range(steps)  # ステップ数を適宜設定

# 各列のデータに対してグラフを描画
plt.figure(figsize=(15, 10))
plt.plot(x_steps, np.array(data[(LINK_NUM-1)*3+VALUE+1][0:steps]), label='link{} value{}'.format(LINK_NUM, VALUE))

# グラフのタイトルと軸ラベルを設定
plt.title('acc-sim')
plt.xlabel('Step')
plt.ylabel('Data Value')
plt.legend()
plt.show()
