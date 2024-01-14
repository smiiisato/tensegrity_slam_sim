import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from rospkg import RosPack

FILE_NAME = 'tension_0114.csv'

# CSVファイルを読み込む
rospack = RosPack()
csv_file_path = rospack.get_path('tensegrity_slam_sim') + '/logs/' + FILE_NAME
data = pd.read_csv(csv_file_path)

# データのステップ数を取得
steps = 100*1
x_steps = range(steps)  # ステップ数を適宜設定

# 各列のデータに対してグラフを描画
plt.figure(figsize=(15, 10))
for column in data.columns:
    if column == 'Step':
        continue
    plt.plot(x_steps, np.array(data[column][0:steps]), label=f'{column}')

# グラフのタイトルと軸ラベルを設定
plt.title('acc-sim')
plt.xlabel('Step')
plt.ylabel('Data Value')
plt.legend()
plt.show()
