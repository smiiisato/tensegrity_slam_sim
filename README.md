# tensegrity_slam_sim

## scripts
### mujoco_sim_PubSitePos.py 
シミュレータからテンセグリティロボットモデルの各頂点の座標をpublishする

### tendon_length_publisher.py
各ワイヤの長さをpublishする

### tensegrity_rl.py
コントローラ用の強化学習。

### tensegrity_sim.py
強化学習の環境を定義したクラス(TensegrityEnv)

## models
### scene.xml
環境モデル

### tensegrity_tension_actuated.xml
張力制御テンセグリティモデル

### scene_old.xml
tensegrity.xmlをincludeする環境モデル

### tensegrity.xml
旧テンセグリティモデル

## launch
### site_position.launch
各頂点の位置をrvizで表示する
