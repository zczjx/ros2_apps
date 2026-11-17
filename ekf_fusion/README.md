# GNSS + IMU EKF 融合

## 运行(三个终端,都先 `source /opt/ros/jazzy/setup.bash`)

```bash
# 1. CARLA 服务器(必须带 --ros2)
./CarlaUE4.sh -RenderOffScreen --ros2

# 2. 起车并发布传感器 + 真值话题
cd examples/ros2_apps/carla_ros2_demo
python3 ros2_native.py --file stack.json --spawn-point 10

# 3. EKF 节点 + 实时三幅图
cd examples/ros2_apps/ekf_fusion
python3 ekf_node.py
python3 ekf_node.py --ros-args -p outage_start:=40.0 -p outage_end:=70.0   # GNSS 中断实验
```

话题: `/carla/hero/gnss`, `/carla/hero/imu`, `/carla/hero/ground_truth`(Odometry, 由 ros2_native.py 发布)。
话题名、噪声参数都可以用 `--ros-args -p 名字:=值` 覆盖。

## 其它
- `record_data.py out.npz 130` 录数据, `replay_tune.py out.npz` 离线比较不同参数。
- `run_ekf.py` 是不依赖 ROS 的旧版本(自己生成车),不要和 `ros2_native.py` 同时运行。
