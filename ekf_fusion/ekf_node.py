#!/usr/bin/env python3
"""ROS2 节点: 订阅 CARLA 的 GNSS / IMU / 真值话题, 实时跑 EKF 并显示 3 幅图。

话题约定 (已用 CARLA 实测确认):
  * /carla/hero/gnss : NavSatFix, 经度对应 CARLA 的 +x, 纬度对应 CARLA 的 +y, 因此 ROS 系的 y = -北向
  * /carla/hero/imu  : Imu, angular_velocity.z 已是 ROS 约定 (与 CARLA 的 z 轴角速度符号相反), 直接使用
  * /carla/hero/ground_truth : Odometry, 由 ros2_native.py 发布, 已是 ROS 右手系
节点内部统一使用 ROS 约定: x 东, y 北, yaw 逆时针为正。
"""
import argparse
import math
import threading
from collections import deque

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, NavSatFix

from ekf import GnssImuEKF, wrap

EARTH_R = 6378137.0


class EkfFusionNode(Node):
    def __init__(self):
        super().__init__("ekf_fusion")
        self.declare_parameter("gnss_topic", "/carla/hero/gnss")
        self.declare_parameter("imu_topic", "/carla/hero/imu")
        self.declare_parameter("truth_topic", "/carla/hero/ground_truth")
        self.declare_parameter("gnss_std", 2.0)
        self.declare_parameter("gyro_std", 0.01)
        self.declare_parameter("accel_std", 0.1)
        self.declare_parameter("bias_walk_std", 1e-3)
        self.declare_parameter("gnss_every", 20)
        self.declare_parameter("skip_imu", 5)
        self.declare_parameter("outage_start", -1.0)
        self.declare_parameter("outage_end", -1.0)
        self.declare_parameter("history", 4000)
        self.declare_parameter("origin_lat", 0.0)   # CARLA 地图原点的经纬度 (实测为 0, 0)
        self.declare_parameter("origin_lon", 0.0)

        p = lambda n: self.get_parameter(n).value
        self.gnss_std, self.gyro_std = p("gnss_std"), p("gyro_std")
        self.accel_std, self.bias_walk_std = p("accel_std"), p("bias_walk_std")
        self.gnss_every, self.skip_imu = p("gnss_every"), p("skip_imu")
        self.outage = (p("outage_start"), p("outage_end"))

        self.lock = threading.Lock()
        n = p("history")
        self.log = {k: deque(maxlen=n) for k in ("t", "gt", "est", "raw", "bg")}
        self.ekf = None
        self.origin = (p("origin_lat"), p("origin_lon"))   # 与 CARLA 世界坐标对齐的 GNSS 原点
        self.gnss_buf = deque(maxlen=200)
        self.last_gnss = None
        self.truth_buf = deque(maxlen=200)   # (stamp, [x, y, yaw, v])
        self.last_imu_t = None
        self.imu_count = 0
        self.t0 = None

        self.create_subscription(NavSatFix, p("gnss_topic"), self.on_gnss, qos_profile_sensor_data)
        self.create_subscription(Imu, p("imu_topic"), self.on_imu, qos_profile_sensor_data)
        self.create_subscription(Odometry, p("truth_topic"), self.on_truth, qos_profile_sensor_data)
        self.get_logger().info("等待 GNSS / IMU / 真值数据...")

    @staticmethod
    def stamp(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    @staticmethod
    def nearest(buf, t, tol=0.075):
        """取时间戳最接近 t 的一条 (同一 tick 内 IMU 常先于 GNSS/真值到达, 所以允许落后 1~2 个 tick)。"""
        best = None
        for ts, v in reversed(buf):
            if best is None or abs(ts - t) < abs(best[0] - t):
                best = (ts, v)
            if ts < t - tol:
                break
        return best[1] if best and abs(best[0] - t) <= tol else None

    def on_gnss(self, msg):
        lat0, lon0 = self.origin
        east = math.radians(msg.longitude - lon0) * EARTH_R * math.cos(math.radians(lat0))
        north = math.radians(msg.latitude - lat0) * EARTH_R
        # 实测: 纬度方向对应 CARLA 的 +y, 转成 ROS 需要取反
        pos = np.array([east, -north])
        t = self.stamp(msg)
        self.gnss_buf.append((t, pos))
        self.last_gnss = pos
        # GNSS 到达即做更新: 此时 EKF 已被同一时间戳的 IMU 预测到 t
        if self.ekf is None or self.t0 is None:
            return
        self.gnss_seq += 1
        rel = t - self.t0
        in_outage = self.outage[0] >= 0 and self.outage[0] <= rel <= self.outage[1]
        if self.gnss_seq % self.gnss_every == 0 and not in_outage:
            with self.lock:
                self.ekf.update_gnss(pos)

    def on_truth(self, msg):
        q = msg.pose.pose.orientation
        yaw = 2.0 * math.atan2(q.z, q.w)
        p = msg.pose.pose.position
        self.truth_buf.append((self.stamp(msg), np.array([p.x, p.y, wrap(yaw), msg.twist.twist.linear.x])))

    def on_imu(self, msg):
        self.imu_count += 1
        if self.imu_count <= self.skip_imu:      # 传感器刚创建时的前几帧有异常值
            return
        t = self.stamp(msg)
        truth = self.nearest(self.truth_buf, t)
        if self.last_gnss is None or truth is None:
            return
        if self.ekf is None:
            self.t0 = t
            self.gnss_seq = 0
            self.ekf = GnssImuEKF(
                x0=[self.last_gnss[0], self.last_gnss[1], truth[2], 0.0, 0.0],
                P0=np.diag([self.gnss_std ** 2, self.gnss_std ** 2, 0.1, 100.0, 0.01]),
                gyro_std=self.gyro_std, accel_std=self.accel_std,
                bias_walk_std=self.bias_walk_std, gnss_std=self.gnss_std)
            self.last_imu_t = t
            return

        dt = t - self.last_imu_t
        self.last_imu_t = t
        if dt <= 0.0 or dt > 0.5:
            return
        with self.lock:
            self.ekf.predict(msg.angular_velocity.z, msg.linear_acceleration.x, dt)
            self.log["t"].append(t - self.t0)
            self.log["gt"].append(truth.copy())
            self.log["est"].append(self.ekf.x.copy())
            self.log["raw"].append(self.last_gnss.copy())
            self.log["bg"].append(self.ekf.x[4])

    def snapshot(self):
        with self.lock:
            if len(self.log["t"]) < 2:
                return None
            return {k: np.array(v) for k, v in self.log.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=10.0, help="图像刷新频率 (Hz)")
    args, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args)
    node = EkfFusionNode()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    l_gt, = ax[0].plot([], [], "k", label="truth")
    l_raw, = ax[0].plot([], [], ".", ms=2, c="r", alpha=0.3, label="GNSS raw")
    l_est, = ax[0].plot([], [], "b", label="EKF")
    ax[0].set_title("trajectory (ENU)"); ax[0].set_aspect("equal", adjustable="datalim"); ax[0].legend()
    l_re, = ax[1].plot([], [], "r", alpha=0.4, label="GNSS raw")
    l_ee, = ax[1].plot([], [], "b", label="EKF")
    ax[1].set_xlabel("t (s)"); ax[1].set_ylabel("pos err (m)"); ax[1].legend(); ax[1].set_title("position error")
    l_bg, = ax[2].plot([], [], label="bg est (ROS frame)")
    ax[2].set_title("gyro bias (rad/s)"); ax[2].set_xlabel("t (s)"); ax[2].legend()
    text = ax[1].text(0.02, 0.95, "", transform=ax[1].transAxes, va="top")
    plt.tight_layout()

    def update(_):
        d = node.snapshot()
        if d is None:
            return
        gt, est, raw, t = d["gt"], d["est"], d["raw"], d["t"]
        l_gt.set_data(gt[:, 0], gt[:, 1]); l_raw.set_data(raw[:, 0], raw[:, 1]); l_est.set_data(est[:, 0], est[:, 1])
        ax[0].relim(); ax[0].autoscale_view()
        pe = np.linalg.norm(est[:, :2] - gt[:, :2], axis=1)
        re = np.linalg.norm(raw - gt[:, :2], axis=1)
        l_re.set_data(t, re); l_ee.set_data(t, pe)
        ax[1].relim(); ax[1].autoscale_view()
        l_bg.set_data(t, d["bg"]); ax[2].relim(); ax[2].autoscale_view()
        yaw_err = np.degrees(np.abs(wrap(est[:, 2] - gt[:, 2])))
        text.set_text("RMSE EKF %.2f m | GNSS %.2f m | yaw %.1f deg" % (
            np.sqrt(np.mean(pe ** 2)), np.sqrt(np.mean(re ** 2)), np.sqrt(np.mean(yaw_err ** 2))))

    anim = FuncAnimation(fig, update, interval=int(1000 / args.rate), cache_frame_data=False)
    try:
        plt.show()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
