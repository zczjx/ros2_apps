#!/usr/bin/env python3
"""录制 IMU + GNSS (+ 真值) 话题, 保存成 CSV, 供 ekf_offline.py 离线重放。

输出两个文件:
    <prefix>_sensors.csv : t,type,a,b,c,d      (按时间戳排序, 一行一条消息)
        imu  : a=gyro_z(rad/s)  b=accel_x(m/s^2)  c=accel_y  d=gyro_x(留空位, 保持 4 列)
        gnss : a=latitude(deg)  b=longitude(deg)  c=altitude(m)
    <prefix>_truth.csv   : t,x,y,yaw,v         (ROS 右手系, 来自 /carla/hero/ground_truth)

注意: 保存的都是话题里的原始值, 不做任何坐标转换, 转换放在 ekf_offline.py 里。
用法: source /opt/ros/jazzy/setup.bash && python3 record_csv.py data/run1 --seconds 120
"""
import argparse
import csv
import math
import os
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, NavSatFix


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class Recorder(Node):
    def __init__(self, args):
        super().__init__("csv_recorder")
        self.sensors, self.truth = [], []
        self.create_subscription(Imu, args.imu_topic, self.on_imu, qos_profile_sensor_data)
        self.create_subscription(NavSatFix, args.gnss_topic, self.on_gnss, qos_profile_sensor_data)
        self.create_subscription(Odometry, args.truth_topic, self.on_truth, qos_profile_sensor_data)

    def on_imu(self, m):
        self.sensors.append((stamp(m), "imu", m.angular_velocity.z, m.linear_acceleration.x,
                             m.linear_acceleration.y, m.angular_velocity.x))

    def on_gnss(self, m):
        self.sensors.append((stamp(m), "gnss", m.latitude, m.longitude, m.altitude, ""))

    def on_truth(self, m):
        q = m.pose.pose.orientation
        self.truth.append((stamp(m), m.pose.pose.position.x, m.pose.pose.position.y,
                           2.0 * math.atan2(q.z, q.w), m.twist.twist.linear.x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", help="输出文件前缀, 例如 data/run1")
    ap.add_argument("--seconds", type=float, default=120.0, help="按 *仿真时间* 录多久 (默认 120 s)")
    ap.add_argument("--imu-topic", default="/carla/hero/imu")
    ap.add_argument("--gnss-topic", default="/carla/hero/gnss")
    ap.add_argument("--truth-topic", default="/carla/hero/ground_truth")
    ap.add_argument("--timeout", type=float, default=600.0, help="墙钟时间上限, 防止仿真暂停时一直等")
    args = ap.parse_args()

    rclpy.init()
    node = Recorder(args)
    wall0, sim0, last_report = time.time(), None, 0.0
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.sensors:
            now = node.sensors[-1][0]
            if sim0 is None:
                sim0 = node.sensors[0][0]
            if now - sim0 >= args.seconds:
                break
            if now - sim0 - last_report >= 10:
                last_report = now - sim0
                print("  录制中 %.0f / %.0f s (imu %d, gnss %d, truth %d)" % (
                    last_report, args.seconds, sum(1 for r in node.sensors if r[1] == "imu"),
                    sum(1 for r in node.sensors if r[1] == "gnss"), len(node.truth)))
        if time.time() - wall0 > args.timeout:
            print("超过墙钟时间上限, 提前结束")
            break
    node.destroy_node()
    rclpy.shutdown()

    os.makedirs(os.path.dirname(os.path.abspath(args.prefix)), exist_ok=True)
    rows = sorted(node.sensors, key=lambda r: (r[0], r[1] != "imu"))   # 同一时刻 imu 排在 gnss 前面
    t0 = rows[0][0] if rows else 0.0
    with open(args.prefix + "_sensors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "type", "a", "b", "c", "d"])
        for r in rows:
            w.writerow(["%.4f" % (r[0] - t0), r[1]] + [("%.9f" % x if isinstance(x, float) else x) for x in r[2:]])
    with open(args.prefix + "_truth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "yaw", "v"])
        for r in node.truth:
            w.writerow(["%.4f" % (r[0] - t0)] + ["%.6f" % x for x in r[1:]])
    n_imu = sum(1 for r in rows if r[1] == "imu")
    print("已保存 %s_sensors.csv (imu %d, gnss %d) 和 %s_truth.csv (%d 行)" % (
        args.prefix, n_imu, len(rows) - n_imu, args.prefix, len(node.truth)))


if __name__ == "__main__":
    main()
