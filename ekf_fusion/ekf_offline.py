#!/usr/bin/env python3
"""离线重放 record_csv.py 录的 CSV: 调用 ekf.py 的 GnssImuEKF, 输出结果 CSV 和 PNG (可选弹窗)。

流程与 ekf_node.py 一致:
    imu  行 -> ekf.predict(gz, ax, dt)
    gnss 行 -> 经纬度转局部 ENU (y 取反, 见 README) -> 每 gnss_every 个 GNSS 更新一次
用法:
    python3 ekf_offline.py data/run1 --show
    python3 ekf_offline.py data/run1 --outage 40 70 --gnss-every 20
"""
import argparse
import csv
import math
import os

import numpy as np

from ekf import GnssImuEKF, wrap

EARTH_R = 6378137.0


def load(prefix):
    imu, gnss = [], []
    with open(prefix + "_sensors.csv") as f:
        for r in csv.DictReader(f):
            t = float(r["t"])
            if r["type"] == "imu":
                imu.append((t, float(r["a"]), float(r["b"])))
            else:
                gnss.append((t, float(r["a"]), float(r["b"])))
    truth = np.genfromtxt(prefix + "_truth.csv", delimiter=",", names=True)
    return np.array(imu), np.array(gnss), truth


def run(imu, gnss, truth, a):
    lat0, lon0 = a.origin_lat, a.origin_lon
    # GNSS 经纬度 -> 局部 (x, y); 纬度方向对应 CARLA +y, 转到 ROS 系取反
    gx = np.radians(gnss[:, 2] - lon0) * EARTH_R * math.cos(math.radians(lat0))
    gy = -np.radians(gnss[:, 1] - lat0) * EARTH_R
    gt_of = lambda t: [np.interp(t, truth["t"], truth[k]) for k in ("x", "y", "yaw", "v")]

    ekf, gi, gnss_seq, last_t = None, 0, 0, None
    last_raw = None
    rows = []
    for k in range(a.skip_imu, len(imu)):
        t, gz, ax = imu[k]
        # 先吃掉时间戳 <= t 的 GNSS, 记录最新一帧原始值; 按 gnss_every 抽样更新
        while gi < len(gnss) and gnss[gi, 0] <= t + 1e-6:
            last_raw = np.array([gx[gi], gy[gi]])
            gnss_seq += 1
            gi += 1
            if ekf is not None:
                in_outage = a.outage and a.outage[0] <= t <= a.outage[1]
                if gnss_seq % a.gnss_every == 0 and not in_outage:
                    ekf.update_gnss(last_raw)
        if last_raw is None:
            continue
        if ekf is None:
            yaw0 = gt_of(t)[2]          # 初始航向取真值 (与 ekf_node.py 一致)
            ekf = GnssImuEKF([last_raw[0], last_raw[1], yaw0, 0.0, 0.0],
                             np.diag([a.gnss_std ** 2, a.gnss_std ** 2, 0.1, 100.0, 0.01]),
                             a.gyro_std, a.accel_std, a.bias_walk_std, a.gnss_std)
            last_t = t
            continue
        dt = t - last_t
        last_t = t
        if not (0.0 < dt <= 0.5):
            continue
        ekf.predict(gz, ax, dt)
        g = gt_of(t)
        rows.append([t, *g, *last_raw, *ekf.x, np.sqrt(ekf.P[0, 0] + ekf.P[1, 1]), ekf.last_nis])
    return np.array(rows)


COLS = ["t", "gt_x", "gt_y", "gt_yaw", "gt_v", "gnss_x", "gnss_y",
        "est_x", "est_y", "est_yaw", "est_v", "est_bg", "est_ba", "pos_sigma", "nis"]


def plot(r, a, png):
    import matplotlib
    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = r[:, 0]
    gt, raw, est = r[:, 1:3], r[:, 5:7], r[:, 7:9]
    pe = np.linalg.norm(est - gt, axis=1)
    re = np.linalg.norm(raw - gt, axis=1)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].plot(gt[:, 0], gt[:, 1], "k", label="truth")
    ax[0].plot(raw[:, 0], raw[:, 1], ".", ms=2, c="r", alpha=0.3, label="GNSS raw")
    ax[0].plot(est[:, 0], est[:, 1], "b", label="EKF")
    ax[0].set_aspect("equal", adjustable="datalim"); ax[0].legend(); ax[0].set_title("trajectory (ENU)")
    ax[1].plot(t, re, "r", alpha=0.4, label="GNSS raw")
    ax[1].plot(t, pe, "b", label="EKF")
    if a.outage:
        ax[1].axvspan(a.outage[0], a.outage[1], color="gray", alpha=0.2, label="GNSS outage")
    ax[1].set_xlabel("t (s)"); ax[1].set_ylabel("pos err (m)"); ax[1].legend(); ax[1].set_title("position error")
    ax[2].plot(t, r[:, 11], label="bg est (ROS frame)")
    ax[2].set_xlabel("t (s)"); ax[2].set_title("gyro bias (rad/s)"); ax[2].legend()
    yaw_rmse = np.degrees(np.sqrt(np.mean(wrap(r[:, 9] - r[:, 3]) ** 2)))
    ax[1].text(0.02, 0.95, "RMSE EKF %.2f m | GNSS %.2f m | yaw %.1f deg" % (
        np.sqrt(np.mean(pe ** 2)), np.sqrt(np.mean(re ** 2)), yaw_rmse),
        transform=ax[1].transAxes, va="top")
    plt.tight_layout()
    fig.savefig(png, dpi=120)
    print("已保存", png)
    if a.show:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", help="record_csv.py 的输出前缀, 例如 data/run1")
    ap.add_argument("--out", default=None, help="输出前缀, 默认 <prefix>_ekf")
    ap.add_argument("--show", action="store_true", help="弹出 matplotlib 窗口")
    ap.add_argument("--gnss-std", type=float, default=2.0)
    ap.add_argument("--gyro-std", type=float, default=0.01)
    ap.add_argument("--accel-std", type=float, default=0.1)
    ap.add_argument("--bias-walk-std", type=float, default=1e-3)
    ap.add_argument("--gnss-every", type=int, default=20, help="每 N 个 GNSS 帧更新一次 (20 Hz -> 1 Hz)")
    ap.add_argument("--skip-imu", type=int, default=5, help="丢弃前 N 帧 IMU (传感器刚创建时有异常值)")
    ap.add_argument("--outage", type=float, nargs=2, default=None, metavar=("START", "END"))
    ap.add_argument("--origin-lat", type=float, default=0.0)
    ap.add_argument("--origin-lon", type=float, default=0.0)
    a = ap.parse_args()

    imu, gnss, truth = load(a.prefix)
    print("读入: imu %d, gnss %d, truth %d" % (len(imu), len(gnss), len(truth)))
    r = run(imu, gnss, truth, a)

    out = a.out or a.prefix + "_ekf"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(["%.6f" % x if i == 0 else "%.6f" % x for i, x in enumerate(row)] for row in r)
    print("已保存", out + ".csv", "(%d 行)" % len(r))

    pe = np.linalg.norm(r[:, 7:9] - r[:, 1:3], axis=1)
    re = np.linalg.norm(r[:, 5:7] - r[:, 1:3], axis=1)
    print("位置 RMSE  EKF %.3f m | 纯 GNSS %.3f m" % (np.sqrt(np.mean(pe ** 2)), np.sqrt(np.mean(re ** 2))))
    print("航向 RMSE  %.2f deg" % np.degrees(np.sqrt(np.mean(wrap(r[:, 9] - r[:, 3]) ** 2))))
    print("速度 RMSE  %.3f m/s | 末尾陀螺零偏估计 %.4f rad/s" % (
        np.sqrt(np.mean((r[:, 10] - r[:, 4]) ** 2)), r[-1, 11]))
    plot(r, a, out + ".png")


if __name__ == "__main__":
    main()
