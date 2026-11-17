"""CARLA 里跑 GNSS + IMU 的 EKF, 并和真值对比 (不依赖 ROS, 纯 Python API)。

用法:
    python3 run_ekf.py --seconds 120 --plot
"""
import argparse
import math
import queue

import numpy as np
import carla

from ekf import GnssImuEKF, wrap

EARTH_R = 6378137.0


def gnss_to_enu(lat, lon, lat0, lon0):
    """小范围内的等距圆柱近似: 返回 (东, 北), 单位 m。"""
    east = math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    north = math.radians(lat - lat0) * EARTH_R
    return east, north


def carla_to_ros_yaw(yaw_deg):
    """CARLA yaw(度, 右转为正, 0 指向 +x) -> ROS yaw(弧度, 左转为正)。"""
    return wrap(-math.radians(yaw_deg))


def make_sensor(world, bp_name, parent, attrs):
    bp = world.get_blueprint_library().find(bp_name)
    for k, v in attrs.items():
        bp.set_attribute(k, str(v))
    s = world.spawn_actor(bp, carla.Transform(), attach_to=parent)
    q = queue.Queue()
    s.listen(q.put)
    return s, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--gnss-every", type=int, default=20, help="每 N 个 tick 给一次 GNSS 更新")
    ap.add_argument("--gnss-std", type=float, default=2.0, help="GNSS 水平噪声 (m)")
    ap.add_argument("--gyro-std", type=float, default=0.01)
    ap.add_argument("--accel-std", type=float, default=0.1)
    ap.add_argument("--gyro-bias", type=float, default=0.005)
    ap.add_argument("--outage", type=float, nargs=2, default=None, metavar=("START", "END"),
                    help="GNSS 中断的起止时间 (s), 例如 40 70")
    ap.add_argument("--spawn", type=int, default=10, help="出生点序号 (0 号在路口, 易停红灯)")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    orig = world.get_settings()
    st = world.get_settings()
    st.synchronous_mode = True
    st.fixed_delta_seconds = args.dt
    world.apply_settings(st)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    vehicle, sensors = None, []
    try:
        bp = world.get_blueprint_library().filter("vehicle.lincoln.mkz_2017")[0]
        vehicle = world.spawn_actor(bp, world.get_map().get_spawn_points()[args.spawn])
        vehicle.set_autopilot(True, tm.get_port())

        # 噪声由 CARLA 传感器本身产生, 这里用 degree 单位的经纬度噪声
        deg_std = math.degrees(args.gnss_std / EARTH_R)
        gnss, gq = make_sensor(world, "sensor.other.gnss", vehicle, {
            "noise_lat_stddev": deg_std, "noise_lon_stddev": deg_std,
            "noise_alt_stddev": 0.5,
        })
        imu, iq = make_sensor(world, "sensor.other.imu", vehicle, {
            "noise_accel_stddev_x": args.accel_std,
            "noise_accel_stddev_y": args.accel_std,
            "noise_accel_stddev_z": args.accel_std,
            "noise_gyro_stddev_z": args.gyro_std,
            "noise_gyro_bias_z": args.gyro_bias,
        })
        sensors = [gnss, imu]

        world.tick()
        gm = world.get_map().transform_to_geolocation(carla.Location(0, 0, 0))
        lat0, lon0 = gm.latitude, gm.longitude
        # 注意: CARLA 的 GNSS 纬度对应 -y 方向, 见下面真值转换

        ekf = None
        log = {"t": [], "gt": [], "est": [], "raw": [], "nis": []}
        n_steps = int(args.seconds / args.dt)

        for k in range(n_steps):
            frame = world.tick()
            t = k * args.dt
            imu_m = iq.get(timeout=2.0)
            gnss_m = gq.get(timeout=2.0)

            tr = vehicle.get_transform()
            vel = vehicle.get_velocity()
            gt = np.array([tr.location.x, -tr.location.y,
                           carla_to_ros_yaw(tr.rotation.yaw),
                           math.hypot(vel.x, vel.y)])

            e, n = gnss_to_enu(gnss_m.latitude, gnss_m.longitude, lat0, lon0)
            # 实测: GNSS 转出的 (e, n) 等于 CARLA 的 (x, y), 即 y 没有翻转(左手系)。
            # 为和 ROS 约定(以及真值 (x, -y)、yaw 取反、gz 取反)一致, 这里把 y 取反
            raw = np.array([e, -n])

            if k < 5:
                # 传感器刚创建时前几帧 IMU 有异常值 (实测 ax≈25830), 直接丢弃
                continue

            if ekf is None:
                ekf = GnssImuEKF(
                    x0=[raw[0], raw[1], gt[2], 0.0, 0.0],
                    P0=np.diag([args.gnss_std ** 2, args.gnss_std ** 2, 0.1, 100.0, 0.01]),
                    gyro_std=args.gyro_std, accel_std=args.accel_std,
                    bias_walk_std=1e-3, gnss_std=args.gnss_std)
                continue

            # CARLA IMU: 左手系。ROS 的 gz = -gz_carla; ax 沿车头, 符号不变
            # 加速度含重力项主要在 z, 对平面 ax 影响很小
            gz = -imu_m.gyroscope.z
            ax = imu_m.accelerometer.x
            ekf.predict(gz, ax, args.dt)

            in_outage = args.outage and args.outage[0] <= t <= args.outage[1]
            if k % args.gnss_every == 0 and not in_outage:
                ekf.update_gnss(raw)

            log["t"].append(t)
            log["gt"].append(gt)
            log["est"].append(ekf.x.copy())
            log["raw"].append(raw)
            log["nis"].append(ekf.last_nis)

        gt = np.array(log["gt"])
        est = np.array(log["est"])
        raw = np.array(log["raw"])
        pos_err = np.linalg.norm(est[:, :2] - gt[:, :2], axis=1)
        raw_err = np.linalg.norm(raw - gt[:, :2], axis=1)
        yaw_err = np.degrees(np.abs(wrap(est[:, 2] - gt[:, 2])))
        print(f"位置 RMSE  EKF: {np.sqrt(np.mean(pos_err**2)):.3f} m   "
              f"纯 GNSS: {np.sqrt(np.mean(raw_err**2)):.3f} m")
        print(f"航向 RMSE  EKF: {np.sqrt(np.mean(yaw_err**2)):.3f} deg")
        v_err = est[:, 3] - gt[:, 3]
        print(f"速度 RMSE: {np.sqrt(np.mean(v_err**2)):.3f} m/s   均值误差: {v_err.mean():.3f}")
        print(f"陀螺零偏估计末值: {est[-1, 4]:.4f} rad/s (设置值 {args.gyro_bias})")

        if args.plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(16, 5))
            ax[0].plot(gt[:, 0], gt[:, 1], "k", label="truth")
            ax[0].plot(raw[:, 0], raw[:, 1], ".", ms=1, c="r", alpha=0.3, label="GNSS raw")
            ax[0].plot(est[:, 0], est[:, 1], "b", label="EKF")
            ax[0].axis("equal"); ax[0].legend(); ax[0].set_title("trajectory (ENU)")
            ax[1].plot(log["t"], raw_err, "r", alpha=0.4, label="GNSS raw")
            ax[1].plot(log["t"], pos_err, "b", label="EKF")
            ax[1].set_xlabel("t (s)"); ax[1].set_ylabel("pos err (m)"); ax[1].legend()
            ax[2].plot(log["t"], est[:, 4], label="bg est")
            ax[2].axhline(args.gyro_bias, c="k", ls="--", label="bg true(config)")
            ax[2].legend(); ax[2].set_title("gyro bias")
            plt.tight_layout(); plt.show()

    finally:
        world.apply_settings(orig)
        for s in sensors:
            s.destroy()
        if vehicle:
            vehicle.destroy()


if __name__ == "__main__":
    main()
