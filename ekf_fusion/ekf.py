"""GNSS + IMU 的 EKF(ROS/ENU 约定:x 东, y 北, yaw 逆时针为正)。

状态 x = [px, py, yaw, v, bg, ba]
    px, py : 局部 ENU 位置 (m)
    yaw    : 航向 (rad), 从 +x(东)逆时针
    v      : 沿车头方向的速度 (m/s)
    bg     : 陀螺仪 z 轴零偏 (rad/s)
    ba     : 加速度计 x 轴零偏 (m/s^2), 包含车身俯仰带来的重力分量

预测(IMU 驱动): 输入 gz(角速度), ax(车头方向加速度)
更新(GNSS)    : 观测 z = [px, py]
"""
import numpy as np


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class GnssImuEKF:
    def __init__(self, x0, P0, gyro_std, accel_std, bias_walk_std, gnss_std, accel_bias_walk_std=5e-3):
        x0 = np.asarray(x0, dtype=float)
        P0 = np.asarray(P0, dtype=float)
        if len(x0) == 5:                      # 兼容旧的 5 维初值: 补上 ba = 0
            x0 = np.append(x0, 0.0)
            P0 = np.pad(P0, ((0, 1), (0, 1)))
            P0[5, 5] = 0.1
        self.x = x0.copy()
        self.P = P0.copy()
        self.accel_bias_walk_std = accel_bias_walk_std
        self.gyro_std = gyro_std
        self.accel_std = accel_std
        self.bias_walk_std = bias_walk_std
        self.R = np.eye(2) * gnss_std ** 2
        self.H = np.zeros((2, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.last_nis = float("nan")

    def predict(self, gz, ax, dt):
        px, py, yaw, v, bg, ba = self.x
        w = gz - bg
        c, s = np.cos(yaw), np.sin(yaw)

        self.x = np.array([
            px + v * c * dt,
            py + v * s * dt,
            wrap(yaw + w * dt),
            v + (ax - ba) * dt,
            bg,
            ba,
        ])

        F = np.eye(6)
        F[0, 2] = -v * s * dt
        F[0, 3] = c * dt
        F[1, 2] = v * c * dt
        F[1, 3] = s * dt
        F[2, 4] = -dt
        F[3, 5] = -dt

        # 过程噪声: 位置给一个很小的模型误差, 其余来自 IMU 噪声和零偏随机游走
        Q = np.diag([
            (0.02 * dt) ** 2,
            (0.02 * dt) ** 2,
            (self.gyro_std * dt) ** 2,
            (self.accel_std * dt) ** 2,
            (self.bias_walk_std * dt) ** 2,
            (self.accel_bias_walk_std * dt) ** 2,
        ])
        self.P = F @ self.P @ F.T + Q

    def update_gnss(self, z):
        y = np.asarray(z, dtype=float) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = wrap(self.x[2])

        # Joseph 形式, 数值上更稳
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.last_nis = float(y @ np.linalg.solve(S, y))
