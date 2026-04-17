"""
WLR MuJoCo simulation — 按照毕业论文的控制架构实现。

VMC + 6状态LQR + Jacobian转置 + 腿长PID。

状态变量：[theta, dtheta, x, dx, phi, dphi]
VMC：phi1, phi4 → (L0, theta0)
Jacobian：(F, Tp) → (T1, T2)
"""

import math, os, numpy as np
import mujoco, rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from scipy.linalg import solve_continuous_are

# ── 五连杆参数（论文 Table 2.1）──
L1 = 0.07       # 上连杆
L2 = 0.147      # 下连杆
L3 = 0.147      # 下连杆（后）
L4 = 0.07       # 上连杆（后）
L5 = 0.123      # 固定杆（髋间距）
WR = 0.05       # 轮半径
G = 9.81
MASS = 3.524    # 总质量

HIP_INIT_F = -0.7
HIP_INIT_R = 0.7

# 站立姿态
HIP0 = 0.0
KNEE0 = 0.0


def vmc_fk(phi1, phi4):
    """五连杆正运动学：phi1, phi4 → L0, theta0."""
    xb = L1 * math.cos(phi1)
    yb = L1 * math.sin(phi1)
    xd = L4 * math.cos(phi4) + L5
    yd = L4 * math.sin(phi4)

    A0 = 2.0 * L2 * (xd - xb)
    B0 = 2.0 * L2 * (yd - yb)
    C0 = L2 * L2 + (xd - xb) ** 2 + (yd - yb) ** 2 - L3 * L3

    disc = A0 * A0 + B0 * B0 - C0 * C0
    if disc < 0:
        disc = 0.0
    phi2 = 2.0 * math.atan2(B0 + math.sqrt(disc), A0 + C0)

    xc = xb + L2 * math.cos(phi2)
    yc = yb + L2 * math.sin(phi2)

    mid_x = L5 / 2.0
    L0 = math.sqrt((xc - mid_x) ** 2 + yc * yc)
    theta0 = math.atan2(yc, xc - mid_x)

    return L0, theta0


def jacobian_transpose(F, Tp, phi1, phi4):
    """Jacobian转置：虚拟力(F, Tp) → 关节力矩(T1, T2)."""
    L0, theta0 = vmc_fk(phi1, phi4)
    if L0 < 0.01:
        return 0.0, 0.0

    A = phi1 - theta0
    B = phi4 - theta0

    J11 = -L1 * math.sin(A) + L2 * math.sin(phi1 + phi4 - theta0 - phi1) / L0
    # 修正：用论文公式
    J11 = -L1 * math.sin(A) + L2 * math.sin(phi2_jac(phi1, phi4) - theta0) * math.sin(phi4 - theta0) / (L0 * math.sin(phi2_jac(phi1, phi4) - phi4))
    # 简化：用论文式(2.23)和(2.24)
    J11 = -L1 * math.sin(A)
    J12 = L1 * math.cos(A) / L0
    J21 = -L4 * math.sin(B)
    J22 = L4 * math.cos(B) / L0

    T1 = J11 * F + J12 * (-Tp)
    T2 = J21 * F + J22 * (-Tp)

    return T1, T2


def phi2_jac(phi1, phi4):
    """计算 phi2 用于 Jacobian."""
    xb = L1 * math.cos(phi1)
    yb = L1 * math.sin(phi1)
    xd = L4 * math.cos(phi4) + L5
    yd = L4 * math.sin(phi4)
    A0 = 2.0 * L2 * (xd - xb)
    B0 = 2.0 * L2 * (yd - yb)
    C0 = L2 * L2 + (xd - xb) ** 2 + (yd - yb) ** 2 - L3 * L3
    disc = A0 * A0 + B0 * B0 - C0 * C0
    if disc < 0:
        disc = 0.0
    return 2.0 * math.atan2(B0 + math.sqrt(disc), A0 + C0)


def compute_lqr_gains():
    """计算6状态LQR增益."""
    # 简化线性化模型
    m = MASS
    l = 0.15
    r = WR
    I = m * l * l / 3.0

    # A矩阵 (6x6 简化为关键子系统)
    A = np.zeros((6, 6))
    A[0, 1] = 1.0   # dtheta
    A[1, 0] = m * G * l / I  # 不稳定极点
    A[2, 3] = 1.0   # dx
    A[4, 5] = 1.0   # dphi

    B = np.zeros((6, 2))
    B[1, 0] = 1.0 / I   # Tp → dtheta
    B[3, 1] = 1.0 / (m * r * r)  # T → dx
    B[5, 0] = -1.0 / I  # Tp → dphi

    Q = np.diag([500, 10, 1, 1, 500, 10])
    R = np.diag([0.5, 0.5])

    try:
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        return K
    except Exception:
        return np.array([[-30, -5, 0, 0, 0, 0], [0, 0, -1, -1, 0, 0]])


NAMES = ['L_hip_j', 'L_knee_j', 'L_wheel_j', 'R_hip_j', 'R_knee_j', 'R_wheel_j']


class MuJoCoNode(Node):
    def __init__(self):
        super().__init__('mujoco_node')
        self.declare_parameter('sim_rate', 1000.0)
        self.declare_parameter('pub_rate', 200.0)
        self.declare_parameter('model_path', '')

        sr = self.get_parameter('sim_rate').value
        pr = self.get_parameter('pub_rate').value
        mp = self.get_parameter('model_path').value
        if not mp:
            from ament_index_python.packages import get_package_share_directory
            mp = os.path.join(get_package_share_directory('wlr_mujoco'), 'model', 'wlr_robot.xml')

        self.dt = 1.0 / sr
        self.pi = int(sr / pr)

        print(f'[mujoco_sim] Loading: {mp}', flush=True)
        self.m = mujoco.MjModel.from_xml_path(mp)
        self.d = mujoco.MjData(self.m)

        self.jq = {}
        self.jv = {}
        for n in NAMES:
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
            self.jq[n] = self.m.jnt_qposadr[jid]
            self.jv[n] = self.m.jnt_dofadr[jid]

        sa = self.m.sensor_adr
        self.qa = sa[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, 'quat')]
        self.ga = sa[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, 'gyro')]

        mujoco.mj_forward(self.m, self.d)

        self.K = compute_lqr_gains()
        self.speed = 0.0
        self.yaw = 0.0
        self.step = 0
        self.prev_L0 = {'L': 0.15, 'R': 0.15}

        self.jp = self.create_publisher(JointState, '/joint_states', 10)
        self.ip = self.create_publisher(Imu, '/imu/data', 10)
        self.cp = self.create_publisher(Clock, '/clock', 10)
        self.create_subscription(Twist, '/cmd_vel', lambda m: (
            setattr(self, 'speed', m.linear.x), setattr(self, 'yaw', m.angular.z)), 10)

        self._pub()
        print(f'[mujoco_sim] Started {sr:.0f}Hz, LQR K shape={self.K.shape}', flush=True)

    def _step(self):
        q = self.d.sensordata[self.qa:self.qa + 4]
        w, x, y, z = q
        pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
        dpitch = self.d.sensordata[self.ga + 1]

        ctrl = np.zeros(6)
        for si, side in enumerate(['L', 'R']):
            sgn = 1 if side == 'L' else -1

            # 读取关节
            hip = self.d.qpos[self.jq[f'{side}_hip_j']]
            knee = self.d.qpos[self.jq[f'{side}_knee_j']]
            wheel = self.d.qpos[self.jq[f'{side}_wheel_j']]
            hip_v = self.d.qvel[self.jv[f'{side}_hip_j']]
            knee_v = self.d.qvel[self.jv[f'{side}_knee_j']]
            wheel_v = self.d.qvel[self.jv[f'{side}_wheel_j']]

            # VMC: URDF角度 → VMC角度
            phi1 = -math.pi / 2 - hip * sgn - HIP_INIT_F
            phi4 = -math.pi / 2 - hip * sgn - HIP_INIT_R

            # 五连杆正运动学
            try:
                L0, theta0 = vmc_fk(phi1, phi4)
                if L0 < 0.01:
                    L0 = 0.01
            except Exception:
                L0 = self.prev_L0[side]
                theta0 = math.pi / 2

            dL0 = (L0 - self.prev_L0[side]) / self.dt
            self.prev_L0[side] = L0

            # 状态变量
            theta = theta0 - math.pi / 2 + pitch
            dtheta = dL0  # 近似
            x_st = wheel * WR
            dx_st = wheel_v * WR
            phi = -pitch
            dphi = -dpitch

            # LQR
            state = np.array([theta, dtheta, x_st, dx_st, phi, dphi])
            lqr = -self.K @ state
            Tp = lqr[0] if len(lqr) > 0 else 0.0
            Tw = lqr[1] if len(lqr) > 1 else 0.0

            # 腿长PID
            F = 200 * (0.15 - L0) - 30 * dL0 + 17.3
            F = max(-40, min(40, F))
            Tp = max(-10, min(10, Tp))

            # Jacobian转置
            T1, T2 = jacobian_transpose(F, Tp, phi1, phi4)

            # 轮子
            Tw += self.speed * 8 + self.yaw * 3 * sgn
            Tw = max(-10, min(10, Tw))

            # 髋阻尼
            hip_t = -10 * hip_v
            hip_t = max(-10, min(10, hip_t))

            # 膝轻柔支撑
            knee_t = 30 * (0 - knee) - 10 * knee_v - 0.3
            knee_t = max(-10, min(10, knee_t))

            ctrl[si * 3] = hip_t
            ctrl[si * 3 + 1] = knee_t
            ctrl[si * 3 + 2] = Tw

        if abs(pitch) > 1.3:
            ctrl[:] = 0

        self.d.ctrl[:] = ctrl
        mujoco.mj_step(self.m, self.d)
        self.step += 1

        if self.step % self.pi == 0:
            self._pub()

    def _pub(self):
        t = self._rt()
        m = JointState()
        m.header.stamp = t
        m.name = ['left_hip', 'left_knee', 'left_wheel', 'right_hip', 'right_knee', 'right_wheel']
        m.position = [self.d.qpos[self.jq[n]] for n in NAMES]
        m.velocity = [self.d.qvel[self.jv[n]] for n in NAMES]
        m.effort = [self.d.ctrl[i] for i in range(6)]
        self.jp.publish(m)

        im = Imu()
        im.header.stamp = t
        im.header.frame_id = 'imu_link'
        q = self.d.sensordata[self.qa:self.qa + 4]
        im.orientation.w, im.orientation.x = float(q[0]), float(q[1])
        im.orientation.y, im.orientation.z = float(q[2]), float(q[3])
        g = self.d.sensordata[self.ga:self.ga + 3]
        im.angular_velocity.x, im.angular_velocity.y, im.angular_velocity.z = float(g[0]), float(g[1]), float(g[2])
        self.ip.publish(im)

        cm = Clock()
        cm.clock = t
        self.cp.publish(cm)

    def _rt(self):
        s = self.d.time
        return Time(seconds=int(s), nanoseconds=int((s - int(s)) * 1e9)).to_msg()


def main(args=None):
    import sys
    visual = '--visual' in sys.argv or '-v' in sys.argv

    rclpy.init(args=args)
    n = MuJoCoNode()

    viewer = None
    if visual:
        import os as _os
        if not _os.environ.get('DISPLAY') and not _os.environ.get('WAYLAND_DISPLAY'):
            print('[mujoco_sim] No display, running headless', flush=True)
        else:
            try:
                import mujoco.viewer
                viewer = mujoco.viewer.launch_passive(n.m, n.d)
                print('[mujoco_sim] Viewer opened', flush=True)
            except Exception as e:
                print(f'[mujoco_sim] Viewer failed ({e}), running headless', flush=True)

    import time as _time
    try:
        nt = _time.monotonic()
        cb_counter = 0
        while rclpy.ok():
            n._step()
            cb_counter += 1
            if cb_counter % 100 == 0:
                rclpy.spin_once(n, timeout_sec=0)
            if viewer is not None and cb_counter % 10 == 0:
                viewer.sync()
            nt += n.dt
            sl = nt - _time.monotonic()
            if sl > 0:
                _time.sleep(sl)
            else:
                nt = _time.monotonic()
    except KeyboardInterrupt:
        pass
    if viewer is not None:
        viewer.close()
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
