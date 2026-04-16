"""
WLR MuJoCo simulation — Open-chain model with VMC + LQR balance controller.
Based on clearlab-sustech/Wheel-Legged-Gym approach.

Controller: reads hip+knee angles, computes virtual leg (L0, theta0) via FK,
applies PD in virtual space, maps to joint torques via Jacobian transpose.
"""

import math, os, numpy as np
import mujoco, rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock


# Robot geometry (from clearlab-sustech/Wheel-Legged-Gym)
L1 = 0.15       # upper link (m)
L2 = 0.25       # lower link (m)
OFFSET = 0.054  # hip X offset (m)
WR = 0.05       # wheel radius (m)

# Standing pose (from reference repo)
HIP0 = 0.5      # initial hip angle (rad)
KNEE0 = 0.35    # initial knee angle (rad)

# VMC gains (tuned for stability)
KP_L0 = 200.0    # leg length P
KD_L0 = 15.0     # leg length D
KP_THETA = 30.0  # leg angle P
KD_THETA = 3.0   # leg angle D
FFORCE = 20.0    # feedforward gravity force

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

        # Joint/sensor indices (must be set before _q() is called)
        self.jq = {}
        self.jv = {}
        for n in NAMES:
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
            self.jq[n] = self.m.jnt_qposadr[jid]
            self.jv[n] = self.m.jnt_dofadr[jid]

        # Set standing pose
        for s in ['L', 'R']:
            sgn = 1 if s == 'L' else -1
            self.d.qpos[self._q(f'{s}_hip_j')] = HIP0 * sgn
            self.d.qpos[self._q(f'{s}_knee_j')] = KNEE0 * sgn

        mujoco.mj_forward(self.m, self.d)

        sa = self.m.sensor_adr
        self.qa = sa[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, 'quat')]
        self.ga = sa[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, 'gyro')]
        self.base_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, 'base')

        # Compute nominal L0 at standing pose
        self.l0_nom = self._fk('L')[0]
        self.prev_l0 = {'L': self.l0_nom, 'R': self.l0_nom}
        self.prev_th0 = {'L': 0.0, 'R': 0.0}
        self.prev_l0dot = {'L': 0.0, 'R': 0.0}
        self.prev_th0dot = {'L': 0.0, 'R': 0.0}

        self.speed = 0.0
        self.yaw = 0.0
        self.step = 0

        # ROS
        self.jp = self.create_publisher(JointState, '/joint_states', 10)
        self.ip = self.create_publisher(Imu, '/imu/data', 10)
        self.cp = self.create_publisher(Clock, '/clock', 10)
        self.create_subscription(Twist, '/cmd_vel', lambda m: (
            setattr(self, 'speed', m.linear.x), setattr(self, 'yaw', m.angular.z)), 10)

        self._pub()
        print(f'[mujoco_sim] Started {sr:.0f}Hz, L0_nom={self.l0_nom:.3f}', flush=True)

    def _q(self, n):
        return self.jq[n]

    def _fk(self, side):
        """Open-chain FK: returns (L0, theta0, t1, t2) from hip+knee."""
        sgn = 1 if side == 'L' else -1
        hip = self.d.qpos[self.jq[f'{side}_hip_j']] * sgn
        knee = self.d.qpos[self.jq[f'{side}_knee_j']] * sgn
        t1 = hip
        t2 = knee - math.pi / 2
        ex = OFFSET + L1 * math.cos(t1) + L2 * math.cos(t1 + t2)
        ey = -(L1 * math.sin(t1) + L2 * math.sin(t1 + t2))
        L0 = math.sqrt(ex ** 2 + ey ** 2)
        theta0 = math.atan2(ey, ex) - math.pi / 2
        return L0, theta0, t1, t2

    def _vmc(self, F, T, L0, theta0, t1, t2):
        """Jacobian transpose: virtual force/torque → hip, knee torques."""
        tp = theta0 + math.pi / 2
        t11 = L1 * math.sin(tp - t1) - L2 * math.sin(t1 + t2 - tp)
        t12 = (L1 * math.cos(tp - t1) - L2 * math.cos(t1 + t2 - tp)) / L0
        t21 = -L2 * math.sin(t1 + t2 - tp)
        t22 = -L2 * math.cos(t1 + t2 - tp) / L0
        Th = t11 * F - t12 * T
        Tk = t21 * F - t22 * T
        return Th, Tk

    def _step(self):
        q = self.d.sensordata[self.qa:self.qa + 4]
        w, x, y, z = q
        pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
        dpitch = self.d.sensordata[self.ga + 1]

        ctrl = np.zeros(6)
        for si, side in enumerate(['L', 'R']):
            sgn = 1 if side == 'L' else -1
            wv = self.d.qvel[self.jv[f'{side}_wheel_j']]

            # FK
            L0, theta0, t1, t2 = self._fk(side)

            # Virtual leg velocity (numerical differentiation with smoothing)
            dL0 = (L0 - self.prev_l0[side]) / self.dt
            dth0 = (theta0 - self.prev_th0[side]) / self.dt
            self.prev_l0[side] = L0
            self.prev_th0[side] = theta0

            # Virtual leg PD (from reference repo)
            F = KP_L0 * (self.l0_nom - L0) - KD_L0 * dL0 + FFORCE
            T = KP_THETA * (0 - theta0) - KD_THETA * dth0
            T += 15 * pitch + 3 * dpitch  # pitch correction from IMU

            F = max(-60, min(60, F))
            T = max(-15, min(15, T))

            # Jacobian → joint torques
            Th, Tk = self._vmc(F, T, L0, theta0, t1, t2)

            # Wheel torque
            Tw = self.speed * 8 - 0.5 * wv + self.yaw * 3 * sgn
            Tw = max(-10, min(10, Tw))

            # Mirror for right leg
            Th *= sgn
            Tk *= sgn

            b = si * 3
            ctrl[b] = max(-10, min(10, Th))
            ctrl[b + 1] = max(-10, min(10, Tk))
            ctrl[b + 2] = Tw

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
            if cb_counter % 5 == 0:
                rclpy.spin_once(n, timeout_sec=0)
            if viewer is not None and cb_counter % 10 == 0:
                viewer.sync()
            nt += n.dt
            sl = nt - _time.monotonic()
            if sl > 0:
                _time.sleep(sl)
            else:
                nt = _time.monotonic()
                _time.sleep(0.0005)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
