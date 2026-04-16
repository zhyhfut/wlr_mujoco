"""
WLR MuJoCo simulation — Open-chain model with balance controller.
"""

import math, os, numpy as np
import mujoco, rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock

L1, L2, OFFSET, WR = 0.07, 0.147, 0.0615, 0.05


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
        mujoco.mj_forward(self.m, self.d)

        self.jq = {}
        self.jv = {}
        for n in ['L_hip_j', 'L_knee_j', 'L_wheel_j', 'R_hip_j', 'R_knee_j', 'R_wheel_j']:
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
            self.jq[n] = self.m.jnt_qposadr[jid]
            self.jv[n] = self.m.jnt_dofadr[jid]

        sa = self.m.sensor_adr
        self.qa = sa[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, 'quat')]
        self.ga = sa[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, 'gyro')]
        self.base_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, 'base')

        self.speed = 0.0
        self.yaw = 0.0
        self.step = 0

        self.jp = self.create_publisher(JointState, '/joint_states', 10)
        self.ip = self.create_publisher(Imu, '/imu/data', 10)
        self.cp = self.create_publisher(Clock, '/clock', 10)
        self.create_subscription(Twist, '/cmd_vel', lambda m: (
            setattr(self, 'speed', m.linear.x), setattr(self, 'yaw', m.angular.z)), 10)

        # Timer-driven simulation (no threading, no GIL issues)
        self.create_timer(self.dt, self._step)

        print(f'[mujoco_sim] Started {sr:.0f}Hz', flush=True)

    def _step(self):
        q = self.d.sensordata[self.qa:self.qa + 4]
        w, x, y, z = q
        pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
        dpitch = self.d.sensordata[self.ga + 1]

        ctrl = np.zeros(6)
        for si, side in enumerate(['L', 'R']):
            sgn = 1 if side == 'L' else -1
            hip = self.d.qpos[self.jq[f'{side}_hip_j']]
            knee = self.d.qpos[self.jq[f'{side}_knee_j']]
            hip_v = self.d.qvel[self.jv[f'{side}_hip_j']]
            knee_v = self.d.qvel[self.jv[f'{side}_knee_j']]
            wv = self.d.qvel[self.jv[f'{side}_wheel_j']]

            wheel_t = -80 * pitch - 10 * dpitch - 0.5 * wv
            wheel_t += self.speed * 5 + self.yaw * 2 * sgn
            hip_t = -10 * hip_v
            knee_t = 200 * (0 - knee) - 30 * knee_v - 2.0

            hip_t = max(-10, min(10, hip_t))
            knee_t = max(-30, min(30, knee_t))
            wheel_t = max(-10, min(10, wheel_t))

            ctrl[si * 3] = hip_t
            ctrl[si * 3 + 1] = knee_t
            ctrl[si * 3 + 2] = wheel_t

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
        m.position = [self.d.qpos[self.jq[n]] for n in
                      ['L_hip_j', 'L_knee_j', 'L_wheel_j', 'R_hip_j', 'R_knee_j', 'R_wheel_j']]
        m.velocity = [self.d.qvel[self.jv[n]] for n in
                       ['L_hip_j', 'L_knee_j', 'L_wheel_j', 'R_hip_j', 'R_knee_j', 'R_wheel_j']]
        m.effort = [0.0] * 6
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
    rclpy.init(args=args)
    n = MuJoCoNode()
    import time as _time
    try:
        # Run simulation at precise 1kHz, process ROS callbacks every 5ms
        nt = _time.monotonic()
        cb_counter = 0
        while rclpy.ok():
            n._step()
            # Process ROS callbacks every 5 steps (200Hz) to reduce overhead
            cb_counter += 1
            if cb_counter % 5 == 0:
                rclpy.spin_once(n, timeout_sec=0)
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
