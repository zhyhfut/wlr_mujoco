"""
WLR MuJoCo simulation — Open-chain model with PD balance controller.

Controller strategy (proven effective):
- Wheel: pitch PD control (inverted pendulum)
- Knee: keep straight with gravity compensation
- Hip: velocity damping only
"""

import math, os, numpy as np
import mujoco, rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock


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

        # Joint/sensor indices
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

        mujoco.mj_forward(self.m, self.d)

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
        print(f'[mujoco_sim] Started {sr:.0f}Hz', flush=True)

    def _step(self):
        q = self.d.sensordata[self.qa:self.qa + 4]
        w, x, y, z = q
        pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
        dpitch = self.d.sensordata[self.ga + 1]
        body_z = self.d.xpos[self.base_id][2]

        ctrl = np.zeros(6)
        for si, side in enumerate(['L', 'R']):
            sgn = 1 if side == 'L' else -1
            knee = self.d.qpos[self.jq[f'{side}_knee_j']]
            knee_v = self.d.qvel[self.jv[f'{side}_knee_j']]
            wv = self.d.qvel[self.jv[f'{side}_wheel_j']]
            hip_v = self.d.qvel[self.jv[f'{side}_hip_j']]

            # Wheel: LQR pitch control (computed from inverted pendulum model)
            # State: [theta, dtheta], A=[[0,1],[mgl/I,0]], B=[[0],[l/(IR)]]
            # Q=[[500,0],[0,10]], R=1 → K=[22.71, 3.18]
            wheel_t = -22.7 * pitch - 3.2 * dpitch - 0.3 * wv
            wheel_t += self.speed * 5 + self.yaw * 2 * sgn

            # Hip: passive damping
            hip_t = -10 * hip_v

            # Knee: keep straight + gravity compensation
            knee_t = 200 * (0 - knee) - 30 * knee_v - 1.0

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
            # Process ROS callbacks every 100 steps (10Hz) to minimize interference
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
