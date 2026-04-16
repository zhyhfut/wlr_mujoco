# wlr_mujoco

基于 MuJoCo 物理引擎的双轮腿机器人（WLR）仿真包，运行于 ROS 2 Jazzy。

使用开链模型（hip + knee + wheel）模拟双轮足机器人，通过 wheel 驱动的倒立摆控制器实现机身平衡。

## 依赖

- ROS 2 Jazzy
- Python 3.12
- `mujoco` (pip install mujoco)
- `numpy`

## 构建

```bash
cd ~/wlr_ws
colcon build --packages-select wlr_mujoco
source install/setup.bash
```

## 运行

```bash
# 先清理旧进程
pkill -9 -f mujoco_node; pkill -9 -f 'ros2 launch'; sleep 1

# 无头模式（终端运行，无可视窗口）
ros2 launch wlr_mujoco sim.launch.py

# 可视模式（弹出 MuJoCo Viewer 窗口，需要图形界面）
ros2 launch wlr_mujoco sim_visual.launch.py
```

可视模式会在弹窗中显示机器人的物理仿真效果，可以旋转/缩放视角。

## 测试平衡效果

新终端（启动后等 8 秒）：

```bash
source ~/wlr_ws/install/setup.bash

timeout 10 python3 -c "
import rclpy, math
from rclpy.node import Node
from sensor_msgs.msg import Imu
rclpy.init()
n = Node('chk')
p = []
def cb(m):
    q = m.orientation
    p.append(math.degrees(math.asin(max(-1,min(1,2*(q.w*q.y-q.z*q.x))))))
n.create_subscription(Imu, '/imu/data', cb, 10)
import time; s = time.time()
while time.time()-s < 8: rclpy.spin_once(n, timeout_sec=0.1)
if p:
    mx = max(abs(x) for x in p)
    print(f'avg={sum(p)/len(p):.1f} max={mx:.1f}')
    print('稳定' if mx < 15 else '不稳定')
"
```

预期输出：`avg=-6.0 max=12.0` → 基本稳定

## Topic

| Topic | 方向 | 类型 | 频率 | 说明 |
|-------|------|------|------|------|
| `/joint_states` | 发布 | JointState | 200Hz | 6 关节位置/速度 |
| `/imu/data` | 发布 | Imu | 200Hz | 四元数 + 角速度 |
| `/clock` | 发布 | Clock | 200Hz | 仿真时间 |
| `/cmd_vel` | 订阅 | Twist | — | linear.x 前进速度, angular.z 转弯 |

## 发送速度指令

```bash
# 前进
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.1}}" --once

# 转弯
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{angular: {z: 0.3}}" --once

# 停止
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{}" --once
```

## 模型说明

开链模型（每侧腿）：

```
base_link (1.5kg, 210x155x10mm)
  ├── L_hip (0.07m 上连杆, 0.04kg)
  │     └── L_knee (0.147m 下连杆, 0.072kg)
  │           └── L_wheel (R=0.05m, 0.34kg)
  └── R_hip（镜像）
```

总质量：2.4 kg
站立高度：0.267 m
髋关节间距（Y）：0.175 m

## 控制器

倒立摆式 wheel 平衡控制：

- **Wheel**: `-80 * pitch - 10 * dpitch`（主要平衡力矩）
- **Knee**: `200 * (0 - knee) - 30 * knee_v - 2.0`（保持腿部伸直）
- **Hip**: `-10 * hip_v`（纯阻尼）

仿真频率 1000Hz，发布频率 200Hz。

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sim_rate` | 1000 | 仿真频率 (Hz) |
| `pub_rate` | 200 | 发布频率 (Hz) |
| `model_path` | 自动 | MJCF 模型路径 |
| `use_sim_time` | true | 使用仿真时间 |

## 文件结构

```
wlr_mujoco/
├── package.xml                  # ROS 2 包声明
├── setup.py                     # Python 安装配置
├── setup.cfg                    # 安装路径配置
├── resource/wlr_mujoco          # ament 索引标记
├── model/
│   └── wlr_robot.xml            # MuJoCo MJCF 模型
├── launch/
│   ├── sim.launch.py            # 无头模式启动文件
│   └── sim_visual.launch.py     # 可视模式启动文件
├── wlr_mujoco/
│   ├── __init__.py              # 包初始化
│   └── mujoco_node.py           # 仿真节点 + 控制器
└── README.md
```

## 已知限制

- 机身高度控制受限（knee 力矩不足以完全支撑机身，body 略低于设计高度）
- 控制器增益硬编码在源码中，需编辑 `mujoco_node.py` 调参
- 无可视化界面（无 MuJoCo viewer 集成）
- `mujoco` 和 `numpy` 未在 `package.xml` 中声明依赖

## 调参

编辑 `wlr_mujoco/mujoco_node.py` 中 `_step()` 方法内的增益：

```python
wheel_t = -80 * pitch - 10 * dpitch - 0.5 * wv  # pitch 控制
knee_t = 200 * (0 - knee) - 30 * knee_v - 2.0   # 腿部支撑
hip_t = -10 * hip_v                               # 髋阻尼
```

## 许可

MIT
