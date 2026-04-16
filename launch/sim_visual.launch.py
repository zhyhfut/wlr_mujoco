from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_mujoco = get_package_share_directory('wlr_mujoco')
    model_path = os.path.join(pkg_mujoco, 'model', 'wlr_robot.xml')

    log_format = SetEnvironmentVariable(
        'RCUTILS_LOGGING_FORMAT',
        '[{severity}] [{name}]: {message}'
    )

    mujoco_node = Node(
        package='wlr_mujoco',
        executable='mujoco_node',
        name='mujoco_sim',
        parameters=[{
            'use_sim_time': True,
            'model_path': model_path,
            'sim_rate': 1000.0,
            'pub_rate': 200.0,
        }],
        arguments=['--visual'],
        output='screen',
    )

    return LaunchDescription([
        log_format,
        mujoco_node,
    ])
