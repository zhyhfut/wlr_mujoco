from setuptools import setup
import os
from glob import glob

package_name = 'wlr_mujoco'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/model', glob('model/*.xml')),
    ],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'mujoco_node = wlr_mujoco.mujoco_node:main',
        ],
    },
)
