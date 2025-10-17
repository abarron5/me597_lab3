from setuptools import find_packages, setup

import os
from glob import glob

package_name = 'task_4'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*'))),
        (os.path.join('share', package_name, 'maps'), glob(os.path.join('maps', '*')))
    ],
    install_requires=['setuptools', 'pandas', 'pyyaml', 'Pillow', 'matplotlib', 'numpy'],
    zip_safe=True,
    maintainer='me597',
    maintainer_email='barron49@purdue.edu',
    description='SLAM lab',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'auto_navigator = task_4.auto_navigator:main', #.py?
            'map_test = task_4.map_test:main',
            'test = task_4.test:main',
        ],
    },
)
