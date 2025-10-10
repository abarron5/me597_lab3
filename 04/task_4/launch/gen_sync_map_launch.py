"""
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Namespace argument (default is /robot for your class robot)
    #namespace = LaunchConfiguration('namespace', default='/robot')
    namespace = LaunchConfiguration('namespace', default='')


    #include .yaml maps from maps folder
    pkg_share = get_package_share_directory('task_4')
    map_path = os.path.join(pkg_share, 'maps', 'classroom_map.yaml')

    # Include the SLAM launch file from turtlebot4_navigation
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('turtlebot4_navigation'),
                'launch',
                'slam.launch.py'
            ])
        ]),
        launch_arguments={'namespace': namespace}.items()
    )

    # Include the RViz visualization launch file from turtlebot4_viz
    view_robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('turtlebot4_viz'),
                'launch',
                'view_robot.launch.py'
            ])
        ]),
        launch_arguments={'namespace': namespace}.items()
    )

    auto_nav_node = Node(
        package='task_4',
        executable='auto_navigator',
        name='auto_navigator',
        output='screen',
        parameters=[{'map_file': map_path}]
    )

    return LaunchDescription([
        slam_launch,
        view_robot_launch,
        auto_nav_node
    ])
"""
"""
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Namespace argument (default is empty, can be set for hardware)
    namespace = LaunchConfiguration('namespace', default='')

    # Map file
    pkg_share = get_package_share_directory('task_4')
    map_path = os.path.join(pkg_share, 'maps', 'classroom_map.yaml')

    # Include SLAM launch
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('turtlebot4_navigation'),
                'launch',
                'slam.launch.py'
            ])
        ]),
        launch_arguments={'namespace': namespace}.items()
    )

    # Include RViz launch
    view_robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('turtlebot4_viz'),
                'launch',
                'view_robot.launch.py'
            ])
        ]),
        launch_arguments={'namespace': namespace}.items()
    )

    # Map server node
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        namespace=namespace,
        output='screen',
        parameters=[{'yaml_filename': map_path}]
    )

    # Auto navigator node
    auto_nav_node = Node(
        package='task_4',
        executable='auto_navigator',
        name='auto_navigator',
        namespace=namespace,
        output='screen',
        parameters=[{'map_file': map_path}]
    )

    return LaunchDescription([
        slam_launch,
        view_robot_launch,
        map_server_node,   # Add this
        auto_nav_node
    ])
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    namespace = LaunchConfiguration('namespace', default='robot')
    pkg_share = get_package_share_directory('task_4')
    map_path = os.path.join(pkg_share, 'maps', 'classroom_map.yaml')
    
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        namespace=namespace,
        output='screen',
        parameters=[{'yaml_filename': map_path}]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        namespace=namespace,
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(pkg_share, 'rviz', 'nav2.rviz')]
    )

    auto_nav_node = Node(
        package='task_4',
        executable='auto_navigator',
        namespace=namespace,
        name='auto_navigator',
        output='screen',
        parameters=[{'map_file': map_path}]
    )

    return LaunchDescription([
        map_server_node,
        rviz_node,
        auto_nav_node
    ])

