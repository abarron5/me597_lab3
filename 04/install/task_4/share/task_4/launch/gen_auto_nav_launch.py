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

    auto_nav_node = Node(
        package='task_4',
        executable='auto_navigator',
        namespace=namespace,
        name='auto_navigator',
        output='screen',
        parameters=[{'map_file': map_path}]
    )

    return LaunchDescription([
        auto_nav_node
    ])

