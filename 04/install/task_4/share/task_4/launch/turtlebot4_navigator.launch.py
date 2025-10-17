from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # ------------------------------------------------------------
    # 1️⃣ Set up map path and namespace
    # ------------------------------------------------------------
    map_path = os.path.join(
        get_package_share_directory('task_4'),
        'maps',
        'classroom_map.yaml'
    )
    namespace = '/robot'

    # ------------------------------------------------------------
    # 2️⃣ Include localization.launch.py from turtlebot4_navigation
    # ------------------------------------------------------------
    turtlebot4_nav_dir = get_package_share_directory('turtlebot4_navigation')
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot4_nav_dir, 'launch', 'localization.launch.py')
        ),
        launch_arguments={
            'map': map_path,
            'namespace': namespace
        }.items()
    )

    # ------------------------------------------------------------
    # 3️⃣ Include view_robot.launch.py from task_4
    # ------------------------------------------------------------
    task4_dir = get_package_share_directory('task_4')
    view_robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(task4_dir, 'launch', 'view_robot.launch.py')
        ),
        launch_arguments={
            'map': map_path,
            'namespace': namespace
        }.items()
    )

    # ------------------------------------------------------------
    # 4️⃣ Launch auto_navigator node from task_4 package
    # ------------------------------------------------------------
    auto_navigator_node = Node(
        package='task_4',
        executable='auto_navigator',
        name='auto_navigator',
        namespace=namespace,
        output='screen',
        remappings=[
            ('/cmd_vel', '/robot/cmd_vel'),
            ('/amcl_pose', '/robot/amcl_pose'),
        ]
    )

    # ------------------------------------------------------------
    # 5️⃣ Return launch description
    # ------------------------------------------------------------
    return LaunchDescription([
        localization_launch,
        view_robot_launch,
        auto_navigator_node
    ])
