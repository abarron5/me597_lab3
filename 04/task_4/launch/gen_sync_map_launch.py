from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # Namespace argument (default is /robot for your class robot)
    #namespace = LaunchConfiguration('namespace', default='/robot')

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

    return LaunchDescription([
        slam_launch,
        view_robot_launch
    ])
