#!/usr/bin/env python3
import rclpy
from geometry_msgs.msg import PoseStamped
from task_4.auto_navigator import Navigation  # adjust import path as needed
import matplotlib.pyplot as plt
import numpy as np
import os

def main(args=None):
    rclpy.init()
    
    map_path = os.path.expanduser("~/ME597/me597_lab3/04/task_4/maps/classroom_map.yaml")
    
    nav = Navigation(map_path)
    
    # Mock start and goal poses
    start_pose = PoseStamped()
    start_pose.pose.position.x = 10.0
    start_pose.pose.position.y = 15.0
    
    goal_pose = PoseStamped()
    goal_pose.pose.position.x = 40.0
    goal_pose.pose.position.y = 45.0
    
    # Call the planner
    path = nav.a_star_path_planner(start_pose, goal_pose)
    
    print("Path length:", len(path.poses))
    for p in path.poses:
        print(f"x: {p.pose.position.x:.2f}, y: {p.pose.position.y:.2f}")
    
    # Optional: plot path
    path_x = [p.pose.position.x for p in path.poses]
    path_y = [p.pose.position.y for p in path.poses]
    plt.imshow(nav.mp.inf_map_img_array, origin='lower', cmap='gray')
    plt.plot(path_y, path_x, 'r')  # note row/col swap
    plt.show()

if __name__ == '__main__':
    main()
