"""
from task_4.auto_navigator import Map, MapProcessor, Navigation
import matplotlib.pyplot as plt
import os

import rclpy
from geometry_msgs.msg import PoseStamped

#from rclpy.node import Node

def make_pose(x, y):
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.orientation.w = 1.0
    return pose

def main():
    rclpy.init()  # <-- Add this before creating any Node!

    # Expand the path to the map file
    
    map_path = os.path.expanduser("~/ME597/me597_lab3/04/task_4/maps/classroom_map.yaml")
    
        # Step 1
    m = Map(map_path)
    print("Map loaded successfully.")
    print("Map limits:", m.limits)
    print("Array shape:", m.image_array.shape)
    print(m)

    plt.figure()  # 👈 create a new figure window
    plt.imshow(m.image_array, extent=m.limits, cmap='gray')
    plt.title("Original Map")
    plt.show(block=False)

    # Step 2
    mp = MapProcessor(map_path)
    kernel = mp.rect_kernel(3, 1)
    mp.inflate_map(kernel)

    plt.figure()  # 👈 another new figure
    plt.imshow(mp.inf_map_img_array)
    plt.title("Inflated Map")
    plt.show(block=False)

    mp.get_graph_from_map()
    print("Graph nodes:", len(mp.map_graph.g))

    start = "10,15"
    goal = "40,45"

    nav = Navigation(map_path, 'navigation_node')  # give it a simple name
    #start = [10, 15]
    #goal = [40, 45]
    #path = nav.find_path(start, goal)

    start_pose = make_pose(10, 15)
    goal_pose = make_pose(40, 45)
    path = nav.a_star_path_planner(start_pose, goal_pose)
    mp.display_path(path)

    plt.show()


    # You can now use nav or spin it if needed
    # rclpy.spin(nav)
    nav.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
"""

#!/usr/bin/env python3
import matplotlib.pyplot as plt
from task_4.auto_navigator import Map, MapProcessor

map_path = "/home/me597/ME597/me597_lab3/04/task_4/maps/classroom_map.yaml"

# Step 1 — Load map
m = Map(map_path)
print("Map loaded successfully.")
print("Map limits:", m.limits)
print("Array shape:", m.image_array.shape)

plt.figure("Original Map")
plt.imshow(m.image_array, extent=m.limits, cmap='gray')
plt.title("Original Map")

# Step 2 — Inflate map
mp = MapProcessor(map_path)
kernel = mp.rect_kernel(3, 1)
mp.inflate_map(kernel)

plt.figure("Inflated Map")
plt.imshow(mp.inf_map_img_array)
plt.title("Inflated Map")

# Step 3 — Show both
plt.show()

