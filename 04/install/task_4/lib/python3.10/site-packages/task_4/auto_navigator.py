#!/usr/bin/env python3

import sys
import os
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist
from std_msgs.msg import Float32

from PIL import Image, ImageOps

import numpy as np
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
"""

import yaml
import pandas as pd

from copy import copy, deepcopy
import time
from graphviz import Graph
from threading import Thread


from ament_index_python.packages import get_package_share_directory



class Map():
    def __init__(self, map_name):
        self.map_im, self.map_df, self.limits = self.__open_map(map_name)
        self.image_array = self.__get_obstacle_map(self.map_im, self.map_df)

        self.resolution = self.map_df.resolution[0]  # meters per pixel
        self.origin = self.map_df.origin[0]          # [x, y, theta]

    """
    def __repr__(self):
        fig, ax = plt.subplots(dpi=150)
        ax.imshow(self.image_array,extent=self.limits, cmap=cm.gray)
        ax.plot()
        return ""
    """

    """
    def __open_map(self,map_name):
        # Open the YAML file which contains the map name and other
        # configuration parameters
        f = open(map_name + '.yaml', 'r')
        map_df = pd.json_normalize(yaml.safe_load(f))
        # Open the map image
        map_name = map_df.image[0]
        im = Image.open(map_name)
        size = 200, 200
        im.thumbnail(size)
        im = ImageOps.grayscale(im)
        # Get the limits of the map. This will help to display the map
        # with the correct axis ticks.
        xmin = map_df.origin[0][0]
        xmax = map_df.origin[0][0] + im.size[0] * map_df.resolution[0]
        ymin = map_df.origin[0][1]
        ymax = map_df.origin[0][1] + im.size[1] * map_df.resolution[0]

        return im, map_df, [xmin,xmax,ymin,ymax]
    """
        
    def __open_map(self, map_name):
        # Handle full path vs relative
        if not os.path.isabs(map_name):
            map_name = os.path.join(os.getcwd(), map_name)
        if not os.path.exists(map_name):
            raise FileNotFoundError(f"Could not find map YAML at {map_name}")

        # Open YAML file
        with open(map_name, 'r') as f:
            map_df = pd.json_normalize(yaml.safe_load(f))

        # Resolve the image path relative to the YAML file’s folder
        yaml_dir = os.path.dirname(map_name)
        img_path = os.path.join(yaml_dir, map_df.image[0])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Could not find map image at {img_path}")

        # Load image and process
        im = Image.open(img_path)
        size = 200, 200
        im.thumbnail(size)
        im = ImageOps.grayscale(im)

        xmin = map_df.origin[0][0]
        xmax = map_df.origin[0][0] + im.size[0] * map_df.resolution[0]
        ymin = map_df.origin[0][1]
        ymax = map_df.origin[0][1] + im.size[1] * map_df.resolution[0]

        return im, map_df, [xmin, xmax, ymin, ymax]


    def __get_obstacle_map(self,map_im, map_df):
        img_array = np.reshape(list(self.map_im.getdata()),(self.map_im.size[1],self.map_im.size[0]))
        up_thresh = self.map_df.occupied_thresh[0]*255
        low_thresh = self.map_df.free_thresh[0]*255

        for j in range(self.map_im.size[0]):
            for i in range(self.map_im.size[1]):
                if img_array[i,j] > up_thresh:
                    img_array[i,j] = 255
                else:
                    img_array[i,j] = 0
        return img_array

class Queue():
    def __init__(self, init_queue = []):
        self.queue = copy(init_queue)
        self.start = 0
        self.end = len(self.queue)-1

    def __len__(self):
        numel = len(self.queue)
        return numel

    def __repr__(self):
        q = self.queue
        tmpstr = ""
        for i in range(len(self.queue)):
            flag = False
            if(i == self.start):
                tmpstr += "<"
                flag = True
            if(i == self.end):
                tmpstr += ">"
                flag = True

            if(flag):
                tmpstr += '| ' + str(q[i]) + '|\n'
            else:
                tmpstr += ' | ' + str(q[i]) + '|\n'

        return tmpstr

    def __call__(self):
        return self.queue

    def initialize_queue(self,init_queue = []):
        self.queue = copy(init_queue)

    def sort(self,key=str.lower):
        self.queue = sorted(self.queue,key=key)

    def push(self,data):
        self.queue.append(data)
        self.end += 1

    def pop(self):
        p = self.queue.pop(self.start)
        self.end = len(self.queue)-1
        return p

class GraphNode():
    def __init__(self,name):
        self.name = name
        self.children = []
        self.weight = []

    def __repr__(self):
        return self.name

    def add_children(self,node,w=None):
        if w == None:
            w = [1]*len(node)
        self.children.extend(node)
        self.weight.extend(w)

class Tree():
    def __init__(self,name):
        self.name = name
        self.root = 0
        self.end = 0
        self.g = {}
        self.g_visual = Graph('G')

    def __call__(self):
        for name,node in self.g.items():
            if(self.root == name):
                self.g_visual.node(name,name,color='red')
            elif(self.end == name):
                self.g_visual.node(name,name,color='blue')
            else:
                self.g_visual.node(name,name)
            for i in range(len(node.children)):
                c = node.children[i]
                w = node.weight[i]
                #print('%s -> %s'%(name,c.name))
                if w == 0:
                    self.g_visual.edge(name,c.name)
                else:
                    self.g_visual.edge(name,c.name,label=str(w))
        return self.g_visual

    def add_node(self, node, start = False, end = False):
        self.g[node.name] = node
        if(start):
            self.root = node.name
        elif(end):
            self.end = node.name

    def set_as_root(self,node):
        # These are exclusive conditions
        self.root = True
        self.end = False

    def set_as_end(self,node):
        # These are exclusive conditions
        self.root = False
        self.end = True

class AStar():
    def __init__(self, in_tree):
        self.in_tree = in_tree
        self.q = Queue()
        self.dist = {name: np.inf for name, node in in_tree.g.items()}
        self.h = {name: 0 for name, node in in_tree.g.items()}
        self.via = {name: None for name, node in in_tree.g.items()}

        # initialize queue
        for __, node in in_tree.g.items():
            self.q.push(node)

    """
    def __init__(self,in_tree):
        self.in_tree = in_tree
        self.q = Queue()
        self.dist = {name:np.inf for name,node in in_tree.g.items()}
        self.h = {name:0 for name,node in in_tree.g.items()}

        for name,node in in_tree.g.items():
            if isinstance(self.in_tree.start, str):
                start = tuple(map(int, self.in_tree.start.split(',')))
            else:
                start = self.in_tree.start  # assume tuple or int already

            if isinstance(self.in_tree.end, str):
                end = tuple(map(int, self.in_tree.end.split(',')))
            else:
                end = self.in_tree.end

            #start = tuple(map(int, name.split(',')))
            #end = tuple(map(int, self.in_tree.end.split(',')))
            self.h[name] = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)

        self.via = {name:0 for name,node in in_tree.g.items()}
        for __,node in in_tree.g.items():
            self.q.push(node)
    """

    def __get_f_score(self,node):
        # pass
        # Place code here (remove the pass
        # statement once you start coding)
        # return self.dist[idx] + self.h[idx]
        return self.dist[node.name] + self.h[node.name]



    def solve(self, sn, en):
        # pass
        # Place code here (remove the pass
        # statement once you start coding)
        self.dist[sn.name] = 0
        while len(self.q) > 0:
            # sort by f-score instead of distance
            self.q.sort(key=self.__get_f_score)
            u = self.q.pop()
            if u.name == en.name:
                break
            for i in range(len(u.children)):
                c = u.children[i]
                w = u.weight[i]
                new_dist = self.dist[u.name] + w
                if new_dist < self.dist[c.name]:
                    self.dist[c.name] = new_dist
                    self.via[c.name] = u.name

    def reconstruct_path(self,sn,en):
        #path = []
        #dist = 0
        # Place code here
        #return path,dist
        start_key = sn.name
        end_key = en.name
        dist = self.dist[end_key]
        u = end_key
        path = [u]
        while u != start_key:
            u = self.via[u]
            path.append(u)
        path.reverse()
        return path,dist

class MapProcessor():
    def __init__(self,name):
        self.map = Map(name)
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        self.map_graph = Tree(name)

    def __modify_map_pixel(self,map_array,i,j,value,absolute):
        if( (i >= 0) and
            (i < map_array.shape[0]) and
            (j >= 0) and
            (j < map_array.shape[1]) ):
            if absolute:
                map_array[i][j] = value
            else:
                map_array[i][j] += value

    def __inflate_obstacle(self,kernel,map_array,i,j,absolute):
        dx = int(kernel.shape[0]//2)
        dy = int(kernel.shape[1]//2)
        if (dx == 0) and (dy == 0):
            self.__modify_map_pixel(map_array,i,j,kernel[0][0],absolute)
        else:
            for k in range(i-dx,i+dx):
                for l in range(j-dy,j+dy):
                    self.__modify_map_pixel(map_array,k,l,kernel[k-i+dx][l-j+dy],absolute)

    def inflate_map(self,kernel,absolute=True):
        # Perform an operation like dilation, such that the small wall found during the mapping process
        # are increased in size, thus forcing a safer path.
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.map.image_array[i][j] == 0:
                    self.__inflate_obstacle(kernel,self.inf_map_img_array,i,j,absolute)
        r = np.max(self.inf_map_img_array)-np.min(self.inf_map_img_array)
        if r == 0:
            r = 1
        self.inf_map_img_array = (self.inf_map_img_array - np.min(self.inf_map_img_array))/r

    def get_graph_from_map(self):
        # Create the nodes that will be part of the graph, considering only valid nodes or the free space
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    node = GraphNode('%d,%d'%(i,j))
                    self.map_graph.add_node(node)
        # Connect the nodes through edges
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    if (i > 0):
                        if self.inf_map_img_array[i-1][j] == 0:
                            # add an edge up
                            child_up = self.map_graph.g['%d,%d'%(i-1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up],[1])
                    if (i < (self.map.image_array.shape[0] - 1)):
                        if self.inf_map_img_array[i+1][j] == 0:
                            # add an edge down
                            child_dw = self.map_graph.g['%d,%d'%(i+1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw],[1])
                    if (j > 0):
                        if self.inf_map_img_array[i][j-1] == 0:
                            # add an edge to the left
                            child_lf = self.map_graph.g['%d,%d'%(i,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_lf],[1])
                    if (j < (self.map.image_array.shape[1] - 1)):
                        if self.inf_map_img_array[i][j+1] == 0:
                            # add an edge to the right
                            child_rg = self.map_graph.g['%d,%d'%(i,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_rg],[1])
                    if ((i > 0) and (j > 0)):
                        if self.inf_map_img_array[i-1][j-1] == 0:
                            # add an edge up-left
                            child_up_lf = self.map_graph.g['%d,%d'%(i-1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_lf],[np.sqrt(2)])
                    if ((i > 0) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i-1][j+1] == 0:
                            # add an edge up-right
                            child_up_rg = self.map_graph.g['%d,%d'%(i-1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_rg],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j > 0)):
                        if self.inf_map_img_array[i+1][j-1] == 0:
                            # add an edge down-left
                            child_dw_lf = self.map_graph.g['%d,%d'%(i+1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_lf],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i+1][j+1] == 0:
                            # add an edge down-right
                            child_dw_rg = self.map_graph.g['%d,%d'%(i+1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_rg],[np.sqrt(2)])

    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        r = np.max(g)-np.min(g)
        sm = (g - np.min(g))*1/r
        return sm

    def rect_kernel(self, size, value):
        m = np.ones(shape=(size,size))
        return m

    def draw_path(self,path):
        path_tuple_list = []
        path_array = copy(self.inf_map_img_array)
        for idx in path:
            tup = tuple(map(int, idx.split(',')))
            path_tuple_list.append(tup)
            path_array[tup] = 0.5
        return path_array

class Navigation(Node):
    """! Navigation node class.
    This class should serve as a template to implement the path planning and
    path follower components to move the turtlebot from position A to B.
    """

    def __init__(self, node_name='Navigation'):
        """! Class constructor.
        @param  None.
        @return An instance of the Navigation class.
        """
        super().__init__(node_name)
        
        # Load parameters
        self.declare_parameter('map_file', '')

        # Try to get map_file from ROS parameter
        map_file = self.get_parameter('map_file').get_parameter_value().string_value

        # If not provided, use local default
        if not map_file:
            # Change this path to your actual map YAML path
            #default_map_path = os.path.join(os.path.dirname(__file__), 'maps', 'sync_classroom_map.yaml')
            #map_file = default_map_path
            pkg_share = get_package_share_directory('task_4')
            map_file = os.path.join(pkg_share, 'maps', 'sync_classroom_map.yaml')

            self.get_logger().warn(f"No map_file parameter found. Using default: {map_file}")
        else:
            # If relative path, make it absolute (useful when running directly)
            if not os.path.isabs(map_file):
                map_file = os.path.join(os.getcwd(), map_file)  
                self.declare_parameter('map_file', '')
                map_file = self.get_parameter('map_file').get_parameter_value().string_value

        # Initialize map processor
        self.mp = MapProcessor(map_file)
        kernel = self.mp.rect_kernel(5, 1)
        self.mp.inflate_map(kernel, True)
        self.mp.get_graph_from_map()
        self.get_logger().info(f"Loaded map and created graph with {len(self.mp.map_graph.g)} nodes")

        # Initialize planner
        self.astar = AStar(self.mp.map_graph)

        # Initialize state variables
        self.current_pose = None
        self.goal_pose = None
        self.path = None

        # Path planner/follower related variables
        self.path = Path()
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()
        self.start_time = 0.0

        # Subscribers
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.__goal_pose_cbk, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.__ttbot_pose_cbk, 10)

        # Publishers
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.calc_time_pub = self.create_publisher(Float32, 'astar_time',10) #DO NOT MODIFY

        # Node rate
        self.rate = self.create_rate(10)

    def __goal_pose_cbk(self, data):
        """! Callback to catch the goal pose.
        @param  data    PoseStamped object from RVIZ.
        @return None.
        """
        print(">>> __goal_pose_cbk triggered <<<")
        self.goal_pose = data
        self.get_logger().info(
            'goal_pose: {:.4f}, {:.4f}'.format(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y))
    
    def __ttbot_pose_cbk(self, data):
        #! Callback to catch the position of the vehicle.
        #@param  data    PoseWithCovarianceStamped object from amcl.
        #@return None.
        
        print(">>> __ttbot_pose_cbk triggered <<<")
        self.ttbot_pose = data.pose
        self.get_logger().info(
            'ttbot_pose: {:.4f}, {:.4f}'.format(self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y))
    """
    def __ttbot_pose_cbk(self, data):
        #! Callback to catch the position of the vehicle.
        print(">>> __ttbot_pose_cbk triggered <<<")
        # Convert AMCL pose to a PoseStamped
        self.ttbot_pose = PoseStamped()
        self.ttbot_pose.header = data.header
        self.ttbot_pose.pose = data.pose.pose  # ✅ Full PoseStamped

        self.get_logger().info(
            'ttbot_pose: {:.4f}, {:.4f}'.format(
                self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y
            )
        )
    """
    """
    def a_star_path_planner(self, start_pose, end_pose):
        #! A Start path planner.
        #@param  start_pose    PoseStamped object containing the start of the path to be created.
        #@param  end_pose      PoseStamped object containing the end of the path to be created.
        #@return path          Path object containing the sequence of waypoints of the created path.
        
        path = Path()
        self.get_logger().info(
            'A* planner.\n> start: {},\n> end: {}'.format(start_pose.pose.position, end_pose.pose.position))
        self.start_time = self.get_clock().now().nanoseconds*1e-9 #Do not edit this line (required for autograder)
        # TODO: IMPLEMENTATION OF THE A* ALGORITHM
        path.poses.append(start_pose)
        path.poses.append(end_pose)
        # Do not edit below (required for autograder)
        self.astarTime = Float32()
        self.astarTime.data = float(self.get_clock().now().nanoseconds*1e-9-self.start_time)
        self.calc_time_pub.publish(self.astarTime)
        
        return path
    """
    def a_star_path_planner(self, start_pose, end_pose):
        """! A* path planner (autograder-safe)"""
        path = Path()
        self.get_logger().info(
            'A* planner.\n> start: {},\n> end: {}'.format(
                start_pose.pose.position, end_pose.pose.position))

        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        # --- Convert world coordinates to map indices ---
        # Map image size
        im_h, im_w = self.mp.map.image_array.shape
        # Map limits
        xmin, xmax, ymin, ymax = self.mp.map.limits
        # Resolution
        res_x = (xmax - xmin) / im_w
        res_y = (ymax - ymin) / im_h

        def world_to_map(p):
            j = int((p.x - xmin) / res_x)
            i = int((p.y - ymin) / res_y)
            # Clip to bounds
            i = np.clip(i, 0, im_h - 1)
            j = np.clip(j, 0, im_w - 1)
            return i, j
        
        def nearest_free_node(i, j):
            """Return the nearest (i,j) that exists in the graph"""
            if f"{i},{j}" in self.mp.map_graph.g:
                return i, j
            # Simple BFS search in 8-connected neighbors
            from collections import deque
            visited = set()
            queue = deque([(i,j)])
            while queue:
                ci,cj = queue.popleft()
                if f"{ci},{cj}" in self.mp.map_graph.g:
                    return ci,cj
                visited.add((ci,cj))
                # 8 neighbors
                for di in [-1,0,1]:
                    for dj in [-1,0,1]:
                        ni, nj = ci+di, cj+dj
                        if 0<=ni<self.mp.map.image_array.shape[0] and 0<=nj<self.mp.map.image_array.shape[1]:
                            if (ni,nj) not in visited:
                                queue.append((ni,nj))
            raise ValueError("No free nodes found nearby")

        start_i, start_j = nearest_free_node(*world_to_map(start_pose.pose.position))
        goal_i, goal_j = nearest_free_node(*world_to_map(end_pose.pose.position))
        print(f"Start node: {start_i},{start_j}, Goal node: {goal_i},{goal_j}")

        start_name = f"{start_i},{start_j}"
        goal_name = f"{goal_i},{goal_j}"

        # --- Run A* ---
        astar_solver = AStar(self.mp.map_graph)
        start_node = self.mp.map_graph.g[start_name]
        goal_node = self.mp.map_graph.g[goal_name]
        astar_solver.solve(start_node, goal_node)
        node_path, _ = astar_solver.reconstruct_path(start_node, goal_node)

        print("A* node path:", node_path)
        print("Map limits:", self.mp.map.limits)
        print("Resolution:", (res_x, res_y))


        # --- Convert path nodes back to world coordinates ---
        for node_name in node_path:
            i, j = map(int, node_name.split(','))
            x = xmin + j * res_x
            y = ymin + i * res_y
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            path.poses.append(pose)

        # --- Autograder timing (do not edit) ---
        self.astarTime = Float32()
        self.astarTime.data = float(self.get_clock().now().nanoseconds*1e-9 - self.start_time)
        self.calc_time_pub.publish(self.astarTime)

        return path
   
    """
    def get_path_idx(self, path, vehicle_pose):
        #! Path follower.
        #@param  path                  Path object containing the sequence of waypoints of the created path.
        #@param  current_goal_pose     PoseStamped object containing the current vehicle position.
        #@return idx                   Position in the path pointing to the next goal pose to follow.
        
        idx = 0
        # TODO: IMPLEMENT A MECHANISM TO DECIDE WHICH POINT IN THE PATH TO FOLLOW idx <= len(path)
        return idx
    """

    def get_path_idx(self, path, vehicle_pose):
        """Return index of the next waypoint to follow."""
        if not path or len(path.poses) == 0:
            return 0

        # Current robot position
        x = vehicle_pose.pose.position.x
        y = vehicle_pose.pose.position.y

        # Compute distance to each path point
        dists = [np.hypot(p.pose.position.x - x, p.pose.position.y - y) for p in path.poses]

        # Pick closest point and move slightly ahead
        idx = int(np.argmin(dists))
        lookahead = 5  # waypoints ahead
        idx = min(idx + lookahead, len(path.poses) - 1)
        return idx

    """
    def path_follower(self, vehicle_pose, current_goal_pose):
        #! Path follower.
        #@param  vehicle_pose           PoseStamped object containing the current vehicle pose.
        #@param  current_goal_pose      PoseStamped object containing the current target from the created path. This is different from the global target.
        #@return path                   Path object containing the sequence of waypoints of the created path.
        
        speed = 0.0
        heading = 0.0
        # TODO: IMPLEMENT PATH FOLLOWER
        return speed, heading
    """
    def path_follower(self, vehicle_pose, current_goal_pose):
        """Compute linear and angular velocity toward next waypoint."""
        # Extract current and goal positions
        x = vehicle_pose.pose.position.x
        y = vehicle_pose.pose.position.y
        gx = current_goal_pose.pose.position.x
        gy = current_goal_pose.pose.position.y

        # Compute heading and distance
        dx = gx - x
        dy = gy - y
        distance = np.hypot(dx, dy)
        desired_yaw = np.arctan2(dy, dx)

        # Extract robot yaw from quaternion
        q = vehicle_pose.pose.orientation
        yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y**2 + q.z**2)
        )

        yaw_error = desired_yaw - yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))  # normalize

        # Control law (tune gains)
        k_lin = 0.3
        k_ang = 1.2
        speed = k_lin * distance
        heading = k_ang * yaw_error

        # Clamp velocities
        speed = np.clip(speed, -0.15, 0.25)
        heading = np.clip(heading, -1.5, 1.5)

        return speed, heading

    """
    def move_ttbot(self, speed, heading):
        #! Function to move turtlebot passing directly a heading angle and the speed.
        #@param  speed     Desired speed.
        #@param  heading   Desired yaw angle.
        #@return path      object containing the sequence of waypoints of the created path.
        
        cmd_vel = Twist()
        # TODO: IMPLEMENT YOUR LOW-LEVEL CONTROLLER
        cmd_vel.linear.x = speed
        cmd_vel.angular.z = heading

        self.cmd_vel_pub.publish(cmd_vel)
    """
    def move_ttbot(self, speed, heading):
        """Publish Twist command."""
        cmd_vel = Twist()
        cmd_vel.linear.x = float(np.clip(speed, -0.2, 0.3))
        cmd_vel.angular.z = float(np.clip(heading, -1.5, 1.5))
        self.cmd_vel_pub.publish(cmd_vel)

    """
    def run(self):
        #! Main loop of the node. You need to wait until a new pose is published, create a path and then
        #drive the vehicle towards the final pose.
        #@param none
        #@return none
        
        while rclpy.ok():
            # Call the spin_once to handle callbacks
            rclpy.spin_once(self, timeout_sec=0.1)  # Process callbacks without blocking

            # 1. Create the path to follow
            path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
            # 2. Loop through the path and move the robot
            idx = self.get_path_idx(path, self.ttbot_pose)
            current_goal = path.poses[idx]
            speed, heading = self.path_follower(self.ttbot_pose, current_goal)
            self.move_ttbot(speed, heading)

            self.rate.sleep()
            # Sleep for the rate to control loop timing
    """

    def run(self):
        """Main loop."""
        self.new_goal_received = False  # flag for new goals

        while rclpy.ok():
            # Process callbacks
            rclpy.spin_once(self, timeout_sec=0.1)

            # Wait for AMCL and goal
            if self.goal_pose is None:
                self.get_logger().info("Waiting for initial poses...")
                continue

            # Wait until they actually contain valid data (not all zeros)
            if (self.goal_pose.pose.position.x == 0.0 and self.goal_pose.pose.position.y == 0.0):
                continue

            # Plan once per goal
            if self.path is None or self.new_goal_received:
                self.get_logger().info("Computing new A* path...")
                self.path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
                self.new_goal_received = False
                self.path_idx = 0

            # Follow the path
            idx = self.get_path_idx(self.path, self.ttbot_pose)
            if idx >= len(self.path.poses):
                self.get_logger().info("Reached final goal!")
                self.move_ttbot(0.0, 0.0)
                self.path = None
                continue

            current_goal = self.path.poses[idx]
            speed, heading = self.path_follower(self.ttbot_pose, current_goal)
            self.move_ttbot(speed, heading)

            self.rate.sleep()


    

def main(args=None):
    rclpy.init(args=args)
    nav = Navigation(node_name='Navigation')

    try:
        nav.run()
    except KeyboardInterrupt:
        pass
    finally:
        nav.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()