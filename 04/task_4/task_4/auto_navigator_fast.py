#!/usr/bin/env python3

import sys
import os
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from std_msgs.msg import Float32

from PIL import Image, ImageOps

import matplotlib.pyplot as plt

import yaml
import pandas as pd

from copy import copy
import time
from graphviz import Graph

import math

from ament_index_python.packages import get_package_share_directory

from scipy.ndimage import convolve

import heapq
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    from scipy import ndimage as _ndimage
    _HAS_SCIPY_ND = True
except Exception:
    _HAS_SCIPY_ND = False


class Map():
    def __init__(self, map_name):
        self.map_im, self.map_df, self.limits, self.size = self.__open_map(map_name)
        self.image_array = self.__get_obstacle_map(self.map_im, self.map_df)

        self.resolution = self.map_df.resolution[0]  # meters per pixel
        self.origin = self.map_df.origin[0]          # [x, y, theta]
       
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
        size = im.size
        print(im.size)
        #size = 200, 200
        #im.thumbnail(size)
        im = ImageOps.grayscale(im)

        xmin = map_df.origin[0][0]
        xmax = map_df.origin[0][0] + im.size[0] * map_df.resolution[0]
        ymin = map_df.origin[0][1]
        ymax = map_df.origin[0][1] + im.size[1] * map_df.resolution[0]

        return im, map_df, [xmin, xmax, ymin, ymax], size


    def __get_obstacle_map(self,map_im, map_df):
        img_array = np.reshape(list(self.map_im.getdata()),(self.map_im.size[1],self.map_im.size[0]))
        up_thresh = self.map_df.occupied_thresh[0]*255
        low_thresh = self.map_df.free_thresh[0]*255

        img_array = np.array(self.map_im)
        img_array = np.where(img_array > up_thresh, 255, 0).astype(np.uint8)
        
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
        
    def __get_f_score(self,node):
        # pass
        # Place code here (remove the pass
        # statement once you start coding)
        # return self.dist[idx] + self.h[idx]
        return self.dist[node.name] + self.h[node.name]

    def solve(self, sn, en):
        self.dist[sn.name] = 0
        start_time = time.time()
        last_print_time = start_time
        processed_nodes = 0
        total_nodes = len(self.dist)  # total number of nodes in the graph

        print("fast")
        print("Planning path...", end="", flush=True)

        while len(self.q) > 0:
            # sort by f-score instead of distance
            self.q.sort(key=self.__get_f_score)
            u = self.q.pop()
            processed_nodes += 1

            # periodically update progress (every 0.5 s)
            now = time.time()
            if now - last_print_time > 0.5:
                percent = 100 * processed_nodes / total_nodes
                elapsed = now - start_time
                print(f"\rPlanning path... {percent:.1f}% ({elapsed:.2f} s)", end="", flush=True)
                last_print_time = now

            if u.name == en.name:
                break

            for i in range(len(u.children)):
                c = u.children[i]
                w = u.weight[i]
                new_dist = self.dist[u.name] + w
                if new_dist < self.dist[c.name]:
                    self.dist[c.name] = new_dist
                    self.via[c.name] = u.name

        elapsed = time.time() - start_time
        print(f"\rPath planning complete! Total time: {elapsed:.2f} s       ")


    def solve_1(self, sn, en):
        self.dist[sn.name] = 0
        start_time = time.time()
        last_print_time = start_time
        print("Planning path...", end="", flush=True)
        

        while len(self.q) > 0:
            # sort by f-score instead of distance
            self.q.sort(key=self.__get_f_score)
            u = self.q.pop()

            # periodically update progress (every 0.5 s)
            now = time.time()
            if now - last_print_time > 0.5:
                elapsed = now - start_time
                print(f"\rPlanning path... {elapsed:.2f} s", end="", flush=True)
                last_print_time = now

            if u.name == en.name:
                break

            for i in range(len(u.children)):
                c = u.children[i]
                w = u.weight[i]
                new_dist = self.dist[u.name] + w
                if new_dist < self.dist[c.name]:
                    self.dist[c.name] = new_dist
                    self.via[c.name] = u.name

        elapsed = time.time() - start_time
        print(f"\rPath planning complete! Total time: {elapsed:.2f} s")

    def reconstruct_path(self, start_key, goal_key):
        path = []
        u = goal_key
        while u != start_key:
            path.append(u)
            u = self.via.get(u)
            if u is None:
                raise KeyError("Goal not reachable from start")
        path.append(start_key)
        path.reverse()
        #print("\nPath in map coordinates:")
        #print(path)
        return path, len(path)

# ---- A* over a grid using heapq (no huge graph creation) ----
class AStarGrid:
    def __init__(self, free_mask, diagonal=True):
        """
        free_mask: 2D boolean numpy array, True => free cell, False => occupied
        diagonal: whether to allow 8-connected moves
        """
        self.free = free_mask
        self.rows, self.cols = free_mask.shape
        self.diagonal = diagonal

        # 8-connected neighbor offsets and their move cost
        if diagonal:
            self.neigh_offsets = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
            self.neigh_costs = {(-1,0):1,(1,0):1,(0,-1):1,(0,1):1,
                                (-1,-1):math.sqrt(2),(-1,1):math.sqrt(2),(1,-1):math.sqrt(2),(1,1):math.sqrt(2)}
        else:
            self.neigh_offsets = [(-1,0),(1,0),(0,-1),(0,1)]
            self.neigh_costs = {(-1,0):1,(1,0):1,(0,-1):1,(0,1):1}

    def in_bounds(self, node):
        i,j = node
        return 0 <= i < self.rows and 0 <= j < self.cols

    def is_free(self, node):
        i,j = node
        return self.free[i, j]

    def neighbors(self, node):
        i,j = node
        for di,dj in self.neigh_offsets:
            ni, nj = i+di, j+dj
            nb = (ni, nj)
            if not self.in_bounds(nb):
                continue
            if not self.is_free(nb):
                continue
            yield nb, self.neigh_costs[(di,dj)]

    def heuristic(self, a, b):
        # Euclidean heuristic, admissible for 8-connected grid with diagonal cost sqrt(2)
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def solve(self, start, goal, max_nodes=np.inf):
        """
        start, goal: (i,j) tuples in grid coordinates
        Returns: path as list of (i,j) from start to goal or raises KeyError if unreachable
        """
        if not self.in_bounds(start) or not self.in_bounds(goal):
            raise KeyError("start or goal out of bounds")
        if not self.is_free(start) or not self.is_free(goal):
            raise KeyError("start or goal is not free")

        # priority queue items: (f_score, g_score, node)
        open_heap = []
        g_score = {start: 0.0}
        f_start = self.heuristic(start, goal)
        heapq.heappush(open_heap, (f_start, 0.0, start))
        came_from = {}

        explored = 0

        while open_heap:
            f, g, current = heapq.heappop(open_heap)
            explored += 1
            if current == goal:
                # reconstruct
                path = []
                u = current
                while u != start:
                    path.append(u)
                    u = came_from[u]
                path.append(start)
                path.reverse()
                return path

            if explored > max_nodes:
                break

            # if this heap entry is stale, skip
            if g > g_score.get(current, np.inf):
                continue

            for nb, cost in self.neighbors(current):
                tentative_g = g + cost
                if tentative_g < g_score.get(nb, np.inf):
                    g_score[nb] = tentative_g
                    f_nb = tentative_g + self.heuristic(nb, goal)
                    came_from[nb] = current
                    heapq.heappush(open_heap, (f_nb, tentative_g, nb))

        raise KeyError("Goal not reachable from start")

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
    
    def inflate_map(self, kernel, absolute=True):
        """
        Create a binary inflated occupancy map (self.inf_map_img_array).
        kernel: either a 2D numpy array (structuring element) or an int size.
        Result: self.inf_map_img_array is float array with 1.0 in inflated obstacle
        cells and 0.0 in free cells (shape = map image shape).
        Uses OpenCV.dilate if available (fast), otherwise scipy.ndimage.binary_dilation.
        """
        # build binary obstacle mask: True where occupied in original (original used 0 for occupied)
        obstacle_mask = (self.map.image_array == 0).astype(np.uint8)  # 1 = obstacle, 0 = free

        # normalize kernel form
        if isinstance(kernel, np.ndarray):
            kern = kernel.astype(np.uint8)
            # if kernel contains non-binary values, threshold it
            kern = (kern > 0).astype(np.uint8)
        else:
            # kernel is integer size
            ksize = int(kernel)
            kern = np.ones((ksize, ksize), dtype=np.uint8)

        # Use cv2 if available for very fast morphological dilation
        if _HAS_CV2:
            inflated = cv2.dilate(obstacle_mask, kern, iterations=1)
            # converted to float 0/1
            inflated = inflated.astype(np.float32)
        elif _HAS_SCIPY_ND:
            # scipy.ndimage.binary_dilation accepts boolean structure
            inflated = _ndimage.binary_dilation(obstacle_mask.astype(bool), structure=kern)
            inflated = inflated.astype(np.float32)
        else:
            # Fallback pure-numpy convolution-like approach (less efficient but safe)
            # We'll do a summed-window approach using convolution via FFT (numpy)
            # Convert kernel to 0/1 and perform correlate using numpy FFT (works but slower)
            obst = obstacle_mask.astype(np.float32)
            # pad sizes
            ky, kx = kern.shape
            pad_y, pad_x = ky//2, kx//2
            padded = np.pad(obst, ((pad_y,pad_y), (pad_x,pad_x)), mode='constant', constant_values=0)
            # sliding-window sum via striding -- memory-efficient vectorized implementation
            out = np.zeros_like(obst)
            for dy in range(ky):
                for dx in range(kx):
                    if kern[dy,dx]:
                        out += padded[dy:dy+obst.shape[0], dx:dx+obst.shape[1]]
            inflated = (out > 0).astype(np.float32)

        # store inflated map (1.0 = obstacle, 0.0 = free)
        self.inf_map_img_array = inflated

    # ---- simple helper to get free_mask used by A* ----
    def get_free_mask(self):
        # Return a boolean grid: True == free, False == obstacle
        return (self.inf_map_img_array == 0)
    
    def inflate_map_old_working(self, kernel, absolute=True):
        obstacle_mask = (self.map.image_array == 0).astype(np.uint8)
        inflated = convolve(obstacle_mask, kernel, mode='constant', cval=0)
        inflated = inflated / np.max(inflated)
        self.inf_map_img_array = inflated

    def inflate_map_old(self,kernel,absolute=True):
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
    
    def circular_kernel(self, radius):
        """
        Create a circular kernel of given radius in pixels.
        Returns a 2D numpy array of 1s inside the circle, 0s outside.
        """
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x**2 + y**2 <= radius**2
        return mask.astype(np.uint8)
    
    def draw_path(self,path):
        path_tuple_list = []
        path_array = copy(self.inf_map_img_array)
        
        if not hasattr(path, 'poses') or not path.poses:
            return path_array

        for pose_stamped in path.poses:
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y

            if 0 <= px < path_array.shape[1] and 0 <= py < path_array.shape[0]:
                path_array[py, px] = 0.5
                path_tuple_list.append((px, py))

        return path_array
    
    def draw_path_wrld(self, path, world_to_map_fn):
        path_array = copy(self.inf_map_img_array)

        if not hasattr(path, 'poses') or not path.poses:
            return path_array

        for pose_stamped in path.poses:
            i, j = world_to_map_fn(pose_stamped)

            if 0 <= i < path_array.shape[0] and 0 <= j < path_array.shape[1]:
                path_array[i, j] = 0.5

        return path_array


    def draw_path_wrld_cpy(self, path, world_to_map_fn):
        """Draw a nav_msgs/Path onto the map image array."""
        path_array = copy(self.inf_map_img_array)
        path_tuple_list = []

        if not hasattr(path, 'poses') or not path.poses:
            return path_array

        for pose_stamped in path.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y

            # Use converter function from Navigation
            px, py = world_to_map_fn(x, y)
            #px = px + int(1/0.05)
            #py = py

            if 0 <= px < path_array.shape[1] and 0 <= py < path_array.shape[0]:
                path_array[py, px] = 0.5
                path_tuple_list.append((px, py))

        return path_array
    
    def display_map_with_path(self, path_image, size, start=None, goal=None):
        #plt.figure(figsize=(8, 8))
        plt.figure()
        plt.imshow(self.inf_map_img_array, cmap='gray', origin='upper')
        plt.imshow(path_image, cmap='plasma', alpha=0.6, origin='upper')

        # Mark start and goal if provided
        print("Display: ", size)
        print("Start: ", start)
        if start is not None:
            plt.plot(start[1], start[0], 'go', label='Start')  # note swapped indices
        if goal is not None:
            plt.plot(goal[1], goal[0], 'ro', label='Goal')

        plt.title("Map with Planned Path")
        plt.legend()
        plt.axis("equal")
        plt.show(block=False)
        plt.pause(0.001)
    

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
        kernel = self.mp.rect_kernel(9, 9)
        #robot_radius_m = 0.15      # meters
        #res = self.mp.map.map_df.resolution[0]  # map resolution in meters/pixel
        #radius_px = int(robot_radius_m / res)
        #kernel = self.mp.circular_kernel(radius_px)
        #self.mp.inflate_map(kernel, absolute=True)


        self.mp.inflate_map(kernel, True)
        self.mp.get_graph_from_map()

        self.free_mask = self.mp.get_free_mask()
        self.astar = AStarGrid(self.free_mask, diagonal=True)

        self.get_logger().info(f"Loaded map and created graph with {len(self.mp.map_graph.g)} nodes")
        
        # Initialize planner
        self.astar = AStarGrid(self.mp.get_free_mask(), diagonal=True)

        # Initialize state variables
        self.new_goal_received = False
        self.current_pose = None
        self.goal_pose = None
        self.path = None
        self.ttbot_pose_x = 0.0
        self.ttbot_pose_y = 0.0
        self.ttbot_yaw = 0.0

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
        # ! Callback to catch the goal pose.
        # @param  data    PoseStamped object from RVIZ.
        # @return None.
        
        print(">>> __goal_pose_cbk triggered <<<")
        self.goal_pose = data
        self.new_goal_received = True
        self.get_logger().info(
            'goal_pose: {:.4f}, {:.4f}'.format(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y))
    
    def __ttbot_pose_cbk(self, data):
        #print(">>> __ttbot_pose_cbk triggered <<<")
        self.ttbot_pose_received = True
        self.ttbot_pose = PoseStamped()
        self.ttbot_pose.header = data.header
        self.ttbot_pose.pose = data.pose.pose
        #self.get_logger().info('ttbot_pose: {:.4f}, {:.4f}'.format(
        #        self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y))

    def a_star_path_planner(self, start_pose, end_pose):
        # A* path planner: converts world <-> map and runs A* search.
        path = Path()
        self.get_logger().info(
            f"A* planner.\n> start: {start_pose.pose.position},\n> end: {end_pose.pose.position}"
        )

        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        # --- Convert world → map ---
        start_i, start_j = self.world_to_map(start_pose)
        goal_i, goal_j = self.world_to_map(end_pose)

        self.get_logger().info(f"Start node (map): {start_i},{start_j}, Goal node (map): {goal_i},{goal_j}")

        # Ensure nodes exist in graph
        start_i, start_j = self.nearest_free_node(start_i, start_j)
        goal_i, goal_j = self.nearest_free_node(goal_i, goal_j)

        start_key = f"{start_i},{start_j}"
        goal_key = f"{goal_i},{goal_j}"

        self.get_logger().info(f"Start node (map): {start_key}, Goal node (map): {goal_key}")

        if start_key not in self.mp.map_graph.g or goal_key not in self.mp.map_graph.g:
            self.get_logger().warn(f"Start or goal not in graph! ({start_key}, {goal_key})")
            return path

        # --- Run A* ---
        astar_solver = AStarGrid(self.free_mask, diagonal=True)
        path = self.astar.solve((start_i, start_j), (goal_i, goal_j))

        try:
            node_path, _ = astar_solver.reconstruct_path(start_key, goal_key)
        except KeyError:
            self.get_logger().warn(f"Goal {goal_key} not reachable from start {start_key}")
            return path
        
        # --- Convert back map → world ---
        for node_name in node_path:
            i, j = map(int, node_name.split(","))
            pose = self.map_to_world(i, j)
            path.poses.append(pose)

        # After generating your path (e.g. nav_path)
        print("\nPath in world coordinates:")
        for i, pose_stamped in enumerate(path.poses):
            pos = pose_stamped.pose.position
            print(f"{i:>2}: (x={pos.x:.3f}, y={pos.y:.3f})")


        # --- Autograder timing ---
        self.astarTime = Float32()
        self.astarTime.data = float(self.get_clock().now().nanoseconds * 1e-9 - self.start_time)
        self.calc_time_pub.publish(self.astarTime)

        return path, node_path
    
    def nearest_free_node(self, i, j):
            print((f"{i},{j}"), " node in graph? ", (f"{i},{j}" in self.mp.map_graph.g))
            #Return the nearest (i,j) that exists in the graph
            if 0 <= i < self.free_mask.shape[0] and 0 <= j < self.free_mask.shape[1] and self.free_mask[i,j]:
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
    
    def world_to_map(self, pose):
        x, y = pose.pose.position.x, pose.pose.position.y
        resx = self.mp.map.map_df.resolution[0]
        resy = self.mp.map.map_df.resolution[0]        

        # Extract origin correctly
        origin_list = self.mp.map.map_df.origin.iloc[0]  # first element is the list [-0.29, -0.768, 0]
        ox, oy = origin_list[0], origin_list[1]
        #print("test: ", x, y)
        #fix
        i = self.mp.map.size[1] - int((y - oy) / resy)
        #j = self.mp.map.size[0] - int((x - ox) / resx)
        j = int((x - ox) / resx)

        #i = i - int(1/resx) + 1
        #j = self.mp.map.size[0] - int(1/0.05) - j - 1
        return i, j

    def map_to_world(self, i, j):
        res = self.mp.map.map_df.resolution[0]
        #res = 0.025

        origin_list = self.mp.map.map_df.origin.iloc[0]
        ox, oy = origin_list[0], origin_list[1]

        x = j * res + ox 
        #fix
        y = (self.mp.map.size[1]-i) * res + oy 
        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.position.y = y
        return pose
    
  
    def get_path_idx(self, path, vehicle_pose, last_idx=0):
        vx = vehicle_pose.pose.position.x
        vy = vehicle_pose.pose.position.y
        lookahead = 0.3
        for i in range(last_idx, len(path.poses)):
            px = path.poses[i].pose.position.x
            py = path.poses[i].pose.position.y
            dist = math.hypot(px - vx, py - vy)
            if dist > lookahead:
                return i
        return len(path.poses) - 1

    def path_follower_1(self, vehicle_pose, current_goal_pose):
        """Compute speed and heading to reach the current goal waypoint."""
        # Current position
        vx = vehicle_pose.pose.position.x
        vy = vehicle_pose.pose.position.y

        # Goal position
        gx = current_goal_pose.pose.position.x
        gy = current_goal_pose.pose.position.y

        # Heading error
        desired_yaw = math.atan2(gy - vy, gx - vx)

        # Robot current orientation (yaw)
        q = vehicle_pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))

        # Simple proportional control for heading
        heading_error = desired_yaw - yaw
        # Normalize between -pi and pi
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

        # Set speed and heading
        speed = 0.1  # m/s, slow forward speed
        heading = 2.0 * heading_error  # proportional gain Kp=2.0

        return speed, heading
    
    def quat_to_yaw(self, q):
        """Convert quaternion to yaw (safe for ROS standard axes)."""
        self.siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        self.cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
        return math.atan2(self.siny_cosp, self.cosy_cosp)

    def path_follower(self, vehicle_pose, current_goal_pose):
        vx = vehicle_pose.pose.position.x
        vy = vehicle_pose.pose.position.y
        gx = current_goal_pose.pose.position.x
        gy = current_goal_pose.pose.position.y

        #goal_dist = math.hypot(gx - vx, gy - vy)

        # desired yaw
        #desired_yaw = math.atan2(gy - vy, gx - vx)

        # current yaw from quaternion
        #q = vehicle_pose.pose.orientation
        #yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))

        # heading error (normalize)
        #heading_error = (desired_yaw - yaw + math.pi) % (2 * math.pi) - math.pi

        yaw = self.quat_to_yaw(vehicle_pose.pose.orientation)

        desired_yaw = math.atan2(gy - vy, gx - vx)

        # Compute heading error safely
        heading_error = (desired_yaw - yaw + math.pi) % (2 * math.pi) - math.pi

        # distance to goal
        dist = math.hypot(gx - vx, gy - vy)

        # PID-like control
        Kp_ang = 1.0   # proportional gain
        Kd_ang = 0.2   # derivative gain (optional, helps damping)
        Kp_lin = 0.5   # slow down when turning

        # heading control
        heading = Kp_ang * heading_error

        # reduce speed when heading error is large
        speed = max(0.05, min(0.2, Kp_lin * dist * math.cos(heading_error)))

        return speed, heading

    def move_ttbot(self, speed, heading):
        """! Function to move turtlebot passing directly a heading angle and the speed.
        @param  speed     Desired speed.
        @param  heading   Desired yaw angle.
        @return path      object containing the sequence of waypoints of the created path.
        """
        cmd_vel = Twist()
        # TODO: IMPLEMENT YOUR LOW-LEVEL CONTROLLER
        cmd_vel.linear.x = speed
        cmd_vel.angular.z = heading

        self.cmd_vel_pub.publish(cmd_vel)

    def move_ttbot_safe(self, speed, heading):

        """Move the TurtleBot using linear speed and angular velocity."""
        cmd_vel = Twist()
        # Limit speed to reasonable ranges
        cmd_vel.linear.x = max(min(speed, 0.2), 0.0)  # max 0.2 m/s
        cmd_vel.angular.z = max(min(heading, 1.0), -1.0)  # max ±1 rad/s
        self.cmd_vel_pub.publish(cmd_vel)
    
    def stop_robot(self):
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)

    def show_map_and_path(self, path):
        if path and len(path.poses) > 1:
            # Convert world coords to map pixels
            start_pose = path.poses[0].pose.position
            goal_pose = path.poses[-1].pose.position
            start = self.world_to_map(path.poses[0])
            goal = self.world_to_map(path.poses[len(path.poses)-1])
            #path2 = self.world_to_map(path)
            # Draw and display
            #path_image = self.mp.draw_path(path, lambda x, y: self.world_to_map(x, y))
            #path_image = self.mp.draw_path_wrld(path, lambda x, y: self.world_to_map(
            #    PoseStamped(pose=Pose(position=Point(x=y, y=x)))
            #    ))
            path_image = self.mp.draw_path_wrld(path, self.world_to_map)
            #print("Show size: ", self.mp.map.size)
            map_size = self.mp.map.size
            self.mp.display_map_with_path(path_image, map_size, start, goal)

    def run(self):
        """Main loop using get_path_idx() to follow waypoints intelligently."""
        self.get_logger().info("Navigation node started, waiting for AMCL and goal updates...")
        
        self.new_goal_received = False  # Flag for new goal
        self.ttbot_pose_received = False  # Flag for first AMCL pose

        while rclpy.ok():
            # 1️⃣ Process callbacks
            rclpy.spin_once(self, timeout_sec=0.05)

            # 2️⃣ Wait for a new goal
            if not self.new_goal_received:
                continue

            # 3️⃣ Plan path once
            path, map_path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
            path_arr = self.mp.draw_path(map_path)

            self.get_logger().info(f"Planned path with {len(path.poses)} waypoints")
            
            # 🔹 Show map + path before starting motion
            self.show_map_and_path(path)
            self.new_goal_received = False  # Reset goal flag

            # 4️⃣ Follow path using get_path_idx
            last_idx = 0

            while True:
                rclpy.spin_once(self, timeout_sec=0.05)

                idx = self.get_path_idx(path, self.ttbot_pose, last_idx)
                current_goal = path.poses[idx]

                if idx != last_idx:
                    print("Pose:", idx, "/", len(path.poses), 
                        "Current Goal: (", current_goal.pose.position.x, ",", current_goal.pose.position.y, ")")

                speed, heading = self.path_follower(self.ttbot_pose, current_goal)
                self.move_ttbot(speed, heading)

                last_idx = idx  # update

                if idx >= len(path.poses) - 1:
                    self.get_logger().info("Reached goal!")
                    self.stop_robot()
                    break

                if self.new_goal_received:
                    self.get_logger().info("New goal received! Replanning path...")
                    break


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