import os 
import sys
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pygame
import sys
from input import *
from Map import Map, is_point_in_polygon
from UAV import UAV
from Swarm import Swarm
import random
from Drawer import Drawer
from utils import *
import time
from Measurer import Measurer


measurer = Measurer(type = "tsunami", num_of_uavs = num_of_uavs, map = maptype)
# Bước 1: Khởi tạo các thực thể, biến đếm
drawer = Drawer("tsunami")                       # Khởi tạo đối tượng Drawer
uavs = []                               # Khởi tạo danh sách các UAVs
for i in range(num_of_uavs):
    uavs.append(UAV(uav_distance.real, 0, time_charge,  min_speed[i], max_speed[i], None, Point(*uav_start), "./images/uav.png"))
swarm = Swarm(uavs, Point(605, 445))   # Khởi tạo đội Swarm
map0 = Map(state, priority) # Khởi tạo đối tượng Map
# wavefront_map = wavefront((uav_end[0] // cell_size, uav_end[1] // cell_size), map0)
# wavefront_map = None
uav_index = 0                          # Chỉ số của UAV hiện tại (Dùng để chọn UAV trong đội)


# Bước 2: Các bước tiền tính toán

clusters = calculate_centroid_priority(map0)  # Tính toán ưu tiên của các vùng cần quét
clusters_centers = []
for cluster in clusters:
    clusters_centers.append((int(cluster.center[0]), int(cluster.center[1])))

# Bước 3: Vòng lặp chính

running = True
current_cluster_index = 0
reached_center = False
start_time = time.time()
swarm = Swarm(uavs, clusters[0].center)
# while running and current_cluster_index < len(clusters):
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
            running = False

    # Cập nhật tâm swarm cho khu vực hiện tại
    current_cluster_center = Point(clusters[current_cluster_index].center[0], clusters[current_cluster_index].center[1])
    swarm.set_center(current_cluster_center)
    current_cluster_end_cell = Point(int(clusters[current_cluster_index].end_of_cluster[0] // cell_size), 
                                        int(clusters[current_cluster_index].end_of_cluster[1] // cell_size))

    # Các UAV quét khu vực hiện tại
    if Map.is_cluster_scanned(map0.state, current_cluster_center, cell_radius):
        print(f"Region {current_cluster_index} scanned completely. Moving to next region.")
        if current_cluster_index <= len(clusters) - 1:
            current_cluster_index += 1
        if current_cluster_index >= len(clusters):
            print("All regions scanned. Mission complete!")
            measurer.print()
            break
        
        
    
    # Phân công UAV quét trong khu vực
    for uav in swarm.uavs:
        if uav.is_blocked == 1:
            uav.time_charge -= 1
            if  uav.time_charge == 0:
                uav.is_blocked = 0
                # engergy_left[u] = 10
                uav.distance= uav_distance
                uav.time_charge = time_charge
        if (uav.status == UAV.UAVState.FREE or (uav.status == UAV.UAVState.BUSY and uav.recent_path is None)) and not uav.is_blocked:
            uav_cell_position = uav.get_cell_position()
            measurer.add_cost(map0.priority[uav_cell_position[0]][uav_cell_position[1]])
            '''Lựa chọn cho các UAV tìm kiếm ô tiếp theo để quét dựa trên vị trí của cluster center hiện tại'''
            # cluster_map = create_cluster_map(map0, clusters[current_cluster_index].available_cells)
            wavefront_map = wavefront((map_width - 1, map_height - 1), map0)
            next_cell, shortest_path, path_to_charge = select_target_cell(wavefront_map, uav_cell_position, map0)
            '''Lựa chọn cho các UAV tìm kiếm ô tiếp theo để quét dựa trên vị trí hiện tại của UAV'''
            # wavefront_map = wavefront((uav_cell_position[0], uav_cell_position[1]), map0)
            # next_cell, shortest_path = select_target_cell(wavefront_map, Point(uav_cell_position[0], uav_cell_position[1]), map0)
            #print('path:', shortest_path)
            
            if next_cell is None: 
                print(f"UAV at {uav_cell_position}: No reachable cell in region {current_cluster_index}")
                uav.recent_path = None
                uav.target_position = None
                uav.status = UAV.UAVState.FREE
            else:
                uav.recent_path = shortest_path
                uav.index_path = 0
                uav.status = UAV.UAVState.BUSY
                map0.state[next_cell[0]][next_cell[1]] = Map.CellState.SCANNING
                uav.target_position = Point(next_cell[0] * cell_size + cell_size // 2, 
                                            next_cell[1] * cell_size + cell_size // 2)
                dis = cal_distance_path(uav.recent_path)
                uav.distance -= dis
                #print(f"UAV moving to {next_cell} in cluster {current_cluster_index}")
                if uav.distance.real < dis_threshold:
                        map0.state[next_cell[0]][next_cell[1]] = Map.CellState.NOT_SCANNED
                        uav.is_blocked = 1
                        uav.index_path = 0
                        uav.recent_path = path_to_charge
                        uav.target_position = Point(10*cell_size + cell_size//2, 10*cell_size + cell_size//2)
    measurer.tick_time()
    swarm.move_a_frame()
    swarm.scan(map0)
    
    # Cập nhật trạng thái UAV sau khi di chuyển
    for uav in swarm.uavs:
        if uav.status == UAV.UAVState.BUSY and uav.recent_path and uav.index_path >= len(uav.recent_path):
            uav.recent_path = None
            uav.target_position = None
            uav.status = UAV.UAVState.FREE

    drawer.draw_all(map0, swarm, clusters_centers, wavefront_map)
    drawer.clock.tick(FPS)

# Kết thúc
drawer.kill_window()