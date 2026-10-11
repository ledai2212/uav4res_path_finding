from UAV import UAV
from utils import Point, Vector
import random
from input import *

class Swarm:
    """
        Swarm class to represent a swarm of UAVs
        Attributes:
            uavs: List of UAV objects
            center: Point object representing the center of the swarm

    """
    def __init__(self, uavs, center):
        self.uavs = uavs
        self.center = center


    def set_center(self, point):
        """
            Set the center of the swarm
            Args:
                point: Point object representing the new center of the swarm
        """
        self.center = point
    
    def get_center_cell_position(self):
        """
            Returns:
                Tuple of cell position (x, y) of the center of the swarm
        """
        return (self.center.x // cell_size, self.center.y // cell_size)

    def move_a_frame(self):
        """
            Move the swarm a frame
        """
        for uav in self.uavs:
            uav.move_a_frame()

    def scan(self, map):
        for uav in self.uavs:
            uav.scan(map)

    def __repr__(self):
        return f"Swarm(posX={self.center.x}, posY={self.center.y}, uavs={self.uavs})"
