from enum import Enum
import math
import random
from input import *

random.seed(1)


class Point:
    """
        Point class to represent a point in the map
        Attributes:
            x: x-coordinate of the point in pixel
            y: y-coordinate of the point in pixel
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Vector:
    """
        Vector class to represent a vector in the map
        Attributes:
            x: x-coordinate of the vector in pixel
            y: y-coordinate of the vector in pixel
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def normalize(self):
        """
            Normalize the vector
            Returns:
                Vector object representing the normalized vector
        """
        magnitude = (self.x ** 2 + self.y ** 2) ** 0.5
        if magnitude == 0:
            return Vector(0, 0)
        return Vector(self.x / magnitude, self.y / magnitude)


def is_point_in_polygon(x, y, polygon):
    """Check if a point (x, y) is inside a polygon."""
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

class Map:
    """
        Map class to represent the map of the area of interest
        Attributes:
            aoi: Area of Interest, List of cell positition
            wind: Wind direction, Vector object
            num_of_obstacles: Number of obstacles to be rendered
            priority: 2D array to store the priority of each cell, dim: (map_width, map_height)
            state: 2D array to store the state of each cell, dim: (map_width, map_height)
            CellState: Enum to represent the state of a cell
    """
    class CellState:
        NOT_SCANNED = 1
        SCANNING = 3
        SCANNED = 2
        UNREACHABLE = -1
        NO_INTEREST = 0
    class DataState:
        NO_DATA = 0
        HAS_DATA = 1

    def __init__(self, state, priority):
        self.priority = priority
        self.state = state
        self.cluster_cells = []

    def top_left_corner_of_the_cell(self, x, y):
        """
            Args:
                x: x-coordinate of the cell in the map
                y: y-coordinate of the cell in the map
            Returns:
                Point object representing the top left corner of the cell
        """
        return Point(x * cell_size, y * cell_size)

    def get_cell_position(self, point):
        """
            Args:
                point: Point object
            Returns:
                Tuple of cell position (x, y) of the given point
        """
        return (point.x // cell_size, point.y // cell_size) 
                    
    @classmethod
    def is_cluster_scanned(cls, map_state, cluster_center, cell_radius):
        """
        Check if the region within radius around cluster_center is fully scanned.
        """
        rows = len(map_state)
        cols = len(map_state[0])
        center_coor_x = cluster_center.x
        center_coor_y = cluster_center.y
        radius = cell_radius * cell_size
        half_cell = cell_size // 2
        cls.cluster_cells = []

        # Only cell centers inside this bounding box can be in the circle.
        # The exact squared-distance check below preserves the old boundary
        # semantics while avoiding two full rows x cols scans every step.
        min_x = max(0, math.ceil((center_coor_x - radius - half_cell) / cell_size))
        max_x = min(rows - 1, math.floor((center_coor_x + radius - half_cell) / cell_size))
        min_y = max(0, math.ceil((center_coor_y - radius - half_cell) / cell_size))
        max_y = min(cols - 1, math.floor((center_coor_y + radius - half_cell) / cell_size))
        radius_squared = radius ** 2
        is_scanned = True

        for x in range(min_x, max_x + 1):
            cell_coor_x = x * cell_size + half_cell
            for y in range(min_y, max_y + 1):
                cell_coor_y = y * cell_size + half_cell
                if ((cell_coor_x - center_coor_x) ** 2 +
                        (cell_coor_y - center_coor_y) ** 2 <= radius_squared):
                    cls.cluster_cells.append((x, y))
                    if map_state[x][y] == cls.CellState.NOT_SCANNED:
                        is_scanned = False

        return is_scanned
