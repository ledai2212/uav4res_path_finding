from utils import Vector, get_sign
from enum import Enum
from Map import Map
import random
from input import *

class UAV:
    """
        UAV class to represent a UAV
        Attributes:
            distance: Remaining energy of the UAV
            status: Status of the UAV (FREE or BUSY)
            min_speed: Minimum speed of the UAV
            max_speed: Maximum speed of the UAV
            buffer_data: Buffer data of the UAV
            recent_position: Recent position of the UAV in pixel
            target_position: Target position of the UAV in pixel
            direction: Direction of the UAV (Vector object which was normalized)
            recent_path: Recent path of the UAV
            image_path: Path to the image of the UAV
            max_distance: Maximum energy capacity of the UAV. Used together with
                `distance` (remaining energy) by Drawer to compute the fraction
                drawn for the energy health bar. None means energy tracking is
                disabled for this UAV, in which case Drawer skips the bar.
            at_base: Whether the UAV is currently parked at the charging
                station. Lets Drawer tell "returning to base" (is_blocked=True,
                at_base=False) apart from "charging" (is_blocked=True,
                at_base=True).
            charging_progress: Normalized render-only charging progress. None
                lets renderers fall back to their legacy charging display.
    """
    class UAVState(Enum):
        FREE = 0
        BUSY = 1

    def __init__(self, distance, is_blocked, time_charge, min_speed, max_speed, buffer_data, recent_position, image_path=None, recent_path=None, target_position=None, max_distance=None):
        self.recent_position = recent_position
        self.recent_path = recent_path
        self.target_position = target_position
        self.status = self.UAVState.FREE
        self.distance = distance
        self.max_distance = max_distance
        self.at_base = False
        self.charging_progress = None
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.direction = Vector(0, 0)
        self.buffer_data = buffer_data
        self.image_path = image_path
        self.is_blocked = is_blocked
        self.time_charge = time_charge
        self.index_path = 0
        self.data = [[Map.DataState.NO_DATA for j in range(map_height)] for i in range(map_width)]

    def set_direction(self, vector):
        """
            Set the direction of the UAV
            Args:
                vector: Vector object representing the direction of the UAV
        """
        self.direction = vector.normalize()

    def get_cell_position(self):
        """
            Returns:
                Tuple of cell position (x, y) of the UAV
        """
        x = int(self.recent_position.x // cell_size)
        y = int(self.recent_position.y // cell_size)
        return (x, y)
    
    def move_a_frame(self):
        """
            Move the UAV a frame
        """
        # self.recent_position.x += self.direction.x * random.uniform(self.min_speed, self.max_speed) / FPS
        # self.recent_position.y += self.direction.y * random.uniform(self.min_speed, self.max_speed) / FPS
        speed = random.uniform(self.min_speed, self.max_speed)
        if self.recent_path != None:
            if self.index_path < len(self.recent_path):
                target_x, target_y = (self.recent_path[self.index_path][0] * cell_size + cell_size // 2, self.recent_path[self.index_path][1] * cell_size + cell_size // 2) 
                dx, dy = target_x - self.recent_position.x, target_y - self.recent_position.y
                self.set_direction(Vector(dx, dy))
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist == 0:  # Nếu UAV gần điểm đích, chuyển sang điểm tiếp theo
                    if self.index_path == len(self.recent_path) - 1:
                        self.status = self.UAVState.FREE
                        self.direction = (0, 0)
                        self.recent_path = None
                    self.index_path += 1
                else:
                    self.recent_position.x += min(abs(speed * self.direction.x / FPS), abs(dx)) * get_sign(self.direction.x)
                    self.recent_position.y += min(abs(speed * self.direction.y / FPS), abs(dy)) * get_sign(self.direction.y)

    def scan(self, map):
        """
            Scan the map, we assume that scanning is intermediately done by the UAV
            Args:
                map: Map object
        """
        x, y = float(self.recent_position.x / cell_size), float(self.recent_position.y / cell_size)
        # print(f"recent_position = {self.recent_position.x}, {self.recent_position.y}")
        # print(f"x, y = {x}, {y}")
        # print(f"int(x), int(y) = {int(x)}, {int(y)}")


        if (map.state[int(x)][int(y)] == Map.CellState.NOT_SCANNED or map.state[int(x)][int(y)] == Map.CellState.SCANNING) and (abs(x - int(x) - 0.5) < 0.01 and abs(y - int(y) - 0.5) < 0.01):
            map.state[int(x)][int(y)] = Map.CellState.SCANNED
            self.data[int(x)][int(y)] = Map.DataState.HAS_DATA
    def transmit_data(self):
        if self.buffer_data > 0:
            print("Transmitting data...")
            self.buffer_data -= 1  # Assuming transmitting data consumes 1 unit of buffer data
        else:
            print("No data to transmit.")
