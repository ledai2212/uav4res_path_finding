import time

class Measurer:
    def __init__(self, type, num_of_uavs, map):
        self.num_of_uavs = num_of_uavs
        self.map = map
        self.type = type
        self.time = 0
        self.cost = 0
        self.initial_time = 0
        # data: [{time: amount of data}]
        self.recent_data = 0
        self.data = [{0: 0}] 
    
    def tick_time(self):
        self.time += 1/60

    def add_cost(self, priority):
        self.cost += self.time * priority

    def get_data(self, data):
        self.recent_data += data
        self.data.append({self.time: self.recent_data}) 

    def print(self):
        f = open("output.txt", "a")
        f.write(f"type = {self.type}, num_of_uavs = {self.num_of_uavs}, time = {self.time}, cost = {self.cost}, data = \n {self.data}\n")
        f.close()