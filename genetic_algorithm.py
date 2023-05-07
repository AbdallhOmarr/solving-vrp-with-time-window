import numpy as np
import random
from vprtw_aco_figure import VrptwAcoFigure
from vrptw_base import VrptwGraph, PathMessage
from ant import Ant
from threading import Thread, Event
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import copy
import time
from multiprocessing import Process
from multiprocessing import Queue as MPQueue

class Solution:
    def __init__(self,travel_path,travel_distance,vehicle_num):
        self.travel_path = travel_path
        self.travel_distance = travel_distance
        self.vehicle_num = vehicle_num
        
class GeneticAlgorithm:
        def __init__(self, graph: VrptwGraph,pop_size=10, whether_or_not_to_show_figure=True):
            super()

            self.graph = graph
            self.pop_size = pop_size

            # best path
            self.best_path_distance = None
            self.best_path = None
            self.best_vehicle_num = None

            self.whether_or_not_to_show_figure = whether_or_not_to_show_figure

        def get_nnh_solution(self):
            self.initial_solution = Solution(self.graph.nearest_neighbor_heuristic()[0],self.graph.nearest_neighbor_heuristic()[1],self.graph.nearest_neighbor_heuristic()[2])
            return self.initial_solution
        
        
        def fitness_fn(self,solution):
            return 1/solution.travel_distance
            
        def run_algorithm(self):
            #function to generate initial solution
            self.get_init_population()
            
            #define fitness function
            