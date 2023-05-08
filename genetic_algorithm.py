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
    def __init__(self,graph,travel_path,travel_distance,vehicle_num):
        self.travel_path = np.array(travel_path)
        self.travel_distance = travel_distance
        self.vehicle_num = vehicle_num
        self.graph = graph
        self.dist_matrix = self.graph.node_dist_mat
        
    def calculate_path_distance(self):
        total_distance = 0 
        for i,customer in enumerate(self.travel_path):
            if i>=len(self.travel_path)-1:
                break
            distance = self.dist_matrix[customer][self.travel_path[i+1]]
            total_distance+=distance
        return total_distance

    def get_total_fitness(self):
        self.travel_distance = self.calculate_path_distance()
        self.fitness_value = 1/self.travel_distance*1000
        return self.fitness_value
        
    def check_capacity_constrain(self):
        capacity_constrain = True
        vehicle_capacity = self.graph.vehicle_capacity
        vehicle_load = 0 
        for customer in self.travel_path:
            if customer == 0 : 
                vehicle_load = 0 
            else:
                vehicle_load += self.graph.nodes[customer].demand
                
            if vehicle_load>vehicle_capacity:
                capacity_constrain= False
            
        return capacity_constrain
    def check_time_constrain(self):
            time_constrain = True
            current_time = 0 
            for customer in self.travel_path:
                if customer == 0 : 
                    current_time = 0 

                wait_time = max(self.graph.nodes[customer].ready_time - current_time, 0)
                service_time = self.graph.nodes[customer].service_time
                
                #Check whether it is possible to return to the service station after visiting a customer.
                if current_time  + wait_time + service_time> self.graph.nodes[0].due_time:
                    time_constrain = False

                # Do not serve customers beyond their due time.
                if current_time  > self.graph.nodes[customer].due_time:
                    time_constrain = False

            return time_constrain
        
    
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
            self.dist_matrix = self.graph.node_dist_mat

        def get_nnh_solution(self):
            self.initial_solution = Solution(self.graph,self.graph.nearest_neighbor_heuristic()[0],self.graph.nearest_neighbor_heuristic()[1],self.graph.nearest_neighbor_heuristic()[2])
            return self.initial_solution
        def get_customers_locations(self,solution):
            customers_locations=[]
            for i in range(0,len(solution.travel_path)):
                
                if solution.travel_path[i]!=0:
                    customers_locations.append(i)
            return customers_locations
        
        
        def apply_2_opt(self,solution,iterations,tabu_limit):
            tabu_list =[]
            customers_locations = self.get_customers_locations(solution)
            population = [solution]
            for i in range(0,iterations):
                print(f"iteration:{i}")
                customers = [c for c in range(1,len(customers_locations))]
                print(f"customers:{customers}")
                num_customers = len(customers)
                if num_customers>0:
                    c1_val = random.choice(customers)
                    c1_idx = np.where(solution.travel_path == c1_val)
                    customers.remove(c1_val)
                    tabu_list.append(c1_val)
                else:
                    continue
                
                if num_customers>0:
                    c2_val = random.choice(customers)
                    c2_idx = np.where(solution.travel_path == c2_val)
                    customers.remove(c2_val)
                    tabu_list.append(c2_val)
                    
                else:
                    continue

                if i%10==0 or len(tabu_list)>=tabu_limit:
                    customers = customers + tabu_list
                    tabu_list = []
                
                
                
                solution_path = solution.travel_path
                # c1_val = solution_path[c1]
                # c2_val = solution_path[c2]
                
                solution_path[np.where(solution.travel_path == c1_val)] = c2_val
                solution_path[np.where(solution.travel_path == c2_val)] = c1_val
                
                solution_dist = self.get_total_solution_distace(solution_path) 
                num_of_vehicles = self.get_total_num_vehicles(solution_path)
                
                new_solution = Solution(self.graph,solution_path, solution_dist, num_of_vehicles)
                solution_fitness = new_solution.get_total_fitness()
                if new_solution.check_capacity_constrain() and new_solution.check_time_constrain() :
                    if solution_fitness>solution.get_total_fitness() :
                        population.append(new_solution)
                    else:
                        p = random.random()
                        if p>0.5:
                            population.append(new_solution)
                        else:
                            continue
                
            population_fitnesses = [solution.get_total_fitness() for solution in population]
            population_best_fitness = max(population_fitnesses)
            index = population_fitnesses.index(population_best_fitness)
            best_sol = population[index]

            return best_sol
        
        def fitness_value(self,solution):
            return solution.get_total_fitness()
            
            
        def get_total_solution_distace(self,travel_path):
            total_distance = 0 
            for i,customer in enumerate(travel_path):
                if i>=len(travel_path)-1:
                    break
                distance = self.dist_matrix[customer][travel_path[i+1]]
                total_distance+=distance
            return total_distance
                        
        def get_total_num_vehicles(self,travel_path):
            num_vehicles = sum(1 for x in travel_path if x == 0) - 2
            return num_vehicles
        
        def generate_population(self,gen_size,pop_size):
            #generate many nnh solution and apply local search to generate random solutions?
            #generate many nnh solution then add changes to each path randomly 
            #i will apply local search first 
            initial_solution = self.get_nnh_solution()
            population = [] 
        
            for i in range(0,gen_size):
                if len(population)>= pop_size:
                    break
                population.append(self.apply_2_opt(initial_solution, 100, 10))

            return population
            
        def run_algorithm(self):
            #function to generate initial solution
            self.get_init_population()
            
            #define fitness function
            