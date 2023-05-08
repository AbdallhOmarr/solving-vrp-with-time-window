from vrptw_base import *
from genetic_algorithm import *
import numpy as np
from vprtw_aco_figure import *


if __name__ == "__main__":
    file_path = "./solomon-100/r101.txt"
    ants_num = 10
    beta = 2
    q0 = 0.1
    show_figure = True

    graph = VrptwGraph(file_path)
    ga = GeneticAlgorithm(graph)
    ga.run(30, 15)

    input("click done to complete")
## this is the working algorithm
