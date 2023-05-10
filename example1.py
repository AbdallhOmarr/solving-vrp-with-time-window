from vrptw_base import VrptwGraph
from multiple_ant_colony_system import MultipleAntColonySystem


if __name__ == "__main__":
    file_path = "./solomon-100/r101.txt"
    write_path = "./solution text.txt"
    ants_num = 10
    beta = 1
    q0 = 0.1
    show_figure = True

    graph = VrptwGraph(file_path)

    macs = MultipleAntColonySystem(
        graph,
        ants_num=ants_num,
        beta=beta,
        q0=q0,
        whether_or_not_to_show_figure=show_figure,
    )
    macs.run_multiple_ant_colony_system(write_path)


## this is the working algorithm
