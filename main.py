import numpy as np
from expand_adjacency_matrix import expand_adjacency_matrix
from tessellate_with_buffer import tessellate_with_buffer
from adjacency_matrix_to_connected_tasks import adjacency_matrix_to_connected_tasks
from adjacency_matrix_from_regions import adjacency_matrix_from_regions
from get_A_regions import get_A_regions
from integer_lp import task_scheduling_ilp


points = np.array([[0, 0], [1, 0], [0.9, 0.9], [0, 1]])
# input_polygon = np.array([(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75), (0.25,0.25)])
input_polygon = [
        [(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)],  # Outer boundary
        [(1, 1), (1.5, 1), (1.5, 2), (1, 2), (1, 1)]  # Hole
    ]
minimum_distance = 0.15

# Tessellate to get all polygons and areas:

polygons_A_star, polygons_B, polygons_A_star_areas, polygons_B_areas = tessellate_with_buffer(points, input_polygon, minimum_distance)
polygons_A, polygons_A_areas, vor_A = get_A_regions(points, input_polygon)

# Get adjacency matrix, and expanded adjacency matrix. 

adjacency_matrix = adjacency_matrix_from_regions(polygons_A, minimum_distance)
expanded_adjacency_matrix = expand_adjacency_matrix(adjacency_matrix)

# print("Expanded Adjacency Matrix: ")
# print(expanded_adjacency_matrix)

# print("Sanity Check for Areas: ")
# for i in range(len(polygons_A)):
#     print(polygons_A_star_areas[i] + polygons_B_areas[i], polygons_A_areas[i])

# Create parameters for mixed-integer solver:

task_list = np.arange(0, 2*len(points))
task_duration = np.array(polygons_A_star_areas + polygons_B_areas)
robots = np.tile(np.arange(1, len(points)+1), 2)
connected_tasks = adjacency_matrix_to_connected_tasks(expanded_adjacency_matrix)
makespan = task_scheduling_ilp(task_list, task_duration, robots, connected_tasks, verbose=True)

