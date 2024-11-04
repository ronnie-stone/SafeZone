import pygad
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point, Polygon, MultiPolygon
import time 
import os 
import csv

from expand_adjacency_matrix import expand_adjacency_matrix
from tessellate_with_buffer import tessellate_with_buffer
from adjacency_matrix_to_connected_tasks import adjacency_matrix_to_connected_tasks
from adjacency_matrix_from_regions import adjacency_matrix_from_regions
from chromatic_number import chromatic_number

from plot_custom_fitness import plot_custom_fitness
from plot_custom_solution import plot_custom_solution

from get_A_regions import get_A_regions
from integer_lp import task_scheduling_ilp
from create_polygon import create_polygon
from matplotlib import cm

# General Parameters

input_polygon = [(0,0), (3,0), (3,3), (0,3), (0,0)] # Square
# input_polygon = [(0,0), (6,0), (6,3), (0,3), (0,0)] # Rectangle
# nput_polygon = [(0,0), (3,0), (1.5,3), (0,0)] # Triangle
# input_polygon = [(0,0), (3,0), (3, 0.5), (2.5, 0.5), (2.5, 2.5), (3, 2.5), (3,3), (0,3), (0, 2.5), (0.5, 2.5), (0.5,0.5), (0,0.5), (0,0)] # I-BEAM
# input_polygon = [(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)] # Z-BEAM
# input_polygon = np.load('bunny_cross_section_scaled.npy')

# input_polygon = [
#     [(0,0), (3,0), (3,3), (0,3), (0,0)],
#     [(0.5, 1), (1, 1), (1,2), (0.5, 2), (0.5, 1)],
#     [(2, 1), (2.5, 1), (2.5, 2), (2, 2), (2, 1)]
#     ]

# Circle:
# radius = 1
# center = np.array([1.5, 1.5])
# num_points = 100
# angles = np.linspace(0, 2 * np.pi, num_points)
# x_points = radius * np.cos(angles) + center[0]
# y_points = radius * np.sin(angles) + center[1]
# input_polygon = np.column_stack((x_points, y_points))

minimum_ridge_length = 0.2
minimum_robot_distance = 0.5
reachability_radius = 2


def generate_gene_space(N, low=-1, high=4):
    # Each point has two coordinates (x, y), so you need 2 * N genes
    return [{'low': low, 'high': high} for _ in range(2 * N)]

def points_in_polygon(points_to_check, polygon_points):
    """
    Return True if any point in points_to_check is inside the polygon defined by polygon_points.
    Return False if all points are outside.

    :param polygon_points: List of tuples representing the vertices of the polygon [(x1, y1), (x2, y2), ..., (xn, yn)].
    :param points_to_check: List of lists representing the points to check [[x1, y1], [x2, y2], ..., [xm, ym]].
    :return: True if any point is inside the polygon, False if all points are outside.
    """
    # Create a Polygon object from the given polygon points
    polygon = create_polygon(polygon_points)

    # Check each point and return True if any point is inside the polygon
    for p in points_to_check:
        if polygon.contains(Point(p)):
            return True

    # If no points are inside, return False
    return False

def points_too_close(points, min_distance):
    """
    Checks if any points in the array are closer to each other than the given minimum distance.

    Parameters:
    points (np.ndarray): A 2D array of points where each row represents a point [x, y].
    min_distance (float): The minimum allowable distance between any two points.

    Returns:
    bool: True if any points are closer than the minimum distance, False otherwise.
    """
    num_points = points.shape[0]
    
    # Iterate over all unique pairs of points
    for i in range(num_points):
        for j in range(i + 1, num_points):
            distance = np.linalg.norm(points[i] - points[j])
            if distance <= min_distance:
                return True
    return False

def is_reachable(points, polygons_A, reachability_radius):
    """
    Checks if all polygons are fully contained within circles of given radius centered at their associated points.

    Parameters:
    polygons (list[Polygon]): A list of Shapely Polygon objects.
    points (np.ndarray): A 2D array of points where each row represents a point [x, y].
    min_distance (float): The radius of the circle to check for containment.

    Returns:
    bool: True if all polygons are fully contained within their respective circles, False otherwise.
    """
    if len(polygons_A) != len(points):
        raise ValueError("The number of polygons and points must be the same.")
    
    # Check each polygon for containment within the circle defined by its associated point
    for polygon, point in zip(polygons_A, points):
        circle = Point(point).buffer(reachability_radius)  # Create a circular buffer with radius `min_distance`
        if not polygon.within(circle):  # If any polygon is not contained, return False
            return False
    
    # If all polygons are contained, return True
    return True

# Define the fitness function
def fitness_func(ga, solution, solution_idx):

    # Get Voronoi points from solution:
    points = np.array([[solution[i], solution[i+1]] for i in range(0, len(solution), 2)])

    # Check if points are in inside of polygon:
    if points_in_polygon(points, input_polygon):
        return (-10)
    
    # Check if points are too close to each other:
    if points_too_close(points, minimum_robot_distance):
        return (-10)

    # Tessellate to get all polygons and areas:
    try:
        _, _, polygons_A, polygons_A_star_areas, polygons_B_areas, _ = tessellate_with_buffer(points, input_polygon, minimum_ridge_length)
    except:
        print("Error in tessellation")
        return -10
    
    # Check if reachability is satisfied:
    if not is_reachable(points, polygons_A, reachability_radius):
        return -10
    
    # Create parameters for mixed-integer solver:
    adjacency_matrix = adjacency_matrix_from_regions(polygons_A, minimum_ridge_length)
    expanded_adjacency_matrix = expand_adjacency_matrix(adjacency_matrix)
    task_list = np.arange(0, 2*len(points))
    task_duration = np.array(polygons_A_star_areas + polygons_B_areas)
    robots = np.tile(np.arange(1, len(points)+1), 2)
    connected_tasks = adjacency_matrix_to_connected_tasks(expanded_adjacency_matrix)

    # Alternatively, create parameters for heuristic driven, solution:
    # adjacency_matrix = adjacency_matrix_from_regions(polygons_A, minimum_ridge_length)
    # task_list = np.arange(0, len(points))
    # task_duration = np.array(polygons_B_areas)
    # robots = np.tile(np.arange(1, len(points)+1), 1)
    # connected_tasks = adjacency_matrix_to_connected_tasks(adjacency_matrix)

    # Find optimal schedule:
    try:
        makespan = task_scheduling_ilp(task_list, task_duration, robots, connected_tasks)

        # If using heuristic-driven solution:

        # makespan += np.max(polygons_A_star_areas)
    except:
        print("Error in graph scheduling")
        return -10

    return -makespan

# Callback function on generartion.
def on_generation(ga_instance, best_solutions_list, ax):
    print(ga_instance.generations_completed) # Counter.
    best_solution_generation = ga_instance.best_solution()[0]
    best_solutions_list.append(best_solution_generation)
    
if __name__ == "__main__":

    # Define the PyGAD parameters
    N = 4
    num_generations = 19  # Number of generations
    num_parents_mating = 20  # Number of solutions to mate
    sol_per_pop = 100  # Population size
    num_genes = N * 2  # Each point has 2 coordinates (x, y)
    gene_space = generate_gene_space(N, low=-1, high=4)
    fig, ax = plt.subplots()

    best_solutions = []

    # Create the PyGAD instance
    ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_func,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        mutation_percent_genes=10,
                        on_generation=lambda instance: on_generation(instance, best_solutions, ax),
                        gene_space=gene_space,
                        suppress_warnings=True)

    # Run the GA
    t1 = time.time()
    ga_instance.run()
    t2 = time.time()
    print("Time taken for GA: " + str(t2-t1))

    # Get the best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Best solution: {solution}")
    print(f"Fitness of the best solution: {solution_fitness}")

    # Run routine to get optimal schedule:

    points = np.array([[solution[i], solution[i+1]] for i in range(0, len(solution), 2)])

    # Tessellate to get all polygons and areas:

    polygons_A_star, polygons_B, polygons_A, polygons_A_star_areas, polygons_B_areas, polygons_A_areas = tessellate_with_buffer(points, input_polygon, minimum_ridge_length)

    # Get adjacency matrix, and expanded adjacency matrix. 

    heuristic = False
    if heuristic == True:
        adjacency_matrix = adjacency_matrix_from_regions(polygons_A, minimum_ridge_length)
        task_list = np.arange(0, len(points))
        task_duration = np.array(polygons_B_areas)
        robots = np.tile(np.arange(1, len(points)+1), 1)
        connected_tasks = adjacency_matrix_to_connected_tasks(adjacency_matrix)
        makespan = task_scheduling_ilp(task_list, task_duration, robots, connected_tasks, verbose=True) + np.max(polygons_A_star_areas)

    else:
        adjacency_matrix = adjacency_matrix_from_regions(polygons_A, minimum_ridge_length)
        expanded_adjacency_matrix = expand_adjacency_matrix(adjacency_matrix)
        # Create parameters for mixed-integer solver:
        task_list = np.arange(0, 2*len(points))
        task_duration = np.array(polygons_A_star_areas + polygons_B_areas)
        robots = np.tile(np.arange(1, len(points)+1), 2)
        connected_tasks = adjacency_matrix_to_connected_tasks(expanded_adjacency_matrix)
        makespan = task_scheduling_ilp(task_list, task_duration, robots, connected_tasks, verbose=True)
    
    print("Adjacency Matrix of Final Solution:") 
    print(adjacency_matrix)
    print("Expanded Adjacency Matrix of Final Solution:") 
    print(expanded_adjacency_matrix)
    print("Chromatic Number of Final Solution:")
    chi = chromatic_number(adjacency_matrix)
    print(chi)
    print("Connected Tasks of Final Solution:")
    print(connected_tasks)

    # Get theoretical best:

    polygon = create_polygon(input_polygon)
    print_area = polygon.area
    best_makespan = -print_area/float(N)

    # Plot final tessellation and fitness history:

    foldername = "square4"
    results = np.array([chi, solution_fitness, best_makespan, solution])

    if foldername is None:
        foldername = os.getcwd()  # Get the current working directory

    # Ensure the folder exists, if not, create it
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    filepath = os.path.join(foldername, "solution_parameters")
    np.save(filepath, results)
        
    plot_custom_solution(solution, polygons_A_star, polygons_B, minimum_robot_distance, reachability_radius, ax, foldername=foldername)
    plot_custom_fitness(ga_instance, best_solutions, best_makespan, minimum_ridge_length, input_polygon, foldername=foldername)
    #plt.show()