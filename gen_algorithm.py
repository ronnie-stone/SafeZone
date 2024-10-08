import pygad
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point, Polygon, MultiPolygon
import time 

from expand_adjacency_matrix import expand_adjacency_matrix
from tessellate_with_buffer import tessellate_with_buffer
from adjacency_matrix_to_connected_tasks import adjacency_matrix_to_connected_tasks
from adjacency_matrix_from_regions import adjacency_matrix_from_regions
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

minimum_distance = 0.2

def generate_gene_space(N, low=-1, high=4):
    # Each point has two coordinates (x, y), so you need 2 * N genes
    return [{'low': low, 'high': high} for _ in range(2 * N)]

def plot_custom_fitness(ga_instance, best_solutions, theoretical_best_fitness):
    # Extract the fitness values from the genetic algorithm instance
    fitness_values = ga_instance.best_solutions_fitness
    
    # Create the plot
    plt.figure()
    plt.plot(fitness_values, label="Best Fitness", color='blue')
    
    # Add a horizontal line for the theoretical best fitness
    plt.axhline(y=theoretical_best_fitness, color='red', linestyle='--', label='Theoretical Best Fitness')
    
    # Add labels, title, and legend
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Over Generations")
    plt.legend()
    plt.savefig("Fig2")

    areas_A = []
    std_areas_A = []
    abs_diff_areas_A = []
    max_diff_areas_A = []
    mean_squared_error_areas_A = []
    sum_of_areas_A = []

    areas_B = []
    std_areas_B = []
    abs_diff_areas_B = []
    max_diff_areas_B = []
    mean_squared_error_areas_B = []
    sum_of_areas_B = []


    for i in range(len(best_solutions)):

        solution = best_solutions[i]
        points = np.array([[solution[i], solution[i+1]] for i in range(0, len(solution), 2)])

        try:
            polygons_A_star, polygons_B, polygons_A, polygons_A_star_areas, polygons_B_areas, polygons_A_areas = tessellate_with_buffer(points, input_polygon, minimum_distance)
            array_A = np.array(polygons_A_star_areas)
            array_B = np.array(polygons_B_areas)
            areas_A.append(array_A)
            areas_B.append(array_B)

            std_areas_A.append(np.std(array_A))
            abs_diff_areas_A.append(np.sum(np.abs(np.diff(array_A))))
            mean_squared_error_areas_A.append(np.mean(np.diff(array_A) ** 2))
            max_diff_areas_A.append(np.max(np.abs(np.diff(array_A))))
            sum_of_areas_A.append(np.sum(array_A))

            std_areas_B.append(np.std(array_B))
            abs_diff_areas_B.append(np.sum(np.abs(np.diff(array_B))))
            mean_squared_error_areas_B.append(np.mean(np.diff(array_B) ** 2))
            max_diff_areas_B.append(np.max(np.abs(np.diff(array_B))))
            sum_of_areas_B.append(np.sum(polygons_B_areas))

        except:
            print("Error in tessellation")

    plt.figure()
    plt.plot(np.arange(1, len(best_solutions)+1), std_areas_A, label="Standard Deviation")
    plt.plot(np.arange(1, len(best_solutions)+1), abs_diff_areas_A, label="Absolute Difference")
    plt.plot(np.arange(1, len(best_solutions)+1), mean_squared_error_areas_A, label="Mean Squared Error")
    plt.plot(np.arange(1, len(best_solutions)+1), max_diff_areas_A, label="Maximum Difference")
    plt.xlabel("Generation")
    plt.title("Difference Metrics of A-Areas over Generations")
    plt.legend()
    plt.savefig("Fig3")

    plt.figure()
    plt.plot(np.arange(1, len(best_solutions)+1), std_areas_B, label="Standard Deviation")
    plt.plot(np.arange(1, len(best_solutions)+1), abs_diff_areas_B, label="Absolute Difference")
    plt.plot(np.arange(1, len(best_solutions)+1), mean_squared_error_areas_B, label="Mean Squared Error")
    plt.plot(np.arange(1, len(best_solutions)+1), max_diff_areas_B, label="Maximum Difference")
    plt.xlabel("Generation")
    plt.title("Difference Metrics of B-Areas over Generations")
    plt.legend()
    plt.savefig("Fig4")

    plt.figure()
    plt.plot(np.arange(1, len(best_solutions)+1), sum_of_areas_A, label="Sum of A")
    plt.xlabel("Generation")
    plt.title("Sum of Areas Generations")
    plt.legend()
    plt.savefig("Fig5")

    plt.figure()
    plt.plot(np.arange(1, len(best_solutions)+1), sum_of_areas_B, label="Sum of B")
    plt.xlabel("Generation")
    plt.title("Sum of Areas Generations")
    plt.legend()
    plt.savefig("Fig6")

    plt.figure()
    for i in range(len(array_A)):
        ith_area = [arr[i] for arr in areas_A]
        plt.plot(np.arange(1, len(best_solutions)+1), ith_area, label="Area " + str(i))
    plt.xlabel("Generation")
    plt.title("Individual A-Areas")
    plt.legend()
    plt.savefig("Fig7")

    plt.figure()
    for i in range(len(array_B)):
        ith_area = [arr[i] for arr in areas_B]
        plt.plot(np.arange(1, len(best_solutions)+1), ith_area, label="Area " + str(i))
    plt.xlabel("Generation")
    plt.title("Individual B-Areas")
    plt.legend()
    plt.savefig("Fig8")


    # Show the plot
    plt.show()

def plot_custom_solution(solution, polygons_A_star, polygons_B, ax):

    # Plotting:
    cmap = plt.get_cmap('hsv')

    # Example color indices for polygons. Adjust based on your number of polygons.
    num_polygons_A_star = len(polygons_A_star)
    num_polygons_B = len(polygons_B)
    colors_A_star = [cmap(i / num_polygons_A_star) for i in range(num_polygons_A_star)]
    colors_B = [cmap(i / num_polygons_B) for i in range(num_polygons_B)]

    #plot_solution(solution, ax, num_generations)

    for i, MainPolygon in enumerate(polygons_A_star):
        color = colors_A_star[i]
        if isinstance(MainPolygon, MultiPolygon):
            for poly in MainPolygon.geoms:
                x, y = poly.exterior.xy
                # Plot the filled polygon with alpha for the fill
                ax.fill(x, y, facecolor=color, alpha=0.1, edgecolor='none')
                # Plot the outline of the same polygon with no alpha (fully opaque)
                ax.plot(x, y, color='black', linewidth=1)
        else:
            x, y = MainPolygon.exterior.xy
            ax.fill(x, y, facecolor=color, alpha=0.1, edgecolor='none')
            ax.plot(x, y, color='black', linewidth=1)

    # Plot polygons_B in the same way
    for i, BoundaryPolygon in enumerate(polygons_B):
        color = colors_B[i]
        if isinstance(BoundaryPolygon, MultiPolygon):
            label = False
            for poly in BoundaryPolygon.geoms:
                x, y = poly.exterior.xy
                if not label:
                    ax.fill(x, y, facecolor=color, alpha=0.4, edgecolor='none', label=str(i))
                    label = True
                else:
                    ax.fill(x, y, facecolor=color, alpha=0.4, edgecolor='none')
                ax.plot(x, y, color='black', linewidth=1)
        else:
            x, y = BoundaryPolygon.exterior.xy
            ax.fill(x, y, facecolor=color, alpha=0.4, edgecolor='none', label=str(i))
            ax.plot(x, y, color='black', linewidth=1)

    voronoi_points = [[solution[i], solution[i+1]] for i in range(0, len(solution), 2)]
    helper_points = [[-100,-100], [-100,100], [100,-100], [100,100]]
    points = voronoi_points + helper_points
    points = np.array(points)

    vor = Voronoi(points)
    voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False, line_colors='black')
    ax.scatter(np.array(voronoi_points)[:, 0],  np.array(voronoi_points)[:, 1], s=20, color='red', edgecolor='black')
    #for point in points:
    #    circle = plt.Circle(point, 0.5, color='gray', alpha=0.1)
    #    ax.add_patch(circle)
    plt.xlim([-1,4])
    plt.ylim([-1,4])
    plt.legend()
    plt.savefig("Fig1")


def points_in_polygon(polygon_points, points_to_check):
    """
    Return True if any point in `points_to_check` is inside the polygon defined by `polygon_points`.
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

# Define the fitness function
def fitness_func(ga, solution, solution_idx):

    # Get Voronoi points from solution:

    points = np.array([[solution[i], solution[i+1]] for i in range(0, len(solution), 2)])

    # Tessellate to get all polygons and areas:

    #t1 = time.time()

    try:
        polygons_A_star, polygons_B, polygons_A, polygons_A_star_areas, polygons_B_areas, polygons_A_areas = tessellate_with_buffer(points, input_polygon, minimum_distance)
    except:
        print("Error in tessellation")
        return -100
    
    #t2 = time.time()
    #print(str(t2-t1) + str(" Tessellation Time"))

    #polygons_A, polygons_A_areas, vor_A = get_A_regions(points, input_polygon)

    # Get adjacency matrix, and expanded adjacency matrix.
    #t1 = time.time() 

    adjacency_matrix = adjacency_matrix_from_regions(polygons_A, minimum_distance)
    expanded_adjacency_matrix = expand_adjacency_matrix(adjacency_matrix)

    #t2 = time.time()
    #print(str(t2-t1) + str(" Adjacency Time"))

    # Create parameters for mixed-integer solver:

    #t1 = time.time() 

    task_list = np.arange(0, 2*len(points))
    task_duration = np.array(polygons_A_star_areas + polygons_B_areas)
    robots = np.tile(np.arange(1, len(points)+1), 2)
    connected_tasks = adjacency_matrix_to_connected_tasks(expanded_adjacency_matrix)

    try:
        makespan = task_scheduling_ilp(task_list, task_duration, robots, connected_tasks)
    except:
        print("Error in graph scheduling")
        return -100
    
    #t2 = time.time()
    #print(str(t2-t1) + str(" Mixed-Integer Time"))


    # Check if points are inside input polygon: 

    B = 0
    if points_in_polygon(input_polygon, points):
        B = 1

    GoalValue = -10*B - makespan

    return GoalValue


# Plot the points and center at every iteration
def plot_solution(solution, ax, generation):
    pass

# Callback function to plot the solution after each generation
def on_generation(ga_instance, best_solutions_list, ax):
    print(ga_instance.generations_completed)
    best_solution_generation = ga_instance.best_solution()[0]
    best_solutions_list.append(best_solution_generation)
    
    #solution, solution_fitness, solution_idx = ga_instance.best_solution()
    #plot_solution(solution, ax, ga_instance.generations_completed)
    
if __name__ == "__main__":

    # Define the PyGAD parameters
    N = 6
    num_generations = 50  # Number of generations
    num_parents_mating = 200  # Number of solutions to mate
    sol_per_pop = 1000  # Population size
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

    polygons_A_star, polygons_B, polygons_A, polygons_A_star_areas, polygons_B_areas, polygons_A_areas = tessellate_with_buffer(points, input_polygon, minimum_distance)
    polygons_A, polygons_A_areas, vor_A = get_A_regions(points, input_polygon)

    # Get adjacency matrix, and expanded adjacency matrix. 

    adjacency_matrix = adjacency_matrix_from_regions(polygons_A, minimum_distance)
    expanded_adjacency_matrix = expand_adjacency_matrix(adjacency_matrix)

    # Create parameters for mixed-integer solver:

    task_list = np.arange(0, 2*len(points))
    task_duration = np.array(polygons_A_star_areas + polygons_B_areas)
    robots = np.tile(np.arange(1, len(points)+1), 2)
    connected_tasks = adjacency_matrix_to_connected_tasks(expanded_adjacency_matrix)
    makespan = task_scheduling_ilp(task_list, task_duration, robots, connected_tasks, verbose=True)

    # Get theoretical best:

    polygon = create_polygon(input_polygon)
    print_area = polygon.area
    best_makespan = -print_area/float(N)

    # Plot final tessellation and fitness history:

    plot_custom_solution(solution, polygons_A_star, polygons_B, ax)
    plot_custom_fitness(ga_instance, best_solutions, best_makespan)