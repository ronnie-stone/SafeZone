from tessellate_with_buffer import tessellate_with_buffer
import matplotlib.pyplot as plt
import numpy as np
from create_polygon import create_polygon
import os


def plot_custom_fitness(ga_instance, best_solutions, theoretical_best_fitness, minimum_distance, input_polygon, foldername=None):

    # Use the current directory if foldername is None
    if foldername is None:
        foldername = os.getcwd()  # Get the current working directory

    # Ensure the folder exists, if not, create it
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    # Extract the fitness values from the genetic algorithm instance
    fitness_values = ga_instance.best_solutions_fitness
    polygon = create_polygon(input_polygon)
    polygon_area = polygon.area
    
    
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
    filepath2 = os.path.join(foldername, "Fig2.pdf")
    plt.savefig(filepath2, bbox_inches='tight', format='pdf')

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

    areas_A_star = []
    std_areas_A_star = []
    abs_diff_areas_A_star = []
    max_diff_areas_A_star = []
    mean_squared_error_areas_A_star = []
    sum_of_areas_A_star = []


    for i in range(len(best_solutions)):

        solution = best_solutions[i]
        points = np.array([[solution[i], solution[i+1]] for i in range(0, len(solution), 2)])

        try:
            polygons_A_star, polygons_B, polygons_A, polygons_A_star_areas, polygons_B_areas, polygons_A_areas = tessellate_with_buffer(points, input_polygon, minimum_distance)
            array_A = np.array(polygons_A_areas)/polygon_area
            array_B = np.array(polygons_B_areas)/polygon_area
            array_A_star = np.array(polygons_A_star_areas)/polygon_area
            areas_A.append(array_A)
            areas_B.append(array_B)
            areas_A_star.append(array_A_star)

            std_areas_A.append(np.std(array_A))
            abs_diff_areas_A.append(np.sum(np.abs(np.diff(array_A))))
            mean_squared_error_areas_A.append(np.mean(np.diff(array_A) ** 2))
            max_diff_areas_A.append(np.max(np.abs(np.diff(array_A))))
            sum_of_areas_A.append(np.sum(array_A))

            std_areas_B.append(np.std(array_B))
            abs_diff_areas_B.append(np.sum(np.abs(np.diff(array_B))))
            mean_squared_error_areas_B.append(np.mean(np.diff(array_B) ** 2))
            max_diff_areas_B.append(np.max(np.abs(np.diff(array_B))))
            sum_of_areas_B.append(np.sum(array_B))

            std_areas_A_star.append(np.std(array_A_star))
            abs_diff_areas_A_star.append(np.sum(np.abs(np.diff(array_A_star))))
            mean_squared_error_areas_A_star.append(np.mean(np.diff(array_A_star) ** 2))
            max_diff_areas_A_star.append(np.max(np.abs(np.diff(array_A_star))))
            sum_of_areas_A_star.append(np.sum(array_A_star))

        except:
            print(polygon_area)
            print(points, input_polygon, minimum_distance)
            print("Error in tessellation")

    plt.figure()
    plt.plot(np.arange(1, len(best_solutions)+1), std_areas_A_star, label="Standard Deviation")
    plt.plot(np.arange(1, len(best_solutions)+1), abs_diff_areas_A_star, label="Absolute Difference")
    plt.plot(np.arange(1, len(best_solutions)+1), mean_squared_error_areas_A_star, label="Mean Squared Error")
    plt.plot(np.arange(1, len(best_solutions)+1), max_diff_areas_A_star, label="Maximum Difference")
    plt.xlabel("Generation")
    plt.title("Difference Metrics of A*-Areas over Generations")
    plt.legend()
    filepath3 = os.path.join(foldername, "Fig3.pdf")
    plt.savefig(filepath3, bbox_inches='tight', format='pdf')

    plt.figure()
    plt.plot(np.arange(1, len(best_solutions)+1), std_areas_B, label="Standard Deviation")
    plt.plot(np.arange(1, len(best_solutions)+1), abs_diff_areas_B, label="Absolute Difference")
    plt.plot(np.arange(1, len(best_solutions)+1), mean_squared_error_areas_B, label="Mean Squared Error")
    plt.plot(np.arange(1, len(best_solutions)+1), max_diff_areas_B, label="Maximum Difference")
    plt.xlabel("Generation")
    plt.title("Difference Metrics of B-Areas over Generations")
    plt.legend()
    filepath4 = os.path.join(foldername, "Fig4.pdf")
    plt.savefig(filepath4, bbox_inches='tight', format='pdf')

    plt.figure()
    plt.plot(np.arange(1, len(best_solutions)+1), sum_of_areas_A_star, label="Sum of A*")
    plt.xlabel("Generation")
    plt.title("Sum of Areas Generations")
    plt.legend()
    filepath5 = os.path.join(foldername, "Fig5.pdf")
    plt.savefig(filepath5, bbox_inches='tight', format='pdf')

    plt.figure()
    plt.plot(np.arange(1, len(best_solutions)+1), sum_of_areas_B, label="Sum of B")
    plt.xlabel("Generation")
    plt.title("Sum of Areas Generations")
    plt.legend()
    filepath6 = os.path.join(foldername, "Fig6.pdf")
    plt.savefig(filepath6, bbox_inches='tight', format='pdf')

    plt.figure()
    for i in range(len(array_A)):
        ith_area = [arr[i] for arr in areas_A]
        plt.plot(np.arange(1, len(best_solutions)+1), ith_area, label="Area " + str(i))
    plt.xlabel("Generation")
    plt.title("Individual A-Areas")
    plt.legend()
    filepath7 = os.path.join(foldername, "Fig7.pdf")
    plt.savefig(filepath7, bbox_inches='tight', format='pdf')

    plt.figure()
    for i in range(len(array_B)):
        ith_area = [arr[i] for arr in areas_B]
        plt.plot(np.arange(1, len(best_solutions)+1), ith_area, label="Area " + str(i))
    plt.xlabel("Generation")
    plt.title("Individual B-Areas")
    plt.legend()
    filepath9 = os.path.join(foldername, "Fig8.pdf")
    plt.savefig(filepath9, bbox_inches='tight', format='pdf')

    plt.figure()
    for i in range(len(array_A_star)):
        ith_area = [arr[i] for arr in areas_A_star]
        plt.plot(np.arange(1, len(best_solutions)+1), ith_area, label="Area " + str(i))
    plt.xlabel("Generation")
    plt.title("Individual A*-Areas")
    plt.legend()
    filepath10 = os.path.join(foldername, "Fig9.pdf")
    plt.savefig(filepath10, bbox_inches='tight', format='pdf')

if __name__ == "__main__":
    pass