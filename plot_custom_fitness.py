from tessellate_with_buffer import tessellate_with_buffer
import matplotlib.pyplot as plt
import numpy as np


def plot_custom_fitness(ga_instance, best_solutions, theoretical_best_fitness, minimum_distance, input_polygon):
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
            # array_A = np.array(polygons_A_star_areas)
            array_A = np.array(polygons_A_areas)
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

if __name__ == "__main__":
    pass