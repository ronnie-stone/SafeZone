import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d
from shapely import geometry
from get_A_regions import get_A_regions


def adjacency_matrix_from_regions(polygons, min_distance):

    n = len(polygons)
    adjacency_matrix = np.zeros((n,n), dtype=int)

    # Iterate over each pair of polygons to calculate the distance

    for i in range(n):
        for j in range(i + 1, n):
            poly1 = polygons[i]
            poly2 = polygons[j]

            # Calculate the minimum distance between the two polygons

            distance = poly1.distance(poly2)

            # If the distance is smaller than the minimum length, mark them as adjacent

            if distance <= min_distance:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    return adjacency_matrix

if __name__ == "__main__":

    # Test Case 1: Simple Square Grid
    points_1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    input_polygon_1 = np.array([(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75), (0.25,0.25)])
    min_distance_1 = 0.1
    polygons_A, _, vor_A = get_A_regions(points_1, input_polygon_1)
    adj_matrix_1 = adjacency_matrix_from_regions(polygons_A, min_distance_1)

    print("Test case 1 (Square grid) adjacency matrix:\n", adj_matrix_1)

    fig, ax = plt.subplots()
    voronoi_plot_2d(vor_A, ax=ax)
    x,y = geometry.Polygon(input_polygon_1).exterior.xy
    ax.fill(x, y, alpha=0.5, edgecolor = "black")
    plt.xlim([-0.5,1.5])
    plt.ylim([-0.5,1.5])
    plt.show()

    # Test case 2: Random points
    points_2 = np.random.rand(5, 2)
    input_polygon_2 = np.array([(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75), (0.25,0.25)])
    min_distance_2 = 0.1
    polygons_A, _, vor_A = get_A_regions(points_2, input_polygon_2)
    adj_matrix_2 = adjacency_matrix_from_regions(polygons_A, min_distance_2)

    print("Test case 2 (Random points) adjacency matrix:\n", adj_matrix_2)
    print("Voronoi Points:\n", vor_A.points[0:len(points_2)])

    fig, ax = plt.subplots()
    voronoi_plot_2d(vor_A, ax=ax)
    x,y = geometry.Polygon(input_polygon_2).exterior.xy
    ax.fill(x, y, alpha=0.5, edgecolor = "black")
    plt.xlim([-0.5,1.5])
    plt.ylim([-0.5,1.5])
    plt.show()

    # Test case 3: Triangular points
    points_3 = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    input_polygon_3 = np.array([(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75), (0.25,0.25)])
    min_distance_3 = 0.1
    polygons_A, _, vor_A = get_A_regions(points_3, input_polygon_3)
    adj_matrix_3 = adjacency_matrix_from_regions(polygons_A, min_distance_3)

    print("Test case 3 (Triangular points) adjacency matrix:\n", adj_matrix_3)

    fig, ax = plt.subplots()
    voronoi_plot_2d(vor_A, ax=ax)
    x,y = geometry.Polygon(input_polygon_3).exterior.xy
    ax.fill(x, y, alpha=0.5, edgecolor = "black")
    plt.xlim([-0.5,1.5])
    plt.ylim([-0.5,1.5])
    plt.show()
