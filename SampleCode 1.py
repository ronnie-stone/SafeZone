import pygad
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, ConvexHull, voronoi_plot_2d
from shapely.geometry import Polygon
import time 

# General Parameters

# input_polygon = [(0,0), (3,0), (3,3), (0,3), (0,0)] # Square
# input_polygon = [(0,0), (6,0), (6,3), (0,3), (0,0)] # Rectangle
# input_polygon = [(0,0), (3,0), (1.5,3), (0,0)] # Triangle
# input_polygon = [(0,0), (3,0), (3, 0.5), (2.5, 0.5), (2.5, 2.5), (3, 2.5), (3,3), (0,3), (0, 2.5), (0.5, 2.5), (0.5,0.5), (0,0.5), (0,0)] # I-BEAM
input_polygon = [(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)] # Z-BEAM

min_length = 0.1

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def calculate_contained_area(polygon_list, input_polygon_points):
    # Create a Polygon object for the input polygon
    input_polygon = Polygon(input_polygon_points)
    
    # Check if the input polygon is valid
    if not input_polygon.is_valid:
        raise ValueError("Input polygon is invalid")
    
    # Calculate total area of the input polygon
    input_polygon_area = input_polygon.area
    
    # List to store the fraction of contained area for each polygon in polygon_list
    contained_areas = []
    
    # Iterate through each set of points in polygon_list
    for polygon_points in polygon_list:
        # Create a Polygon object for the current polygon
        current_polygon = Polygon(polygon_points)
        
        # Check if the current polygon is valid
        if not current_polygon.is_valid:
            raise ValueError("One of the polygons in the list is invalid")
        
        # Calculate the intersection (overlapping area) between the input and current polygon
        intersection = input_polygon.intersection(current_polygon)
        
        # Calculate the area of the intersection
        intersection_area = intersection.area
        
        # Calculate the fraction of the input polygon's area contained in the current polygon
        fraction_contained = intersection_area / input_polygon_area
        
        # Append the fraction contained to the result list
        contained_areas.append(fraction_contained)
    
    return contained_areas

def check_sites_in_convex_hull(voronoi_sites):
    """
    Function to check whether all Voronoi sites (points generating the Voronoi regions)
    are within the convex hull of the Voronoi vertices.

    Parameters:
    vor (scipy.spatial.Voronoi): The Voronoi tessellation object.

    Returns:
    bool: True if all Voronoi sites are within the convex hull of the Voronoi vertices, False otherwise.
    """

    # Compute the convex hull of the Voronoi sites
    hull = ConvexHull(voronoi_sites)
    
    # Get the vertices of the convex hull
    hull_vertices = hull.vertices
    
    # Convert the list of convex hull vertices to a set for quick lookup
    hull_vertex_set = set(hull_vertices)
    
    # Check if every site is a vertex of the convex hull
    for i in range(len(voronoi_sites)):
        if i not in hull_vertex_set:
            return False
    
    # If all sites are on the convex hull, return True
    return True

def check_ridge_lengths(vor, min_length):
    """
    Function to check if all Voronoi ridges are larger than the given minimum length.

    Parameters:
    vor (scipy.spatial.Voronoi): The Voronoi diagram object
    min_length (float): The minimum allowable length for the ridges

    Returns:
    bool: True if all ridges are larger than or equal to min_length, False otherwise
    """
    
    # Loop over each ridge (pair of vertices that form an edge)
    for ridge in vor.ridge_vertices:
        # Check if the ridge is finite (both vertices are not at infinity)
        if ridge[0] != -1 and ridge[1] != -1:
            # Get the coordinates of the two vertices forming the ridge
            point1 = vor.vertices[ridge[0]]
            point2 = vor.vertices[ridge[1]]
            
            # Calculate the distance between the two vertices (ridge length)
            ridge_length = np.linalg.norm(point1 - point2)
            
            # Check if the ridge length is smaller than the minimum length
            if ridge_length < min_length:
                return False
    
    # If all ridges meet the length requirement, return True
    return True

def check_degree_3_vertices(vor):
    """
    Function to check if all Voronoi vertices in the given Voronoi diagram
    have a degree of 3.

    Parameters:
    vor (scipy.spatial.Voronoi): The Voronoi diagram object

    Returns:
    bool: True if all vertices have degree 3, False otherwise
    """

    # Initialize a dictionary to count the number of edges connected to each vertex
    vertex_degree = {i: 0 for i in range(len(vor.vertices))}
    
    # Loop over the ridge_vertices (pairs of vertices that form edges)
    for ridge in vor.ridge_vertices:
        for vertex in ridge:
            if vertex != -1:  # -1 represents a point at infinity
                vertex_degree[vertex] += 1
    
    # Check if all vertices have degree 3
    for vertex, degree in vertex_degree.items():
        if degree != 3:
            return False
    
    return True

def GoalFunction(areas):

    # Return range of areas
    areas = np.array(areas)
    mean_area = np.mean(areas)

    goal1 = np.max(areas) - np.min(areas)
    goal2 = np.sum((areas - mean_area)**2)
    
    return goal2

# Define the fitness function
def fitness_func(ga, solution, solution_idx):

    # Convert to numpy array of list of lists:
    voronoi_points = [[solution[i], solution[i+1]] for i in range(0, len(solution), 2)]
    helper_points = [[-100,-100], [-100,100], [100,-100], [100,100]]
    points = voronoi_points + helper_points
    points = np.array(points)

    # Compute Voronoi tesselation
    vor = Voronoi(points)

    # Compute finite Voronoi cells
    finite_cells = []
    for i in range(len(voronoi_points)):
        region_index = vor.point_region[i]
        region = vor.regions[region_index]
        if all(v >= 0 for v in region):
            polygon = [vor.vertices[v] for v in region]
            finite_cells.append(np.array(polygon))

    areas = calculate_contained_area(finite_cells, input_polygon)

    GoalValue = GoalFunction(areas)

    # Boolean Variables:

    B1 = check_degree_3_vertices(vor)
    w1 = 0
    if not B1: w1 = -10
    
    B2 = check_ridge_lengths(vor, min_length)
    w2 = 0
    if not B2: w2 = -10
        
    B3 = check_sites_in_convex_hull(voronoi_points)
    w3 = 0
    if not B3: w3 = -10
    
    return w1 + w2 + w3 - GoalValue


# Plot the points and center at every iteration
def plot_solution(solution, generation):

    voronoi_points = [[solution[i], solution[i+1]] for i in range(0, len(solution), 2)]
    helper_points = [[-100,-100], [-100,100], [100,-100], [100,100]]
    points = voronoi_points + helper_points
    points = np.array(points)

    vor = Voronoi(points)

    # Compute finite Voronoi cells
    finite_cells = []
    for i in range(len(voronoi_points)):
        region_index = vor.point_region[i]
        region = vor.regions[region_index]
        if all(v >= 0 for v in region):
            polygon = [vor.vertices[v] for v in region]
            finite_cells.append(np.array(polygon))
    
    plt.clf()  # Clear previous plot

    # Plotting the result for visualization
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black')

    colors = ['red', 'blue', 'green', 'yellow']

    # Plot finite Voronoi cells with different colors
    for idx, cell in enumerate(finite_cells):
        color = colors[idx % 4]  # Cycle through red, blue, green, yellow
        polygon = plt.Polygon(cell, color=color, alpha=0.1)
        ax.add_patch(polygon)

    min_x = min(point[0] for point in voronoi_points)
    max_x = max(point[0] for point in voronoi_points)
    min_y = min(point[1] for point in voronoi_points)
    max_y = max(point[1] for point in voronoi_points)      

    plt.xlim(-1, 4)
    plt.ylim(-1, 4)
    plt.gca().set_aspect('equal', adjustable='box')

    input_polygon = [(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)]
    input_poly = Polygon(input_polygon)
    x, y = input_poly.exterior.xy
    plt.plot(x, y, color='black', linewidth=2)
    plt.pause(0.25)  # Pause for a moment to show the plot

# # Callback function to plot the solution after each generation
def on_generation(ga_instance):
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    plot_solution(solution, ga_instance.generations_completed)
    
    
# Define the PyGAD parameters
N = 4
num_generations = 100  # Number of generations
num_parents_mating = 50  # Number of solutions to mate
sol_per_pop = 500  # Population size
num_genes = N * 2  # Each point has 2 coordinates (x, y)
# gene_space = [{'low': 0, 'high': 3}, {'low': 0, 'high': 3}, {'low': 0, 'high': 3}, {'low': 0, 'high': 3}, {'low': 0, 'high': 3}, {'low': 0, 'high': 3}, {'low': 0, 'high': 3}, {'low': 0, 'high': 3}]

# Create the PyGAD instance
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       mutation_percent_genes=50)

# Run the GA
t1 = time.time()
ga_instance.run()
t2 = time.time()

print(t2-t1)

# Get the best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Best solution: {solution}")
print(f"Fitness of the best solution: {solution_fitness}")

plot_solution(solution, None)
ga_instance.plot_fitness()