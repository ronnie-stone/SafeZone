import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.ops import unary_union
from shapely import geometry, buffer
from shapely.geometry import MultiPolygon, Polygon
from create_polygon import create_polygon


def tessellate_with_buffer(robot_positions, printing_part, BufferSize):

    minx = min(robot_positions[:,0])
    maxx = max(robot_positions[:,0])
    miny = min(robot_positions[:,1])
    maxy = max(robot_positions[:,1])
    
    BoundaryPoints = np.array([[minx - 100, miny - 100], [maxx + 100, miny - 100], [maxx + 100, maxy + 100], [minx - 100, maxy + 100]])
    
    Points = np.vstack((robot_positions, BoundaryPoints))
    
    vor = Voronoi(Points)
    verticies = vor.vertices
    regions = vor.regions
    RegionIndex = vor.point_region
    
    PrintingPolygon = create_polygon(printing_part)
    
    a_regions = []
    a_star_regions = []
    enlarged_a_regions = []
    b_regions = []
    a_regions_areas = []
    a_star_regions_areas = []
    b_regions_areas = []

    # Get A regions from tessellation:

    for i in range(len(robot_positions[:,1])):
        Cell = verticies[regions[RegionIndex[i]], :]
        unbounded_a_region = geometry.Polygon(Cell)
        bounded_a_region = unbounded_a_region.intersection(PrintingPolygon)
        a_regions.append(bounded_a_region)
        a_regions_areas.append(bounded_a_region.area)
        enlarged_a_region = buffer(bounded_a_region, BufferSize, quad_segs=8)
        enlarged_a_regions.append(enlarged_a_region)

    # Get buffer regions by intersecting the bounded A regions with the union of the remaining enlarged regions

    for j in range(len(robot_positions[:,1])):
        other_enlarged_a_regions = [enlarged_a_regions[k] for k in range(len(enlarged_a_regions)) if k != j]
        union_of_other_enlarged_a_regions = unary_union(other_enlarged_a_regions)
        b_region = a_regions[j].intersection(union_of_other_enlarged_a_regions)
        b_regions.append(b_region)
        b_regions_areas.append(b_region.area)

        if a_regions[j].within(b_region):
            a_star_regions.append(Polygon())
            a_star_regions_areas.append(0)
        
        else:
            a_star_regions.append(a_regions[j].difference(b_region))
            a_star_regions_areas.append(a_star_regions[j].area)

    # return MainPolygons, BoundaryPolygons, MainAreas, BoundaryAreas
    return a_star_regions, b_regions, a_regions, a_star_regions_areas, b_regions_areas, a_regions_areas


if __name__ == "__main__":

    RobotPositions = np.array([[1.31561465, 1.59776848], [1.01787483, 1.44344593], [1.13758911, 1.29757163], [1.13394131, 1.24222339]])

    #RobotPositions = np.array([[1.55228693, 2.3101726 ], [1.44672131, 2.3831709 ], [0.69051771, 1.46513396], [1.66605381, 2.24547536]])

    #RobotPositions = np.array([[0, 0], [3, 0], [3, 3], [0, 3]])
    Part = np.array([(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)])
    #Part = np.array([(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)])
    BufferSize = 0.1

    Part = [
    [(0,0), (3,0), (3,3), (0,3), (0,0)],
    [(1, 1), (1.5, 1), (1.5,1.5), (1, 1.5), (1, 1)]
    ]

    polygons_A_star, polygons_B, polygons_A, polygons_A_star_areas, polygons_B_areas, polygons_A_areas = tessellate_with_buffer(RobotPositions, Part, BufferSize)

    print(polygons_A_star)
    print(polygons_A_star_areas)
    print(polygons_B_areas)

    # Plotting:
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('hsv')

    # Example color indices for polygons. Adjust based on your number of polygons.

    num_polygons_A_star = len(polygons_A_star)
    num_polygons_B = len(polygons_B)
    colors_A_star = [cmap(i / num_polygons_A_star) for i in range(num_polygons_A_star)]
    colors_B = [cmap(i / num_polygons_B) for i in range(num_polygons_B)]

    for i, MainPolygon in enumerate(polygons_A_star):
        color = colors_A_star[i]
        if isinstance(MainPolygon, MultiPolygon):
            for poly in MainPolygon.geoms:
                x, y = poly.exterior.xy
                # Plot the filled polygon with alpha for the fill
                ax.fill(x, y, facecolor=color, alpha=0.1, edgecolor='none')
                # Plot the outline of the same polygon with no alpha (fully opaque)
                ax.plot(x, y, color='black', linewidth=1)
        elif not MainPolygon: continue
        else:
            x, y = MainPolygon.exterior.xy
            ax.fill(x, y, facecolor=color, alpha=0.1, edgecolor='none')
            ax.plot(x, y, color='black', linewidth=1)

    # Plot polygons_B in the same way
    for i, BoundaryPolygon in enumerate(polygons_B):
        color = colors_B[i]
        if isinstance(BoundaryPolygon, MultiPolygon):
            for poly in BoundaryPolygon.geoms:
                x, y = poly.exterior.xy
                ax.fill(x, y, facecolor=color, alpha=0.4, edgecolor='none')
                ax.plot(x, y, color='black', linewidth=1)
        else:
            x, y = BoundaryPolygon.exterior.xy
            ax.fill(x, y, facecolor=color, alpha=0.4, edgecolor='none')
            ax.plot(x, y, color='black', linewidth=1)

    voronoi_points = RobotPositions.tolist()
    helper_points = [[-100,-100], [-100,100], [100,-100], [100,100]]
    points = voronoi_points + helper_points
    points = np.array(points)

    vor = Voronoi(points)
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black')
    plt.xlim([-1,4])
    plt.ylim([-1,4])
    plt.show()