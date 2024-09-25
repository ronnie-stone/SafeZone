import pyclipper
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely import geometry
from shapely.geometry import MultiPolygon
from matplotlib.patches import Polygon as mpl_polygon


def tessellate_with_buffer(RobotPositions, Part, BufferSize):

    minx = min(RobotPositions[:,0])
    maxx = max(RobotPositions[:,0])
    
    miny = min(RobotPositions[:,1])
    maxy = max(RobotPositions[:,1])
    
    BoundaryPoints = np.array([[minx - 100, miny - 100], [maxx + 100, miny - 100], [maxx + 100, maxy + 100], [minx - 100, maxy + 100]])
    
    Points = np.vstack((RobotPositions, BoundaryPoints))
    
    vor = Voronoi(Points)
    verticies = vor.vertices
    regions = vor.regions
    RegionIndex = vor.point_region
    
    PrintingPolygon = geometry.Polygon(Part)
    
    Cells = []
    MainPolygons = []
    MainAreas = []
    BoundaryPolygons = []
    BoundaryAreas = []

    for i in range(len(RobotPositions[:,1])):

        Cell = verticies[regions[RegionIndex[i]],:]
        Cells.append(Cell)
        Polygon = geometry.Polygon(Cell)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(Cell*10000, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        solution = pco.Execute(-BufferSize*10000)
    
    
        OffsetPolygon = []
        if solution != []:
            solution = solution[0]
            
            for j in solution:
                OffsetPolygon.append([j[0]/10000, j[1]/10000])
            
            OffsetPolygon=np.array(OffsetPolygon)
    
            OffsetPolygon = geometry.Polygon(OffsetPolygon)
            BoundaryPolygon = Polygon.difference(OffsetPolygon)    
            BoundaryPolygon = BoundaryPolygon.intersection(PrintingPolygon)
            BoundaryPolygons.append(BoundaryPolygon)
            BoundaryAreas.append(BoundaryPolygon.area)
            
            MainPolygon = OffsetPolygon.intersection(PrintingPolygon)
            MainPolygons.append(MainPolygon)
            MainAreas.append(MainPolygon.area)

        else:
            #There is no main area
            BoundaryPolygon = Polygon.intersection(PrintingPolygon)
            BoundaryPolygons.append(BoundaryPolygon)
            BoundaryAreas.append(BoundaryPolygon.area)
            
            MainPolygons.append([])
            MainAreas.append(0)
        
    return MainPolygons, BoundaryPolygons, MainAreas, BoundaryAreas


if __name__ == "__main__":

    RobotPositions = np.array([[1.31561465, 1.59776848], [1.01787483, 1.44344593], [1.13758911, 1.29757163], [1.13394131, 1.24222339]])

    #RobotPositions = np.array([[1.55228693, 2.3101726 ], [1.44672131, 2.3831709 ], [0.69051771, 1.46513396], [1.66605381, 2.24547536]])

    #RobotPositions = np.array([[0, 0], [3, 0], [3, 3], [0, 3]])
    Part = np.array([(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)])
    #Part = np.array([(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)])
    BufferSize = 0.1

    polygons_A_star, polygons_B, polygons_A_star_areas, polygons_B_areas = tessellate_with_buffer(RobotPositions, Part, BufferSize)

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