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
    
    #print(Points)
    
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
    # Create a plot
    #fig, ax = plt.subplots()
    
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
    
    
        
        #print(BoundaryPolygon)
        
        # if isinstance(MainPolygon, MultiPolygon):
        #     for poly in MainPolygon.geoms:
        #         x, y = poly.exterior.xy
        #         ax.fill(x, y, alpha=0.5, edgecolor = "black")         # Optionally fill the Polygon
        # else:
        #     x, y = MainPolygon.exterior.xy
        #     ax.fill(x, y, alpha=0.5, edgecolor = "black")         # Optionally fill the Polygon
        
        # if isinstance(BoundaryPolygon, MultiPolygon):
        #     for poly in BoundaryPolygon.geoms:
        #         x, y = poly.exterior.xy
        #         ax.fill(x, y, alpha=0.5, edgecolor = "black")         # Optionally fill the Polygon
        # else:
        #     x, y = BoundaryPolygon.exterior.xy
        #     ax.fill(x, y, alpha=0.5, edgecolor = "black")         # Optionally fill the Polygon

    #plt.show()
        
    return MainPolygons, BoundaryPolygons, MainAreas, BoundaryAreas


if __name__ == "__main__":

    RobotPositions = np.array([[1.31561465, 1.59776848], [1.01787483, 1.44344593], [1.13758911, 1.29757163], [1.13394131, 1.24222339]])

    #RobotPositions = np.array([[1.55228693, 2.3101726 ], [1.44672131, 2.3831709 ], [0.69051771, 1.46513396], [1.66605381, 2.24547536]])

    #RobotPositions = np.array([[0, 0], [3, 0], [3, 3], [0, 3]])
    Part = np.array([(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)])
    #Part = np.array([(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)])
    BufferSize = 0.1

    MainPolygons, BoundaryPolygons, MainAreas, BoundaryAreas = tessellate_with_buffer(RobotPositions, Part, BufferSize)