import numpy as np


def chromatic_number(adjacency_matrix):
    def is_safe(vertex, color, colors):
        # Check if any adjacent vertex has the same color
        for adjacent in range(len(adjacency_matrix)):
            if adjacency_matrix[vertex][adjacent] == 1 and colors[adjacent] == color:
                return False
        return True

    def graph_coloring(m, colors, vertex):
        if vertex == len(adjacency_matrix):
            return True
        
        for color in range(1, m + 1):
            if is_safe(vertex, color, colors):
                colors[vertex] = color
                if graph_coloring(m, colors, vertex + 1):
                    return True
                colors[vertex] = 0  # Backtrack
        
        return False

    n = len(adjacency_matrix)
    colors = [0] * n

    for m in range(1, n + 1):  # Try coloring with 1 to n colors
        if graph_coloring(m, colors, 0):
            return m

    return n  # In the worst case, return n

if __name__ == "__main__":
    # Example usage
    adj_matrix = np.array([[0, 1, 0, 1],
                        [1, 0, 1, 1],
                        [0, 1, 0, 1],
                        [1, 1, 1, 0]])

    print(chromatic_number(adj_matrix))  # Output: 2