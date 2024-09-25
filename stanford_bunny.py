import numpy as np
import matplotlib.pyplot as plt

# Load the numpy array from the .npy file
polygon = np.load('bunny_cross_section.npy')

print(polygon)

# Find the bounding box of the polygon
min_x, min_y = np.min(polygon, axis=0)
max_x, max_y = np.max(polygon, axis=0)

# Compute the width and height of the bounding box
width = max_x - min_x
height = max_y - min_y

# Calculate the scaling factor to fit the larger dimension within the 3x3 square
scale_factor = 3 / max(width, height)

# Scale the polygon
scaled_polygon = (polygon - [min_x, min_y]) * scale_factor

# Now translate the polygon to fit within the [0, 3] range
# Find the new min_x and min_y for the scaled polygon
scaled_min_x, scaled_min_y = np.min(scaled_polygon, axis=0)

# Translate the polygon so that its minimum point is at (0, 0)
translated_polygon = scaled_polygon - [scaled_min_x, scaled_min_y]

# Unpack the x and y coordinates
x, y = translated_polygon[:, 0], translated_polygon[:, 1]

# Plot the closed polygon
fig, ax = plt.subplots()
ax.fill(x, y, 'b', alpha=0.5)  # 'b' for blue color, alpha for transparency
ax.set_title('Polygon Shape')

# Show the plot
np.save('bunny_cross_section_scaled.npy', translated_polygon)
plt.show()