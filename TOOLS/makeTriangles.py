#%% Given a grid of points for potential reward configurations, create maximum triangles that:
# Have at least 4 spaces between points
# All triangles have the same path length

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from scipy.spatial import Delaunay

def define_cheeseboard_grid():
    # The cheeseboard has a specific non uniform grid layout:
    # [-7, 7] have range [-2,2]
    # [-6,6] have range [-4,4]
    # [-5,5] have range [-5,5]
    # [-4,-3,3,4] have range [-6,6]
    # [-2,-1,0,1,2] have range [-7,7]
    # Make an array with all of these points, and NaNs for missing points
    # Also, there is a coordinate shift such that y=0 is at index 7
    cheeseboard = np.full((15, 15), np.nan)
    cheeseboard[0, 5:10] = np.arange(-2, 3) + 7
    cheeseboard[1, 3:12] = np.arange(-4, 5) + 7
    cheeseboard[2, 2:13] = np.arange(-5, 6) + 7
    cheeseboard[3, 1:14] = np.arange(-6, 7) + 7
    cheeseboard[4, 1:14] = np.arange(-6, 7) + 7
    cheeseboard[5, 0:15] = np.arange(-7, 8) + 7
    cheeseboard[6, 0:15] = np.arange(-7, 8) + 7
    cheeseboard[7, 0:15] = np.arange(-7, 8) + 7
    cheeseboard[8, 0:15] = np.arange(-7, 8) + 7
    cheeseboard[9, 0:15] = np.arange(-7, 8) + 7
    cheeseboard[10, 1:14] = np.arange(-6, 7) + 7
    cheeseboard[11, 1:14] = np.arange(-6, 7) + 7
    cheeseboard[12, 2:13] = np.arange(-5, 6) + 7
    cheeseboard[13, 3:12] = np.arange(-4, 5) + 7
    cheeseboard[14, 5:10] = np.arange(-2, 3) + 7
    
    return cheeseboard

def make_triangles(grid, min_spacing=4):
    points = []
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i, j]):
                points.append((i, grid[i, j]))
    points = np.array(points)

    # Filter points to ensure minimum spacing
    filtered_points = []
    for point in points:
        if all(np.linalg.norm(point - np.array(other_point)) >= min_spacing for other_point in filtered_points):
            filtered_points.append(point)
    filtered_points = np.array(filtered_points)

    # Create Delaunay triangulation
    delaunay = Delaunay(filtered_points)
    triangles = delaunay.simplices

    return filtered_points, triangles

# Plot the cheeseboard grid 
plt.figure(figsize=(8, 8))
cheeseboard_grid = define_cheeseboard_grid()
plt.scatter(*np.where(~np.isnan(cheeseboard_grid)), c='black')

filtered_points, triangles = make_triangles(define_cheeseboard_grid(), min_spacing=4)

# Delaunay triangulation plot
# plt.triplot(filtered_points[:, 0], filtered_points[:, 1], triangles)
# plt.plot(filtered_points[:, 0], filtered_points[:, 1], 'o')
# plt.title('Delaunay Triangulation on Cheeseboard Grid')
# plt.show()

#%% Given the set of coordinates defining the cheeseboard, generate a set of triangles that follow criteria:
# 1) Minimum spacing of 4 between points
# 2) All triangles have the similar path length

def triangle_path_length(triangle):
    a = np.linalg.norm(triangle[0] - triangle[1])
    b = np.linalg.norm(triangle[1] - triangle[2])
    c = np.linalg.norm(triangle[2] - triangle[0])
    return a + b + c

# Generate a set of triangles to test if they meet the criteria
def generate_triangle(cheeseboard, min_spacing=4, ):
    # Randomly sample 3 points from the cheeseboard
    valid_points = np.argwhere(~np.isnan(cheeseboard))
    while True:
        sampled_indices = random.sample(range(len(valid_points)), 3)
        triangle = valid_points[sampled_indices]
        
        # Check minimum spacing
        dists = [np.linalg.norm(triangle[i] - triangle[j]) for i in range(3) for j in range(i+1, 3)]
        if all(dist >= min_spacing for dist in dists):
            # Translate the coordinates by subtracting 7 from xy-coordinates to get actual cheeseboard positions
            triangle[:, 0] -= 7
            triangle[:, 1] -= 7
            return triangle

# Continue adding triangles until no more can be added, and do this 1000 times to find the largest set
# Given the constrain that the path lengths must be similar (within 10% of each other)
def find_max_triangles(cheeseboard, min_path=20, path_max=25, dist=3, num_iterations=1000, path_length_tolerance=0.1):
    best_triangles = []
    
    for _ in range(num_iterations):
        triangles = []
        path_lengths = []
        its = 0
        while its < 100:
            triangle = generate_triangle(cheeseboard)
            length = triangle_path_length(triangle)
            its += 1
            # Check if this triangle can be added
            if not path_lengths:
                triangles.append(triangle)
                path_lengths.append(length)
            else:
                avg_length = np.mean(path_lengths)
                if abs(length - avg_length) / avg_length <= path_length_tolerance:
                    if min_path <= length <= path_max:
                        # Finally, check for minimum overlap with area of existing triangles
                        overlap = False
                        for existing_triangle in triangles:
                            # Simple overlap check: if any vertex is too close to existing triangle vertices
                            dists = [np.linalg.norm(triangle[i] - existing_triangle[j]) for i in range(3) for j in range(3)]
                            if any(d < dist for d in dists):
                                overlap = True
                                break
                        if not overlap:
                            triangles.append(triangle)
                            path_lengths.append(length)
                else:
                    continue
        
        if len(triangles) > len(best_triangles):
            best_triangles = triangles
            # print(f"New best with {len(best_triangles)} triangles, avg path length: {np.mean(path_lengths)}")
    
    return best_triangles

# best_triangles = find_max_triangles(define_cheeseboard_grid(), min_path=15, path_max=30, num_iterations=1000, path_length_tolerance=0.1)

# Plot best triangles found with different colors
# plt.figure(figsize=(8, 8))
# cheeseboard_grid = define_cheeseboard_grid()
# plt.scatter(*np.where(~np.isnan(cheeseboard_grid)), c='black')
# for triangle in best_triangles:
#     plt.fill(triangle[:, 0] + 7, triangle[:, 1] + 7, alpha=0.5)
# plt.title(f'Best Triangles Found: {len(best_triangles)}')
# plt.show()


# Search different parameters to find the best set of triangles
# values = pd.DataFrame(columns=['MinPath', 'PathMax', 'Dist', 'NumTriangles', 'AvgPathLength'])

# for min_path in [20]:
#     for path_max in [25]:
#         for dist in [2,3,4]:
#             best_triangles = find_max_triangles(define_cheeseboard_grid(), min_path=min_path, path_max=path_max, dist=dist, num_iterations=1000, path_length_tolerance=0.1)
#             print(f"Testing min_path: {min_path}, path_max: {path_max}, dist: {dist}")
#             print(f"Found {len(best_triangles)} triangles for min_path: {min_path}, path_max: {path_max}, dist: {dist}")
#             if best_triangles:
#                 avg_length = np.mean([triangle_path_length(triangle) for triangle in best_triangles])
#             else:
#                 avg_length = 0
#             temp_value = pd.DataFrame({'MinPath': [min_path], 'PathMax': [path_max], 'Dist': [dist], 'NumTriangles': [len(best_triangles)], 'AvgPathLength': [avg_length]})
#             values = pd.concat([values, temp_value], ignore_index=True)
# #%%
# print(values)


best_triangles = find_max_triangles(define_cheeseboard_grid(), min_path=20, path_max=25, dist=2, num_iterations=5000, path_length_tolerance=1)
# Plot best triangles found with different colors
plt.figure(figsize=(8, 8))
cheeseboard_grid = define_cheeseboard_grid()
plt.scatter(*np.where(~np.isnan(cheeseboard_grid)), c='black')
for triangle in best_triangles:
    plt.fill(triangle[:, 0] + 7, triangle[:, 1] + 7, alpha=0.5)
plt.title(f'Best Triangles Found: {len(best_triangles)}')
plt.show(block=True)