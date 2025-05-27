import open3d as o3d
import numpy as np

# Load your point cloud (assuming it's a NumPy array)
# shape: (N, 3)
points = np.load("points.npy")  

# Convert to Open3D format
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Estimate normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

# Optionally orient them consistently (e.g. outward-facing)
pcd.orient_normals_consistent_tangent_plane(k=30)

# Retrieve normals as a NumPy array
normals = np.asarray(pcd.normals)

# Combine XYZ + normals
points_with_normals = np.hstack((points, normals))  # shape: (N, 6)

# Save if needed
np.save("points_with_normals.npy", points_with_normals)
