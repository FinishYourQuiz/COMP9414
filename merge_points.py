import trimesh
import numpy as np
import open3d as o3d


FILE_i3d = "data/meshes/I3D_0084.ply"
FILE_atlas = "data/meshes/atlas_0084.ply"

data_i3d = trimesh.load(FILE_i3d)
data_atlas = trimesh.load(FILE_atlas)

# print(data_1.vertices.shape) # (5767704, 3)
# print(data_2.vertices.shape) # (54661, 3)
Bound_atlas = {
    'x_min': np.min(data_atlas.vertices[:, 0]),
    'y_min': np.min(data_atlas.vertices[:, 1]),
    'z_min': np.min(data_atlas.vertices[:, 2]),
    'x_max': np.max(data_atlas.vertices[:, 0]),
    'y_max': np.max(data_atlas.vertices[:, 1]),
    'z_max': np.max(data_atlas.vertices[:, 2])
}

'''
# remove the outliers
new_i3d_ver = []
new_i3d_nor = []
for index, vertex in enumerate(data_i3d.vertices):
    if Bound_atlas['x_min'] < vertex[0] < Bound_atlas['x_max'] and \
        Bound_atlas['y_min'] < vertex[1] < Bound_atlas['y_max'] and \
        Bound_atlas['z_min'] < vertex[2] < Bound_atlas['z_max']:
        new_i3d_ver.append(vertex)
        new_i3d_nor.append(data_i3d.vertex_normals[index, :])
        
new_i3d_ver = np.asarray(new_i3d_ver)
new_i3d_nor = np.asarray(new_i3d_nor)
print(new_i3d_ver.shape)
print(new_i3d_nor.shape)
'''

# add point cloud
# final_points = new_i3d_ver + data_atlas.vertex_normals
final_points = np.concatenate((data_i3d.vertices, data_atlas.vertices), axis=0)
# final_points = new_i3d_ver + data_atlas.vertex_normals
final_points_nor = np.concatenate((data_i3d.vertex_normals, data_atlas.vertex_normals), axis=0)

print(final_points.shape)
print(final_points_nor.shape)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(final_points)
pcd.normals = o3d.utility.Vector3dVector(final_points_nor)

o3d.visualization.draw_geometries([pcd])

poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
poisson_mesh.compute_vertex_normals()
bbox = pcd.get_axis_aligned_bounding_box()
p_mesh_crop = poisson_mesh.crop(bbox)