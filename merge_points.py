import trimesh
import numpy as np
import open3d as o3d

SCENES = ["0050", "0084", "0580", "0616"]

def bounding_box(data):
    box = {
        'x_min': np.min(data.vertices[:, 0]),
        'y_min': np.min(data.vertices[:, 1]),
        'z_min': np.min(data.vertices[:, 2]),
        'x_max': np.max(data.vertices[:, 0]),
        'y_max': np.max(data.vertices[:, 1]),
        'z_max': np.max(data.vertices[:, 2])
    }

    return box
        

def method1(FILE_i3d, FILE_atlas, out_file="save.ply"):

    data_i3d = trimesh.load(FILE_i3d)
    data_atlas = trimesh.load(FILE_atlas)

    # print(data_1.vertices.shape) # (5767704, 3)
    # print(data_2.vertices.shape) # (54661, 3)

    Bound_atlas = bounding_box(data_atlas)
    
    # -------- Remove the outliers  -------- 
    
    indices = []

    for index, vertex in enumerate(data_i3d.vertices):
        if Bound_atlas['x_min'] < vertex[0] < Bound_atlas['x_max'] and \
            Bound_atlas['y_min'] < vertex[1] < Bound_atlas['y_max'] and \
            Bound_atlas['z_min'] < vertex[2] < Bound_atlas['z_max']:
            indices.append(index)
            
    new_i3d_ver = np.copy(data_i3d.vertices[indices, ])
    new_i3d_nor = np.copy(data_i3d.vertex_normals[indices, ])
    print("new_i3d_ver.shape: ", new_i3d_ver.shape)
    print("new_i3d_nor.shape: ", new_i3d_nor.shape)

    # -------- Brutely concatenate two point clouds  -------- 

    # add point cloud
    final_points = np.concatenate((new_i3d_ver, data_atlas.vertices), axis=0)
    final_points_nor = np.concatenate((new_i3d_nor, data_atlas.vertex_normals), axis=0)
    # final_points = np.concatenate((data_i3d.vertices, data_atlas.vertices), axis=0)
    # final_points_nor = np.concatenate((data_i3d.vertex_normals, data_atlas.vertex_normals), axis=0)


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_points)
    pcd.normals = o3d.utility.Vector3dVector(final_points_nor)
    pcd.colors = o3d.utility.Vector3dVector(np.full(final_points.shape, 0.87))

    # print("pcd.points.shape: ", pcd.points.shape)
    # print("pcd.normals.shape: ", pcd.normals.shape)
    # print("pcd.colors.shape: ", pcd.colors.shape)
    # o3d.visualization.draw_geometries([pcd])

    # -------- save results  -------- 

    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
    poisson_mesh.compute_vertex_normals()
    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)
    
    o3d.visualization.draw_geometries([p_mesh_crop])

    o3d.io.write_triangle_mesh(out_file, p_mesh_crop, print_progress=True)


def call_method1():
    for scene in SCENES:
        print(f'------- SCENE: {scene} -------')
        FILE_i3d = f"data/meshes/I3D_{scene}.ply"
        FILE_atlas = f"data/meshes/atlas_{scene}.ply"
        method1(FILE_i3d, FILE_atlas, out_file=f"save_{scene}.ply")

def method2(FILE_i3d, FILE_atlas, scene):
    data_i3d = trimesh.load(FILE_i3d)
    data_atlas = trimesh.load(FILE_atlas)

    # print(data_1.vertices.shape) # (5767704, 3)
    # print(data_2.vertices.shape) # (54661, 3)

    Bound_atlas = bounding_box(data_atlas)
    
    indices = []

    # -------- 1. Remove the outliers  -------- 
    for index, vertex in enumerate(data_i3d.vertices):
        if Bound_atlas['x_min'] < vertex[0] < Bound_atlas['x_max'] and \
            Bound_atlas['y_min'] < vertex[1] < Bound_atlas['y_max'] and \
            Bound_atlas['z_min'] < vertex[2] < Bound_atlas['z_max']:
            indices.append(index)
            
    # -------- 2. Down sample  -------- 
    trial = [2, 3, 4] # down sample by reduce points
    for n in trial:
        print(f'    ------- Down sample by {n} -------')
        indices_down = [indices[i*n] for i in range(len(indices)//n)]

        new_i3d_ver = np.copy(data_i3d.vertices[indices_down, ])
        new_i3d_nor = np.copy(data_i3d.vertex_normals[indices_down, ])
        print("     new_i3d_ver.shape: ", new_i3d_ver.shape)
        print("     new_i3d_nor.shape: ", new_i3d_nor.shape)

    # -------- Brutely concatenate two point clouds  -------- 
        final_points = np.concatenate((new_i3d_ver, data_atlas.vertices), axis=0)
        final_points_nor = np.concatenate((new_i3d_nor, data_atlas.vertex_normals), axis=0)
        pcd_i3d = o3d.geometry.PointCloud()
        pcd_i3d.points = o3d.utility.Vector3dVector(final_points)
        pcd_i3d.normals = o3d.utility.Vector3dVector(final_points_nor)
        pcd_i3d.colors = o3d.utility.Vector3dVector(np.full(final_points.shape, 0.87))

        poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_i3d, depth=8, width=0, scale=1.1, linear_fit=False)[0]
        poisson_mesh.compute_vertex_normals()
        bbox = pcd_i3d.get_axis_aligned_bounding_box()
        p_mesh_crop = poisson_mesh.crop(bbox)
        
        o3d.visualization.draw_geometries([p_mesh_crop])

        # o3d.io.write_triangle_mesh(f'method2_{scene}_{n}.ply', p_mesh_crop, print_progress=True)
        o3d.io.write_triangle_mesh(f'method2_{scene}_{n}.ply', p_mesh_crop)

'''
from ai_benchmark import AIBenchmark
benchmark = AIBenchmark()
results = benchmark.run()
'''

def call_method2():
    for scene in SCENES:
        print(f'------- SCENE: {scene} -------')
        FILE_i3d = f"data/meshes/I3D_{scene}.ply"
        FILE_atlas = f"data/meshes/atlas_{scene}.ply"
        method2(FILE_i3d, FILE_atlas, scene)

call_method2()
