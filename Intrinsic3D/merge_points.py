import trimesh
import numpy as np
import open3d as o3d
import random
import time
from plyfile import PlyData, PlyElement

SCENES = ["0050", "0084", "0580", "0616"]

def bounding_box(data):
    box = {
        'x_min': np.min(data[:, 0]),
        'y_min': np.min(data[:, 1]),
        'z_min': np.min(data[:, 2]),
        'x_max': np.max(data[:, 0]),
        'y_max': np.max(data[:, 1]),
        'z_max': np.max(data[:, 2])
    }

    return box

def get_eara(v1, v2, v3):
    AB, AC = v2 - v1, v3 - v1
    AB_norm = np.sqrt(AB[0]**2 + AB[1]**2 + AB[2]**2)
    AC_norm = np.sqrt(AC[0]**2 + AC[1]**2 + AC[2]**2)
    cos_theta = np.dot(AB, AC) / (AB_norm * AC_norm)
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    return 0.5 * AB_norm * AC_norm * sin_theta

def uniform_points(faces, vertices, normals, N=1e6):
    points_all = []
    norms_all = []
    for face in faces:
        [i1, i2, i3] = face 
        v1, v2, v3 = np.array(vertices[i1]), np.array(vertices[i2]), np.array(vertices[i3])
        n = int( get_eara(v1, v2, v3) * N )
        if n > 0:
            # print(n)
            u, v = np.random.rand(n, 1), np.random.rand(n, 1)
            isBeyond = u + v > 1
            u[isBeyond] = 1 - u[isBeyond]
            v[isBeyond] = 1 - v[isBeyond]
            w = 1 - u - v
            points = u * v1 + v * v2 + w * v3
            normal = (normals[i1] + normals[i2] + normals[i3])/3
            points_all += points.tolist()
            normals_ = np.repeat([normal.tolist()], n, axis=0)
            norms_all += normals_.tolist()
    # print(np.array(points_all).shape)
    # print(np.array(norms_all).shape)
    return np.array(points_all), np.array(norms_all)


def method1(FILE_i3d, FILE_atlas, out_file="save.ply"):
    print("Loading files ... ")
    time_s = time.time()
    data_i3d = trimesh.load(FILE_i3d)
    data_atlas = trimesh.load(FILE_atlas)

    print("Finding uniform points ... ")
    points_i3d, norms_i3d =uniform_points(data_i3d.faces, data_i3d.vertices, data_i3d.vertex_normals)
    points_atlas, norms_atlas =uniform_points(data_atlas.faces, data_atlas.vertices, data_atlas.vertex_normals, 1e4)
    # points_i3d = np.load("data/npy_data/points_i3d.npy")
    # norms_i3d = np.load("data/npy_data/norms_i3d.npy")
    # points_atlas = np.load("data/npy_data/points_atlas.npy")
    # norms_atlas = np.load("data/npy_data/norms_atlas.npy")
    # np.save("data/npy_data/points_i3d.npy", points_i3d)
    # np.save("data/npy_data/norms_i3d.npy", norms_i3d)
    # np.save("data/npy_data/points_atlas.npy", points_atlas)
    # np.save("data/npy_data/norms_atlas.npy", norms_atlas)
    Bound_atlas = bounding_box(points_atlas)
    print("     Found ", points_i3d.shape[0], " points for intrinsic3D")
    print("     Found ", points_atlas.shape[0], " points for ATLAS")
    
    # -------- Remove the outliers  -------- 
    indices = []

    print("Removing outliers ...")
    indices_1 = np.where(
        (Bound_atlas['x_min'] <= points_i3d[:, 0]) & (points_i3d[:, 0] <= Bound_atlas['x_max'])
        & (Bound_atlas['y_min'] <= points_i3d[:, 1]) & (points_i3d[:, 1] <= Bound_atlas['y_max'])
        & (Bound_atlas['z_min'] <= points_i3d[:, 2]) & (points_i3d[:, 2] <= Bound_atlas['z_max'])
    )[0]
    
    # -------- Downsample  -------- 
    print("Downsampling points ...")
    indices_1  = random.sample(indices_1.tolist(), indices_1.shape[0]//10)
    # indices_2  = random.sample(range(points_atlas.shape[0]), points_atlas.shape[0]//2)

    new_i3d_ver = np.copy(points_i3d[indices_1, ])
    new_i3d_nor = np.copy(norms_i3d[indices_1, ])
    
    # new_atlas_ver = np.copy(points_atlas[indices_2, ])
    # new_atlas_nor = np.copy(norms_atlas[indices_2, ])
    new_atlas_ver = np.copy(points_atlas)
    new_atlas_nor = np.copy(norms_atlas)
    print("     Final #points for intrinsic3D: ", points_i3d.shape[0])
    print("     Final #points for ATLAS: ", points_atlas.shape[0])

    # -------- Concatenate two point clouds  -------- 
    print("Concatenating two point clouds ...")
    # add point cloud
    final_points = np.concatenate((new_i3d_ver, new_atlas_ver), axis=0)
    final_points_nor = np.concatenate((new_i3d_nor, new_atlas_nor), axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_points)
    pcd.normals = o3d.utility.Vector3dVector(final_points_nor)
    pcd.colors = o3d.utility.Vector3dVector(np.full(final_points.shape, 0.87))

    # -------- save results  -------- 
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
    poisson_mesh.compute_vertex_normals()
    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)
    # o3d.visualization.draw_geometries([p_mesh_crop])
    print('Saving results....')
    o3d.io.write_triangle_mesh(out_file, p_mesh_crop, print_progress=False)
    print('Finish!')
    time_d = time.time()
    print('Time used: ', time_d - time_s, 's')

def call_method1():
    # for scene in ["0050", "0084", "0580", "0616"]:
    scene = "0616"
    print(f'------- SCENE: {scene} -------')
    FILE_i3d = f"data/meshes/I3D_{scene}.ply"
    FILE_atlas = f"data/meshes/atlas_{scene}.ply"
    method1(FILE_i3d, FILE_atlas, out_file=f"merged_results/merged_mesh_{scene}.ply")

call_method1()

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
    # trial = [5, 6, 7] # [2, 3, 4] # down sample by reduce points
    # trial = [1.3, 1.5, 1.7] # down sample by reduce points
    trial = [7]
    # trial = [1.3, 1.5, 1.7]

    for n in trial:
        print(f'    ------- Down sample by {n} -------')
        indices_down = [indices[int(i*n)] for i in range(int(len(indices)//n))]

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
        
        # o3d.visualization.draw_geometries([p_mesh_crop])

        o3d.io.write_triangle_mesh(f'method2_{scene}_{n}.ply', p_mesh_crop, print_progress=True)
        # o3d.io.write_triangle_mesh(f'method2_{scene}_{n}_nn.ply', p_mesh_crop)

def call_method2(scenes):
    import time

    for scene in SCENES:
        start = time.time()
        print(f'------- SCENE: {scene} -------')
        FILE_i3d = f"data/meshes/I3D_{scene}.ply"
        FILE_atlas = f"data/meshes/atlas_{scene}.ply"
        # FILE_atlas = f"data/meshes/neural_result_{scene}_fixed.ply"
        method2(FILE_i3d, FILE_atlas, scene)
        end = time.time()
        print("time: ", end - start)

# call_method2(["0050"])
# call_method2(["0580"])
# call_method2(["0084"])
# call_method2(["0616"])
