import trimesh 
from J_trial import get_restrictions 

PATH_a = "F:/meshes/atlas_0084.ply"
PATH_i = "F:/meshes/I3D_0084.ply"

atlas_data = trimesh.load(PATH_a)
i3d_data = trimesh.load(PATH_i)

print(atlas_data.vertices.shape) # (5767704, 3)
print(i3d_data.vertices.shape)


# 1. find the bounding box 
def bounding_box(points):
    min_x, max_x = min(points[:, 0]), max(points[:, 0])
    min_y, max_y = min(points[:, 1]), max(points[:, 1])
    min_z, max_z = min(points[:, 2]), max(points[:, 2])
    return (min_x, min_y, min_z), (max_x, max_y, max_z)


# 2. truncate the I3D
def truncate(points, min_, max_):
    
    get_restrictions()

    return 


# 3. Method1: Wweighted average
def merge(cloud1, cloud2, w1, w2):
    return