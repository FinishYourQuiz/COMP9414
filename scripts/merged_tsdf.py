# /home/deltamarine/COMP9491/venv37/bin/python ./merged_tsdf.py --scene scene0580_00
import argparse
import os, sys
import struct
import numpy as np
import trimesh
from tqdm import tqdm
import time

import atlas.tsdf as tsdf

t0 = time.time()

parser = argparse.ArgumentParser(description='Fuse ground truth tsdf on Scannet')
parser.add_argument("--scene", required=True, metavar="DIR", help="scene to display")
parser.add_argument("--gen_npz", required=False, default=False, metavar="DIR", help="scene to display")
# parser.add_argument("--display", default=1, metavar="DISP", help="show generated meshes")
args = parser.parse_args()

if args.gen_npz:
  file = open('../tsdf_intrinsic_' + args.scene + '.tsdf', 'rb')

  voxel_size_ = struct.unpack('<f', file.read(4))
  print("Voxel size:", voxel_size_)
  truncation_ = struct.unpack('<f', file.read(4))
  print("Truncation:", truncation_)
  integration_weight_sample_ = struct.unpack('<f', file.read(4))
  print("Intergation weight sample:", integration_weight_sample_)
  size = struct.unpack('<Q', file.read(8))
  print("Size:", size)
  max_load_factor = struct.unpack('<f', file.read(4))
  print("Max load factor:", max_load_factor)

  vec_arr = np.array([])

  index = 0
  dt = np.dtype('<i,<i,<i,<f,<f,<f')
  vec_arr = np.fromfile(file, dt)

  vec_arr = np.array(vec_arr.tolist())
  print(vec_arr.shape)
  vec_arr = vec_arr.reshape((-1, 6))
  print(vec_arr.shape)
  print(type(vec_arr[0]))
  print()
  sdf_arr = vec_arr[:, 3]
  weight_arr = vec_arr[:, 4]
  vec_arr = vec_arr[:, :3]

  print(vec_arr.shape, sdf_arr.shape, weight_arr.shape)
  print(vec_arr[:5], sdf_arr[:5], weight_arr[:5])
  file.close()

  np.savez('../tsdf_intrinsic_' +  args.scene + '.npz', vec_arr=vec_arr, sdf_arr=sdf_arr, weight_arr=weight_arr, voxel_size_=voxel_size_, truncation_=truncation_, size=size, max_load_factor=max_load_factor)

  quit()
atlas_vol = tsdf.TSDF.load('../tsdf_atlas_' +  args.scene + '.npz')
# tranf = np.array([[2., 0., 0., 0.], [0., 2., 0., 0.], [0., 0., 2., 0.]])
# atlas_vol = atlas_vol.transform()
atlas_upscale = 4
atlas_vol.tsdf_vol = np.repeat(atlas_vol.tsdf_vol, atlas_upscale, axis=2)
atlas_vol.tsdf_vol = np.repeat(atlas_vol.tsdf_vol, atlas_upscale, axis=1)
atlas_vol.tsdf_vol = np.repeat(atlas_vol.tsdf_vol, atlas_upscale, axis=0)
voxel_size = int(atlas_vol.voxel_size*100)

print(atlas_vol.tsdf_vol.shape, voxel_size)

vals = np.load('../tsdf_intrinsic_' + args.scene + '.npz')
vec_arr = vals['vec_arr'].reshape((-1, 3))
sdf_arr = vals['sdf_arr']
weight_arr = vals['weight_arr']
voxel_size_ = vals['voxel_size_']
truncation_ = vals['truncation_']
size = vals['size']
max_load_factor = vals['max_load_factor']

max_x = np.max(vec_arr[:,0])
max_y = np.max(vec_arr[:,1])
max_z = np.max(vec_arr[:,2])
print(max_x, max_y, max_z, voxel_size_)

min_x = np.min(vec_arr[:,0])
min_y = np.min(vec_arr[:,1])
min_z = np.min(vec_arr[:,2])
print(min_x, min_y, min_z, voxel_size_)
print()
print()

def get_index(vec_arr, coord):
  val = np.where((vec_arr == (coord[0], coord[1], coord[2])).all(axis=1))[0]
  if not len(val):
    return -1
  else:
    return val[0]

# print(get_index(vec_arr, (221., 621., 799.)))

scale_ratio = 2.5  # bigger value -> smaller hd room #8. / atlas_upscale
x_max = atlas_vol.tsdf_vol.shape[0] * 4
y_max = atlas_vol.tsdf_vol.shape[0] * 4
z_max = atlas_vol.tsdf_vol.shape[0] * 4
col0 = np.ones(vec_arr.shape[0]) * x_max
col1 = np.ones(vec_arr.shape[0]) * y_max
col2 = np.ones(vec_arr.shape[0]) * z_max
restriction = np.vstack((col0, col1, col2)).T.reshape((-1, 3))
print(restriction[:5])
lt_restriction = np.where((vec_arr < restriction).all(axis=1))[0]
print(lt_restriction, lt_restriction.shape)

restriction2 = restriction*0
gt_restriction = np.where((vec_arr > restriction2).all(axis=1))[0]
print(gt_restriction, gt_restriction.shape)
total_restriction = np.intersect1d(lt_restriction, gt_restriction)
print(total_restriction, total_restriction.shape)

vol_cpy = atlas_vol.tsdf_vol.detach().numpy()

restrict_sdf_arr =  sdf_arr[total_restriction]
restrict_vec_arr =  vec_arr[total_restriction]
restrict_weight_arr =  weight_arr[total_restriction]
o_x = atlas_vol.origin[0,0]
o_y = atlas_vol.origin[0,1]
o_z = atlas_vol.origin[0,2]

offset =  np.repeat(np.array([o_x+0,o_y,o_z-130]).reshape((1,3)), restrict_sdf_arr.shape[0], axis=0)
ind_arr = ((restrict_vec_arr - offset)/scale_ratio).astype(int)
vol_cpy[ind_arr[:,0], ind_arr[:,1], ind_arr[:,2]] = np.maximum(vol_cpy[ind_arr[:,0], ind_arr[:,1], ind_arr[:,2]], restrict_sdf_arr)
vol_cpy = np.clip(vol_cpy, -1., 1.)

atlas_vol.tsdf_vol = []

mesh_pred = atlas_vol.get_mesh_precleaned(vol_cpy)
print("Mesh generated.")

mesh_pred = mesh_pred.process(validate=True)

# Cleaning code
def get_connected_component(mesh, arg_index=1):
  mesh_pred = mesh.copy()
  class ufds:
    parent_node = {}
    rank = {}

    def make_set(self, u):
      for i in u:
        self.parent_node[i] = i
        self.rank[i] = 0

    def op_find(self, k):
      if self.parent_node[k] != k:
        self.parent_node[k] = self.op_find(self.parent_node[k])
      return self.parent_node[k]

    def op_union(self, a, b):
      x = self.op_find(a)
      y = self.op_find(b)

      if x == y:
        return
      if self.rank[x] > self.rank[y]:
        self.parent_node[y] = x
      elif self.rank[x] < self.rank[y]:
        self.parent_node[x] = y
      else:
        self.parent_node[x] = y
        self.rank[y] = self.rank[y] + 1

  u = np.array(range(mesh_pred.vertices.shape[0]))
  data = ufds()
  data.make_set(u)

  for i in tqdm(range(mesh_pred.vertices.shape[0])):
    adjecent_verts = mesh_pred.vertex_neighbors[i]
    for j in adjecent_verts:
      data.op_union(i, j)

  cols = np.array([data.op_find(i) for i in range(mesh_pred.vertices.shape[0])])
  colors = np.unique(cols)

  counts = np.zeros(colors.shape)
  for c in range(colors.shape[0]):
    counts[c] = np.count_nonzero(cols == colors[c])
  max_n_args = np.argsort(counts)


  max_n_i = max_n_args[-arg_index]

  max_c = colors[max_n_i]
  # mesh_pred.vertices = mesh_pred.vertices[cols != max_c]
  mesh_pred.update_vertices(cols == max_c)
  # mesh_pred.update_faces(mesh_pred.remove_degenerate_faces())
  mesh_pred = mesh_pred.process(validate=True)
  mesh_pred.invert()

  return mesh_pred

mesh_pred = get_connected_component(mesh_pred)
S0 = trimesh.transformations.scale_matrix(0.25, [0,0,0])
mesh_pred = mesh_pred.apply_transform(S0)
S1 = trimesh.transformations.translation_matrix([0.05,-.02,-0.43])
mesh_pred = mesh_pred.apply_transform(S1)

mesh_pred.export('merged_result_' + args.scene + '.ply')
print("Saved.")
mesh_pred.show()
# out_f = open('merged_result_' + args.scene + '.ply', 'rb')
# out_data = trimesh.load(out_f, 'ply')
# out_data.show()
print("Time taken:", time.time() - t0)