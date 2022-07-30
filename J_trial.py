# /home/deltamarine/COMP9491/venv37/bin/python ./merged_tsdf.py --scene scene0580_00
import argparse
import os, sys
import struct
import numpy as np
import trimesh
from tqdm import tqdm
import torch

import atlas.tsdf as tsdf

def test():

    parser = argparse.ArgumentParser(description='Fuse ground truth tsdf on Scannet')
    parser.add_argument("--scene", required=True, metavar="DIR", help="scene to display")
    # parser.add_argument("--display", default=1, metavar="DISP", help="show generated meshes")
    args = parser.parse_args()

    atlas_vol = tsdf.TSDF.load('../tsdf_atlas_' +  args.scene + '.npz')
    # tranf = np.array([[2., 0., 0., 0.], [0., 2., 0., 0.], [0., 0., 2., 0.]])
    # atlas_vol = atlas_vol.transform()
    atlas_upscale = 8
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

    vol_cpy = atlas_vol.tsdf_vol.detach().numpy()

    # print(weight_arr[total_restriction[:5]])
    # print(sdf_arr[total_restriction[:5]])
    # print(vec_arr[total_restriction[:50]])

    restrict_sdf_arr =  sdf_arr[total_restriction]
    restrict_vec_arr =  vec_arr[total_restriction]
    restrict_weight_arr =  weight_arr[total_restriction]
    '''
    for i in tqdm(range(total_restriction.shape[0])):
        x, y, z = restrict_vec_arr[i]
    # print(restrict_sdf_arr[i], vol_cpy[int(x/scale_ratio), int(y/scale_ratio), int(z/scale_ratio)])
    # if vol_cpy[int(x/scale_ratio), int(y/scale_ratio), int(z/scale_ratio)] < 0:
    atlas_vol.tsdf_vol[int(x/scale_ratio), int(y/scale_ratio), int(z/scale_ratio)] = (vol_cpy[int(x/scale_ratio), int(y/scale_ratio), int(z/scale_ratio)] + restrict_sdf_arr[i]) / 2

    print(np.min(vol_cpy), np.max(vol_cpy))


    mesh_pred = atlas_vol.get_mesh()
    mesh_pred.export('merged_result_' + args.scene + '.ply')

    out_f = open('merged_result_' + args.scene + '.ply', 'rb')
    out_data = trimesh.load(out_f, 'ply')
    out_data.show()
    '''


def get_index(vec_arr, coord):
  val = np.where((vec_arr == (coord[0], coord[1], coord[2])).all(axis=1))[0]
  if not len(val):
    return -1
  else:
    return val[0]

# print(get_index(vec_arr, (221., 621., 799.)))

scale_ratio = 1

def get_restrictions(vec_arr, atlas_vol):
  col0 = np.ones(vec_arr.shape[0]) * atlas_vol.tsdf_vol.shape[0] * 4
  col1 = np.ones(vec_arr.shape[0]) * atlas_vol.tsdf_vol.shape[1] * 4
  col2 = np.ones(vec_arr.shape[0]) * atlas_vol.tsdf_vol.shape[2] * 4
  restriction = np.vstack((col0, col1, col2)).T.reshape((-1, 3))
  print(restriction[:5])
  lt_restriction = np.where((vec_arr < restriction).all(axis=1))[0]
  print(lt_restriction, lt_restriction.shape)

  restriction2 = restriction*0
  gt_restriction = np.where((vec_arr > restriction2).all(axis=1))[0]
  print(gt_restriction, gt_restriction.shape)
  total_restriction = np.intersect1d(lt_restriction, gt_restriction)
  print(total_restriction, total_restriction.shape)

  return total_restriction

