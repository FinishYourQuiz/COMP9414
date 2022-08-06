# /home/deltamarine/COMP9491/venv37/bin/python ./metrics.py --scene scene0000_00 --display 0

# from ai_benchmark import AIBenchmark
# benchmark = AIBenchmark()
# results = benchmark.run()
# quit()

import trimesh
import argparse
from chamferdist import ChamferDistance
import numpy as np
import torch

from atlas.evaluation import eval_mesh

parser = argparse.ArgumentParser(description='Fuse ground truth tsdf on Scannet')
parser.add_argument("--scene", required=True, metavar="DIR", help="scene to display")
parser.add_argument("--display", default=1, metavar="DISP", help="show generated meshes")
args = parser.parse_args()

# output_path = "./results/release/semseg/test_final/"
# gt_path = "/mnt/g/datasets_tmp/applied AI/scannet_data/full_scannet/scans/"

# out_f = open(output_path + args.scene + '.ply', 'rb')
# gt_f = open(gt_path + args.scene + '/' + args.scene + '_vh_clean_2.ply', 'rb')

output_mer_path = "../"
output_int_path = "../"
output_ner_path = "../"
output_man_path = "../"
output_atl_path = "./results/release/semseg/test_final/"
gt_path = "/mnt/g/datasets_tmp/applied AI/scannet_data/full_scannet/scans/"

out_man_f = open(output_man_path + 'manhattan_result_' + args.scene.split('_')[0][5:] + '.obj', 'rb')
out_ner_f = open(output_ner_path + 'neural_result_' + args.scene.split('_')[0][5:] + '.ply', 'rb')
out_atl_f = open(output_atl_path + args.scene + '.ply', 'rb')
# out_int_f = open(output_int_path + 'intrinsic3d_result_' + args.scene.split('_')[0][5:] + '.ply', 'rb')
# gt_f = open(gt_path + args.scene + '/' + args.scene + '_vh_clean_2.ply', 'rb')
gt_f = open(gt_path + args.scene + '/' + args.scene + '_vh_clean.ply', 'rb')

out_man_data = trimesh.load(out_man_f, 'obj')
out_ner_data = trimesh.load(out_ner_f, 'ply')
out_atl_data = trimesh.load(out_atl_f, 'ply')
# out_int_data = trimesh.load(out_int_f, 'ply')
gt_data = trimesh.load(gt_f, 'ply')

out_f_new = open(output_atl_path + args.scene + '_fixed.ply', 'wb')
out_f_new.write(trimesh.exchange.ply.export_ply(out_atl_data))
out_f_new.close()

out_f_new = open(output_man_path + 'manhattan_result_' + args.scene.split('_')[0][5:] + '_converted.ply', 'wb')
out_f_new.write(trimesh.exchange.ply.export_ply(out_man_data))
out_f_new.close()

out_man_f.close()
# out_ner_f.close()
# out_int_f.close()
out_atl_f.close()
gt_f.close()

# out_mer_f = open(output_mer_path + 'merged_result_' + args.scene.split('_')[0][5:] + '.obj', 'rb')
# out_mer_data = trimesh.load(out_mer_f, 'obj')

# out_f_new = open(output_man_path + 'merged_result_' + args.scene.split('_')[0][5:] + '_fixed.ply', 'wb')
# out_f_new.write(trimesh.exchange.ply.export_ply(out_mer_data))
# out_f_new.close()

# out_f_new = open(output_path + args.scene + '_new.ply', 'rb')
# out_data_new = trimesh.load(out_f_new, 'ply')
# out_f_new.close
gt_path_full =  gt_path + args.scene + '/' + args.scene + '_vh_clean.ply'
metrics_mer = eval_mesh('merged_result_' + args.scene + '.ply', gt_path_full)
# metrics_mer = eval_mesh(output_mer_path + 'merged_result_' + args.scene.split('_')[0][5:] + '.ply', gt_path_full)
metrics_atl = eval_mesh(output_atl_path + args.scene + '_fixed.ply', gt_path_full)
# metrics_man = eval_mesh(output_man_path + 'manhattan_result_' + args.scene.split('_')[0][5:] + '_converted.ply', gt_path_full)
metrics_man = eval_mesh(output_man_path + 'manhattan_result_' + args.scene.split('_')[0][5:] + '_fixed.ply', gt_path_full)
metrics_ner = eval_mesh(output_ner_path + 'neural_result_' + args.scene.split('_')[0][5:] + '_fixed.ply', gt_path_full)
metrics_int = eval_mesh(output_int_path + 'intrinsic3d_result_' + args.scene.split('_')[0][5:] + '.ply', gt_path_full)
print("Merged:", metrics_mer)
print("Atlas:", metrics_atl)
print("Manhattan SDF:", metrics_man)
print("Neural RGBD:", metrics_ner)
print("Intrinsic3D:", metrics_int)

scene = trimesh.scene.Scene()
scene.add_geometry(out_atl_data, geom_name="generated_atlas_mesh")
scene.add_geometry(out_man_data, geom_name="generated_manhattan_mesh")
# scene.add_geometry(gt_data, geom_name="ground_truth_mesh")

if int(args.display):
  out_atl_data.show()
  out_ner_data.show()
  out_man_data.show()
  # out_int_data.show()
  # gt_data.show()
  # scene.show()

# out_pc = torch.tensor([out_data.vertices], dtype=torch.float)
# gt_pc = torch.tensor([gt_data.vertices], dtype=torch.float)

# output_mesh = IO().load_mesh(output_path + args.scene + '.ply')
# gt_mesh = IO().load_mesh(gt_path + args.scene + '/' + args.scene + '_vh_clean_2.ply')
# output_mesh = pytorch3d.io.ply_io.load_ply(out_f)
# gt_mesh = pytorch3d.io.ply_io.load_ply(gt_f)

# print("Processing", args.scene)
# chamferDist = ChamferDistance()
# dist_forward = chamferDist(out_pc, gt_pc)
# print(args.scene, 'chamfer distance:', dist_forward.item(), '   normalised:', dist_forward.item() / out_pc.shape[1], dist_forward.item() / gt_pc.shape[1])