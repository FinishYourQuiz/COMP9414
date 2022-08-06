import trimesh
import argparse

# parser = argparse.ArgumentParser(description='Fuse ground truth tsdf on Scannet')
# parser.add_argument("--scene", required=True, metavar="DIR",
#     help="scene to display")
# args = parser.parse_args()

# f = open("./results/release/semseg/test_final/" + args.scene+'.ply', 'rb')
# data = trimesh.load(f, 'ply')
# data.show()
# f.close()

scene = '0084'

f = open('./manhattan_gt_' + scene + '.obj', 'rb')
data_gt = trimesh.load(f, 'obj')
# data_gt.show()
f.close()

# f = open('../manhattan_result_0050.obj', 'rb')
f = open('./Atlas/results/release/semseg/test_final/scene' + scene + '_00.ply', 'rb')
data_res = trimesh.load(f, 'ply')
# data_res.show()
f.close()

gt_v = trimesh.visual.color.ColorVisuals(data_gt, vertex_colors=(0,0,255)) # blue
res_v = trimesh.visual.color.ColorVisuals(data_res, vertex_colors=(0,255,0)) # green

gt_c = trimesh.Trimesh(data_gt.vertices, data_gt.faces, data_gt.face_normals, data_gt.vertex_normals, visual=gt_v)
res_c = trimesh.Trimesh(data_res.vertices, data_res.faces, data_res.face_normals, data_res.vertex_normals, visual=res_v)
# data_gt.visual.to_color()
# data_res.visual.to_color()

scene = trimesh.scene.Scene()
scene.add_geometry(gt_c, geom_name="ground_truth_mesh")
scene.add_geometry(res_c, geom_name="generated_mesh")

scene.show()

# f = open("tmp.txt", 'w')

# for i in range(81, 101):
#   f.write("scene00" + str(i)+ "_00\n")
# f.close()

# cd ./Atlas
# /home/deltamarine/COMP9491/venv37/bin/python prepare_data.py --path ./data --path_meta ./processed_data --dataset sample
# /home/deltamarine/COMP9491/venv37/bin/python inference.py --model results/release/semseg/final.ckpt --scenes processed_data/sample/sample1/info.json
# /home/deltamarine/COMP9491/venv37/bin/python ../show_res.py --scene scene0000_00
# /home/deltamarine/COMP9491/venv37/bin/python ./metrics.py --scene scene0000_00