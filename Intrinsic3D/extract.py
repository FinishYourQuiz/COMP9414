import zipfile
import os
from PIL import Image
from tqdm import tqdm
import shutil

def unzip_():
    with zipfile.ZipFile("secnes40-44.zip", 'r') as zip_ref:
        zip_ref.extractall("./data")

# color image
def format_color():
    root = "data/processed/"
    new_root = os.path.join(root, "new_color")
    if not os.path.exists(new_root):
        os.mkdir(new_root)
    for file in tqdm(os.listdir(os.path.join(root, "color"))):
        odd_path = os.path.join(root, "color", file)
        curr = Image.open(odd_path)
        num = str(int(file.strip('.jpg'))).zfill(6)
        new_name = f"frame-{num}.color.png"
        new_path = os.path.join(new_root, new_name)
        curr.save(new_path)

# depth image
def format_depth():
    root = "data/processed/"
    new_root = os.path.join(root, "new_depth")
    if not os.path.exists(new_root):
        os.mkdir(new_root)
    for file in tqdm(os.listdir(os.path.join(root, "depth"))):
        odd_path = os.path.join(root, "depth", file)
        curr = Image.open(odd_path)
        num = str(int(file.strip('.png'))).zfill(6)
        new_name = f"frame-{num}.depth.png"
        new_path = os.path.join(new_root, new_name)
        curr.save(new_path)

# pose image
def format_pose():
    root = "data/processed/"
    new_root = os.path.join(root, "new_pose")
    if not os.path.exists(new_root):
        os.mkdir(new_root)
    for file in tqdm(os.listdir(os.path.join(root, "pose"))):
        odd_path = os.path.join(root, "pose", file)
        num = str(int(file.strip('.txt'))).zfill(6)
        new_name = f"frame-{num}.pose.txt"
        new_path = os.path.join(new_root, new_name)
        shutil.copyfile(odd_path, new_path)


def format(color=False, depth=False, pose=False):
    if color:
        print("----- Process Color Image -----")
        format_color()
    if depth:
        print("----- Process Depth Image -----")
        format_depth()
    if pose:
        print("----- Process Pose Image -----")
        format_pose()

format(color=True, depth=True, pose=True)
