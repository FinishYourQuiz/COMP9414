# code for displaying multiple images in one figure
  
#import libraries
import cv2
from matplotlib import pyplot as plt
import os

# setting values to rows and column variables
rows = 3
# rows = 4
columns = 2

def show_one(image, num, name, fig):
    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, num)
    
    # showing image
    plt.imshow(image)
    plt.axis('off')
    plt.title(name, fontsize=30)

def show_all(scene_num, names_):
    print(f"scene_num: {scene_num}")

    in_path = f"../Rendered_scenes/{scene_num}"
    out_path = f"../Rendered_scenes/{scene_num}_combined"
    if not os.path.exists(in_path):
        print("Error! Cannot find the path!!")
        return
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    imgs = os.listdir(os.path.join(in_path, names_[0]))
    for img_name in imgs:
        # create figure
        fig = plt.figure(figsize=(40, 28))
        # fig = plt.figure()
        
        # reading images
        for index, name_ in enumerate(names_):
            file = os.path.join(in_path, name_, img_name)
            if not os.path.exists(file):
                continue
            print("     reading: ",file)
            image = cv2.imread(file)
            show_one(image, index+1, name_, fig)
        plt.savefig(os.path.join(out_path, img_name))

names_1 = [ "Atlas", "I3D", "Manhattan", "Neural", "GT", "PC_2_4" ]
# names_2 = [ "PC_1" ] + [f"PC_2_{i}" for i in range(2, 8, 1)]
# print(names_2)
show_all(scene_num="0050", names_=names_1)
# show_all(scene_num="0086")
# show_all(scene_num="0580")
# show_all(scene_num="0616")