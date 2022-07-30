from numpy import load

def print_item(data, lst):
    '''
    FILE = 'D:/Applied_AI/Data/atlas_tsdf/scene0050_00_attributes.npz'
        semseg
        (52740,)

    FILE = 'D:/Applied_AI/Data/atlas_tsdf/scene0050_00.npz'
        origin
        (1, 3)
        voxel_size
        ()
        tsdf
        (416, 416, 128) -> (22151168, ) VS (73399227,)
        semseg
        (416, 416, 128) 
    '''
    for item in lst:
        print(item)
        print(data[item].shape)

def get_tsdf(data):
    return data['tsdf'] 


# find the bouding box of a tsdf
def bounding_box(data):
    return 



if __name__ == '__main__':
    # FILE = 'D:/Applied_AI/Data/atlas_tsdf/scene0050_00_attributes.npz'
    FILE_1 = 'D:/Applied_AI/Data/atlas_tsdf/scene0580_00.npz'
    FILE_2 = 'D:/Applied_AI/Data/I3D_tsdf/scene0580_00.npz'
    print("-------atlas_tsdf-------")
    data = load(FILE_1)
    lst = data.files
    print_item(data, lst)

    print("-------I3D_tsdf-------")
    data_2 = load(FILE_2)
    lst_2 = data_2.files
    print_item(data_2, lst_2)
    # tsdf = get_tsdf(data)
    # print(tsdf)