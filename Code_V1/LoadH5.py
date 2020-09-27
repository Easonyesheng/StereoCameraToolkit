import h5py
import numpy as np 
import os
from shutil import copyfile


calib_list =  '/Users/zhangyesheng/Documents/GitHub/OANet/data/big_ben_1/test/calibration.txt'
img_list = '/Users/zhangyesheng/Documents/GitHub/OANet/data/big_ben_1/test/images.txt'
img_file = '/Users/zhangyesheng/Documents/GitHub/OANet/Rectify/pics/big_ben_test/'
calib_file = '/Users/zhangyesheng/Documents/GitHub/OANet/data/big_ben_1/test/calibration/'
calib_move_file = '/Users/zhangyesheng/Documents/GitHub/OANet/Rectify/calibration/big_ben_test/'


# f = h5py.File('/Users/zhangyesheng/Documents/GitHub/OANet/Rectify/calib/calibration_000009.h5','r')
# print(np.array(f['K']))

def find_calib():
    """
    Find calib_file name accroding to images' name
    """
    img_id = parse_img_list(img_list)

    img_query = [] 
    img_name_list = os.listdir(img_file)
    for img_name in img_name_list:
        if img_name[-1] != 'g':
            continue
        img_name = img_name.split('.')[0]
        img_query.append(img_name)
    print(img_query)
    left_id, right_id = img_id.index(img_query[0]),img_id.index(img_query[1])

    # print(left_id,right_id)

    with open(calib_list,'r') as f:
        calib_name_list = f.readlines()
    
    calib_left_name, calib_right_name = calib_name_list[left_id], calib_name_list[right_id]
    calib_left_name = calib_left_name[:-1]
    calib_right_name = calib_right_name[:-1]
    # print(calib_right_name,calib_left_name)

    calib_file_query = [os.path.join(calib_file,calib_left_name.split('/')[-1]),os.path.join(calib_file,calib_right_name.split('/')[-1])]
    calib_file_move = [os.path.join(calib_move_file,img_query[0]+'.h5'),os.path.join(calib_move_file,img_query[1]+'.h5')]
    copyfile(calib_file_query[0],calib_file_move[0])
    copyfile(calib_file_query[1],calib_file_move[1])



def parse_K(calib_file,img_left_name,img_right_name):
    """
    get K,T from the .h5 file
    """
    calib_list = os.listdir(calib_file)
    K = {img_left_name:np.ones((3,3)),img_right_name:np.ones((3,3))}
    T = {img_left_name:np.ones((3,3)),img_right_name:np.ones((3,3))}
    for calib in calib_list:
        if calib.split('.')[0] == img_left_name.split('.')[0]:
            H = h5py.File(os.path.join(calib_file,calib))
            K[img_left_name] = np.array(H['K'])
            T[img_left_name] = np.array(H['T'])
            
        if calib.split('.')[0] == img_right_name.split('.')[0]:
            H = h5py.File(os.path.join(calib_file,calib))
            K[img_right_name] = np.array(H['K'])
            T[img_right_name] = np.array(H['T'])
    if len(K) == 0:
        print('There is no .h5 file in %s' %calib_file)
    # print(K)
    # print(T[img_right_name].shape)
    return K, T





def parse_img_list(img_list):
    """
    get the pure images id
    """
    img_id = []
    with open(img_list,'r') as f:
        for temp in f.readlines():
            temp = temp.split('/')[-1]
            temp = temp.split('.')[0]
            img_id.append(temp)
    # print(img_id)
    return img_id
        








def main():
    img_nameL = "40982537_3102972880.jpg"
    img_nameR = "42002003_1635942632.jpg"

    find_calib()
    # parse_img_list(img_list)
    # parse_K(calib_move_file,img_nameL,img_nameR)

if __name__ == "__main__":
    main()
