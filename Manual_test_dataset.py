"""Get all KITTI images together"""

# KITTI files are saved in .../2011_09_26/2011_09_26_drive_00xx_sync/image_02(3)/data/*.png


import os
import shutil

KITTI_PATH = '/Users/zhangyesheng/Desktop/Research/GraduationDesign/StereoVision/self-calibration/DataSet/2011_09_26'
DIREC_PATH = '/Users/zhangyesheng/Desktop/Research/GraduationDesign/StereoVision/StereoCamera/ManualDataset'
Index_l = 0
Index_r = 0

# os.makedirs(os.path.join(DIREC_PATH,'Left'))
# os.makedirs(os.path.join(DIREC_PATH,'Right'))

file_names = os.listdir(KITTI_PATH)
# print(file_names)
for file_name in file_names:
    if file_name[0] != '2': continue
    left_img_path = os.path.join(KITTI_PATH,file_name+'/image_02/data/')
    right_img_path = os.path.join(KITTI_PATH,file_name+'/image_03/data/')

    left_imgs_names = []
    for name in os.listdir(left_img_path):
        if '.png' in name:
            left_imgs_names.append(name) 
    for left_name in left_imgs_names:
        new_name = str(Index_l)+'.png'
        shutil.copyfile(os.path.join(left_img_path,left_name),os.path.join(DIREC_PATH+'/Left/',new_name))
        Index_l += 1

    right_imgs_names = []
    for name in os.listdir(right_img_path):
        if '.png' in name:
            right_imgs_names.append(name) 
    for right_name in right_imgs_names:
        new_name = str(Index_r)+'.png'
        shutil.copyfile(os.path.join(right_img_path,right_name),os.path.join(DIREC_PATH+'/Right/',new_name))
        Index_r += 1

print(Index_r,'=',Index_l)