"""The class of image & parameter loader """
# function comment template
"""name
    description
Args:
            
        
Returns:

"""
# class comment template
"""The class of 

    description

Attributes:      

"""

import os
import glob
import sys
import cv2
import numpy as np
from tqdm import tqdm
import logging

from Set.settings import *
from Util.util import *
from Util.kitti_ana import KittiAnalyse



class Loader(object):
    """The class of image & parameter loader in different ways

        A function room.
        Never initialization.
        All the func for imgs load should be without input args and return imgs.

    Attributes:

        image_path:
        para_path:

    """

    def __init__(self):
        self.image_path = ''
        self.para_path = ''
    
    def load_image_single(self):
        """Loads a image
            load a single image 
        Args:
            image_path: the absolute path of image
        
        Return: 
        """

        if check_string_is_empty(self.image_path):
            sys.exit("Load without path! ")

        img = cv2.imread(self.image_path)

        # make sure images are valid
        if img is None:
            sys.exit("Image " +self.image_path + " could not be loaded.")
       
        return img
    
    def load_images_calibration(self, load_num = -1):
        """name
            load imgs default as 3 channels
        Args:
            
        
        Returns:

        """

        if check_string_is_empty(self.image_path):
            sys.exit("Load without path! ")
        
        img_names = glob.glob(os.path.join(self.image_path,'*.jpg'))
        img_names = img_names[:load_num]
        if len(img_names) < 10: 
            logging.warning('Images not enough!')
            sys.exit('Images not enough!')
        
        N = len(img_names)
        img_temp = cv2.imread(img_names[0])
        # print('load temp img', img_names[0])
        # img_show(img_temp,'temp')
        # print('Shape', img_temp.shape)
        H, W, _ = img_temp.shape
        grayimg_shape = np.array([W, H])

        img = np.zeros((N,H,W))

        logging.info('Load %d images for calibration \n The shape is: %d , %d ' %(N, H, W))
        print('Calibration Images Loading... ')
        for i in tqdm(range(N)):
            img_temp_gray = cv2.imread(img_names[i], 0)

            # print(img_temp_gray[0,0])
            img_temp_gray.astype(np.int8)
            # print(np.max(img_temp_gray))

            # img_temp_gray = (img_temp_gray / np.max(img_temp_gray))
            # img_temp_gray = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
            img[i,:,:] = img_temp_gray
        
        return img, N, grayimg_shape

    def load_images(self, load_num = -1):
        """name
            description
        Args:
                    
                
        Returns:

        """
        img_names = glob.glob(os.path.join(self.image_path,'*.jpg'))
        img_names = img_names[:load_num]
        N = len(img_names)

        img_temp = cv2.imread(img_names[0])

        if len(img_temp.shape) == 3:
            H, W, _ = img_temp.shape
            img = np.zeros((N,H,W,3))
        else:
            H, W = img_temp.shape
            img = np.zeros((N,H,W))

        for i in tqdm(range(N)):
            img_temp = cv2.imread(img_names[i])
            
            img[i] = img_temp
        
        return img, N

    def Load_F_txt(self,FPath):
        """load F to evaluate from a txt file
            :para
                FPath : the F stored path
            :output
                FE
        """
        FE = np.loadtxt(FPath).reshape((3,3))
        return FE
    
    def load_F_form_Fs(self, F_file, line_index):
        """Get corresponding F from nx9 matrix
            The i_th line 1x9 -> F correspond to the 'i_th.jpg' 
        """
        with open(F_file) as f:
            f_list = f.readlines()
        
        F = np.array(f_list[line_index].split(),dtype=float).reshape((3,3))
        F_abs = abs(F)
        F = F / (F_abs.max()+1e-8)
        return F
    
    def Load_F_index(self, FTxtFile, Index):
        """Load F from FTxtFile
           every line in FTxtFile is a single F for single pair of images with index [Index]
        """
        with open(FTxtFile) as f:
            F_list = f.readlines()
        # print(F_list[Index])
        F_gt = F_list[Index].split()[2:] # before processed to KITTI type
        # F_gt = F_list[Index].split() # after processed to KITTI type
        # print(F_gt)
        F = np.array(F_gt,dtype=float).reshape((3,3))
        F_abs = abs(F)
        F = F/F_abs.max()
        return F
    
    def LoadFMGT_KITTI(self, F_file):
        """Load the fundamental matrix file(.txt)
            KITTI's rectified images!
            Calculate the F by 
            F = [e']P'P^+
            where e' = P'C
            where PC = 0  
        """
        paser = KittiAnalyse('', F_file, '')
        calib = paser.calib
        f_cam = '0'
        t_cam = '1'
        
        P, P_ = calib['P_rect_0{}'.format(f_cam)], calib['P_rect_0{}'.format(t_cam)]
        P = P.reshape(3,4)
        P_ = P_.reshape(3,4)
        # print('P: ',P)
        P_c = P[:,:3]
        zero = P[:,3:]
        zero = -1*zero
        c = np.linalg.solve(P_c,zero)
        C = np.ones([4,1])
        C[:3,:] = c
        e_ = np.dot(P_,C)
        e_M = np.array([
            [0, -e_[2,0], e_[1,0]],
            [e_[2,0], 0, -e_[0,0]],
            [-e_[1,0], e_[0,0], 0]
        ])
        P = np.matrix(P)
        P_wn = np.linalg.pinv(P)
        F = np.dot(np.dot(e_M, P_),P_wn)
        F_abs = abs(F)
        F = F/ F_abs.max()

        return F

    