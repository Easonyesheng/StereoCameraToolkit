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



class Loader(object):
    """The class of image & parameter loader in different ways

        A function room.
        Never initialization.
        All the func should be without input args and return imgs.

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
    
    def load_images_calibration(self):
        """name
            load imgs default as 3 channels
        Args:
            
        
        Returns:

        """

        if check_string_is_empty(self.image_path):
            sys.exit("Load without path! ")
        
        img_names = glob.glob(os.path.join(self.image_path,'*.jpg'))
        if len(img_names) < 10: 
            logging.warning('Images not enough!')
            sys.exit('Images not enough!')
        
        N = len(img_names)
        img_temp = cv2.imread(img_names[0])
        # print('load temp img', img_names[0])
        # img_show(img_temp,'temp')
        # print('Shape', img_temp.shape)
        H, W, _ = img_temp.shape

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
        
        return img, N


        