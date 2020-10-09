"""The class for camera calibration """
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

import cv2
from tqdm import tqdm
import numpy as np
import sys
import logging
from Set.settings import *


from Util.util import *



class Calibrator(object):
    """The class for monocular camera calibration 
        
        use images to get the parameters
        https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration

    Attributes:
        img: the images used in calibrator [NxWxH]


    """
    def __init__(self):
        """name
            description
        Args:
            img: need init
            img_points:
            object_points: 
            chess_board_size: [W, H] ; need init ; like [6,7]

        Returns:

        """
        self.img = None
        self.chess_board_size = None 
        self.criteria = None
        self.object_points = [] # 3d point in real world space
        self.img_points = [] # 2d points in image plane.

    def __pre_set(self):
        """name
            termination criteria
        Args:
            

        Returns:

        """
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        return criteria

    def __get_object_point(self):
        """name
            description
        Args:

        Returns:

        """
        objp = np.zeros((self.chess_board_size[0]*self.chess_board_size[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:self.chess_board_size[1],0:self.chess_board_size[0]].T.reshape(-1,2)
        return objp

    def __find_corners(self, gray):
        """name
            description
        Args:

        Returns:
            ret: whether we find all the corners
        """
        ret, corners = cv2.findChessboardCorners(gray, (self.chess_board_size[1],self.chess_board_size[0]),None)
        return ret, corners

    def __find_corners_subpix(self, img, corners):
        """name
            description
        Args:

        Returns:
        """
        corners2 = cv2.cornerSubPix(img,corners,(11,11), (-1,-1), self.criteria)
        return corners2
    
    def __draw_and_display(self, img, corners2, index, save_flag = False, show_img_flag = False):
        """name
            Draw and display the corners
        Args:

        Returns:
        """
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_drawn = cv2.drawChessboardCorners(color_img, (self.chess_board_size[1],self.chess_board_size[0]), corners2, True)

        if show_img_flag:
            img_show(img_drawn, 'draw_corners')

        if save_flag:
            save_path = os.path.join(os.path.join(SAVEPATH,'draw_corner'),CAMERANAME)
            test_dir_if_not_create(save_path)
            save_img_with_prefix(img_drawn, save_path, SAVEPREFIX+'_DrawConners_'+str(index))
        
    def __calibrate(self):
        """name
            Draw and display the corners
        Args:

        Returns:
        """
        # print((self.img.shape[2], self.img.shape[1]))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.object_points, self.img_points, (self.img.shape[2], self.img.shape[1]),None,None)
        return ret, mtx, dist, rvecs, tvecs

    def run(self, draw_flag = True, save_flag = False):
        """name
            calibration work flow
        Args:
            
        Returns:
            ret: error
            mtx: intrinsic matrix
            dist: distortion matrix
            rvecs: R vector
            tvecs: t vector
        """
        logging.info('\n+++++++Calibration Start+++++++\n')
        if not check_numpy_array(self.img): 
            # print("Images for calibration not load! ")
            # print('img: ', self.img)
            sys.exit("Images for calibration not load! ")

        if not check_numpy_array(self.chess_board_size):
            # print("Chess board size not input !")
            # print('chess_board_size: ', type(self.chess_board_size))
            sys.exit("Chess board size not load!")
            

        logging.info('img for calibration shape: '+str(self.img.shape))
        logging.info('chess_board_size: '+str(self.chess_board_size))

        self.criteria = self.__pre_set()
        objp_temp = self.__get_object_point()

        logging.info("Calibration...")
        for i in tqdm(range(self.img.shape[0])):
            gray = self.img[i,:,:]
            gray = gray.astype(np.uint8)
            # print(gray.dtype)
            # img_show(gray,'test')
            ret, corners_temp = self.__find_corners(gray)

            if ret: 
                self.object_points.append(objp_temp)
                corners2 = self.__find_corners_subpix(gray, corners_temp)
                self.img_points.append(corners2)
                
                if draw_flag:
                    self.__draw_and_display(gray, corners2, i, save_flag, show_img_flag=False)
        
        ret, mtx, dist, rvecs, tvecs = self.__calibrate()
        logging.info("Calibration done, and the error of intrinsic is %f" % ret)

        return ret, mtx, dist, rvecs, tvecs, self.object_points, self.img_points