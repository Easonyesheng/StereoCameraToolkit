"""The class which defines camera """
# function comment template
"""name
    description
Args:
            
        
Returns:

"""

# class comment template
"""The class of 

    descriptor

Attributes:      

"""

import cv2
import numpy as np
from tqdm import tqdm
import yaml
import logging

from ModelLoader import Loader
from ModelEvaluator import Evaluator
from ModelCalibrator import Calibrator
from Util.util import *
from Set.settings import *

class Camera(object):
    """Top class Camera

        Base class. 
        Can be initialized by config file or settings.
        Default as settings.

    Attributes:
        name: name of the camera
        task: the usage of camera
        Config: a dictionary can be used to set parameters   

        IntP: intrinsic matrix for camera [3x3]
        fx: 1x1
        fy: 1x1
        cx: 1x1
        cy: 1x1

        ExtP: extrinsic matrix for cemera [R t]
        R: Nx3x3
        t: Nx3x1
        N is the Nth R, t in the Nth calibration chess board

        DisP: distortion parameters [k1 k2 p1 p2 k3]

        Image: Images taken by this camera [NxHxW] 
            N - number of images
            HxW - size 


        Loader: class-Loader for loading images & parameters

        Calibrator: class-Calibrator
        flag_calib: has been calibrated or not [bool]

        Evaluator: class-Evaluator

    Functions:
        Calibrate camera
        Undistort image
        show the img

    """

    def __init__(self, config=None, camera_name='camera'):
        """name
            descriptor
        Args:
            
        
        Returns:

        """
        self.name = camera_name
        self.task = TASK

        self.config = config

        self.IntP = np.zeros((3,3))
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.IntError = 0.0
        
        self.ExtP = np.zeros((3,4))
        self.R = np.zeros((3,3))
        self.t = np.zeros((3,1))

        self.DisP = np.zeros((1,5))

        self.Image = None
        self.Image_num = 0

        self.Loader = Loader()

        self.Calibrator = Calibrator()
        self.flag_calib = False

        self.Evaluator = Evaluator()

        log_init(LOGFILE)
        logging.info('\n==========================================\n')

        logging.info('\n==========New='+self.name+'Turn===========\n')

    
    def init_by_config(self, yaml_path):
        """name
            initialize with yaml file
        Args:
            
        
        Returns:
        """
        pass


    def load_images(self, image_path, load_mod_flag):
        """name
            
        Args:
            
        
        Returns:

        """
        self.Loader.image_path = image_path

        if load_mod_flag == 'Signal':
            self.Image = self.Loader.load_image_single()
            self.Image_num = 1
        if load_mod_flag == 'Calibration':
            self.Image, self.Image_num = self.Loader.load_images_calibration()


        

        logging.info('In Mod: %s, Image loading DONE.\nself.Image shape is: ' %load_mod_flag+str(self.Image.shape))
    
    def show_attri(self, show_img_flag=False):
        """name
            descriptor
        Args:
            
        
        Returns:

        """
        logging.info('\nShow attributes\n')
        # print('Camera name: ', self.name)
        logging.info('Camera name: '+self.name)

        logging.info("Images: ")
        if check_numpy_array(self.Image):
            if self.Image_num == 1:
                logging.info('Image shape:'+str(self.Image.shape))
                if show_img_flag:
                    img_show(self.Image, 'Img')
            else:
                logging.info('%d Images got.\nShape is:'%self.Image_num+str(self.Image.shape))
                if show_img_flag:
                    img_show(self.Image[0], 'Img')

        else:
            logging.warning('No Image')
        
        
        if not check_numpy_array(self.IntP):
            logging.warning("Intinsic Parameters are not loaded")
        else:
            logging.info("Intrinsic Parameters: \n K:"+str(self.IntP))

        if not check_numpy_array(self.ExtP):
            logging.warning("Extrinsic Parameters are not loaded")
        else:
            logging.info("Extrinsic Parameters: \n EP:"+str(self.ExtP))


        if not check_numpy_array(self.R[0]):
            logging.warning('R is empty')
        else:
            logging.info('R: '+str(self.R[0]))
        
        if not check_numpy_array(self.t[0]):
            logging.warning('t is empty')
        else:
            logging.info('t: '+str(self.t[0]))

        if not check_numpy_array(self.DisP):
            logging.warning("Disortion Parameters are not loaded")
        else:
            logging.info("Distortion Parameters: \n D:"+str(self.DisP))


        logging.info('The camera has been calibrated: '+str(self.flag_calib))


    def calibrator(self):
        """name
            description
        Args:
            
        Returns:
            obj_points: used for evaluation
            img_points: used for evaluation
        """
        self.Calibrator.img = self.Image
        self.Calibrator.chess_board_size = np.array(CHESSBOARDSIZE)

        self.IntError, self.IntP, self.DisP, self.R, self.t, obj_points, img_points = self.Calibrator.run(save_flag=True)
        self.flag_calib = True

        return obj_points, img_points

    def evaluate_calibration(self, obj_points, img_points):
        """name
            through reprojection error
        Args:
            
        Returns:

        """
        logging.info('\nStart evaluation calibration\n')
        self.Evaluator = Evaluator()
        self.Evaluator.save_path = SAVEPATH
        self.Evaluator.save_prefix = SAVEPREFIX
        error = self.Evaluator.evaluate_calibration(obj_points, img_points, self.R, self.t, self.IntP, self.DisP)
        logging.info("Calibration error (Reprojection) is:"+str(error))


    def undistort(self, index = 0, save_flag=False):
        """name
            description
        Args:
            
        Returns:

        """
        img = self.Image[index]
        h, w = img.shape
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.IntP, self.DisP, (w,h), 1, (w,h))

        dst = cv2.undistort(img, self.IntP, self.DisP, None, newcameramtx)
        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        img_show(dst, 'undistort')

        if save_flag:
            test_dir_if_not_create(os.path.join(SAVEPATH,'undistort'))
            save_img_with_prefix(dst, os.path.join(SAVEPATH,'undistort'), SAVEPREFIX+'_undistort')

    def show_img(self, index = 0, save_flag=True):
        """name
            description
        Args:
            
        Returns:

        """
        img_show(self.Image[index], 'img'+str(index))

        if save_flag:
            test_dir_if_not_create(os.path.join(SAVEPATH,self.name+'_ori_img'))
            save_img_with_prefix(self.Image[index], os.path.join(SAVEPATH,self.name+'_ori_img'), SAVEPREFIX+'ori_img')

    def write_log(self):
        """name
            description
        Args:
            
        Returns:
        """
        pass



if __name__ == "__main__":

    test = Camera(camera_name=CAMERANAME)

    test.show_attri()

    test.load_images(IMGPATH ,'Calibration')

    obj_points, img_points = test.calibrator()
    # print(test.R)

    test.show_attri()

    test.evaluate_calibration(obj_points, img_points)

    test.show_img(save_flag=True)

    test.undistort(save_flag=True)
