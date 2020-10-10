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
    """Basic class Camera

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

    def __init__(self, config=None):
        """name
            descriptor
        Args:
            
        
        Returns:

        """
        self.name = CAMERANAME
        self.task = TASK

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
        self.img_path = ''

        self.Calibrator = Calibrator()
        self.chess_board_size = np.array(CHESSBOARDSIZE)
        self.gary_img_shape = None
        self.flag_calib = False
        self.obj_pts = []
        self.img_pts = []

        self.Evaluator = Evaluator()
        self.Reproject_err = 0.
        self.save_path = ''
        self.save_prefix = ''

        # log_init(LOGFILE)
        logging.info('\n==========================================\n')

        logging.info('\n==========New='+self.name+'=Turn==========\n')

    def init_by_config(self, yaml_path):
        """name

            initialize with yaml file
            set for stereo camera initialization
            
        Args:
            
        
        Returns:
        """

        with open(yaml_path) as f:
            logging.info('Initial camera from '+yaml_path)
            config = yaml.load(f)
            self.name = config['name']
            self.task = config['task']
            self.flag_calib = config['flag_calib']
            self.chess_board_size = np.array(config['chess_board_size'])
            self.IntP = np.array(config['intrinsic_matrix'])
            self.fx = self.IntP[0,0]
            self.fy = self.IntP[1,1]
            self.cx = self.IntP[0,2]
            self.cy = self.IntP[1,2]
            self.ExtP = np.array(config['extrinsic_matrix'])
            self.R = self.ExtP[:,:-1]
            self.t = self.ExtP[:,3]
            self.DisP = np.array(config['distortion'])
            self.img_path = config['img_path']
            self.save_path = config['save_path']
            self.save_prefix = config['save_prefix']
            logging.info('Camera initialization done.')

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
            self.Image, self.Image_num, self.gary_img_shape = self.Loader.load_images_calibration()
        if load_mod_flag == 'imgs':
            self.Image, self.Image_num = self.Loader.load_images()
            
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

        if len(self.R.shape) == 3:
            if not check_numpy_array(self.R[0]):
                logging.warning('R is empty')
            else:
                logging.info('R: '+str(self.R[0]))
        elif len(self.R.shape) == 2:
            if not check_numpy_array(self.R):
                logging.warning('R is empty')
            else:
                logging.info('R: '+str(self.R))

        if len(self.t.shape) == 3:
            if not check_numpy_array(self.t[0]):
                logging.warning('t is empty')
            else:
                logging.info('t: '+str(self.t[0]))
        elif len(self.t.shape) == 2:
            if not check_numpy_array(self.t):
                logging.warning('t is empty')
            else:
                logging.info('t: '+str(self.t))

        if not check_numpy_array(self.DisP):
            logging.warning("Disortion Parameters are not loaded")
        else:
            logging.info("Distortion Parameters: \n D:"+str(self.DisP))


        logging.info('The camera has been calibrated: '+str(self.flag_calib))

    def calibrate_camera(self):
        """name
            description
        Args:
            
        Returns:
            obj_points: used for evaluation
            img_points: used for evaluation
        """
        self.Calibrator.img = self.Image
        self.Calibrator.chess_board_size = self.chess_board_size

        self.IntError, self.IntP, self.DisP, self.R, self.t, self.obj_pts, self.img_pts = self.Calibrator.run(save_flag=True)
        self.ExtP[:,:3] = self.R[0]
        self.ExtP[:,3:] = self.t[0]
        self.flag_calib = True

        # return self.obj_pts, self.img_pts

    def evaluate_calibration(self):
        """name
            through reprojection error
        Args:
            
        Returns:

        """
        logging.info('\nStart evaluation calibration\n')
        self.Evaluator = Evaluator()
        self.Evaluator.save_path = SAVEPATH
        self.Evaluator.save_prefix = SAVEPREFIX
        error = self.Evaluator.evaluate_calibration(self.obj_pts, self.img_pts, self.R, self.t, self.IntP, self.DisP)
        self.Reproject_err = error
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
        # img_show(dst, 'undistort')

        if save_flag:
            save_path = os.path.join(os.path.join(SAVEPATH,'undistort'),self.name)
            test_dir_if_not_create(save_path)
            save_img_with_prefix(dst, save_path, self.name+'_'+str(index)+'_undistort')

    def show_img(self, index = 0, save_flag=True):
        """name
            description
        Args:
            
        Returns:

        """
        img_show(self.Image[index], 'img'+str(index))

        if save_flag:
            save_path = os.path.join(os.path.join(SAVEPATH,'ori_img'), self.name)
            test_dir_if_not_create(save_path)
            save_img_with_prefix(self.Image[index], save_path, SAVEPREFIX+'ori_img')

    def write_yaml(self, postfix=''):
        """name
            description
        Args:
            
        Returns:
        """
        camera_model = {
            'name': self.name,
            'task': self.task,
            'flag_calib': self.flag_calib,
            'chess_board_size': self.chess_board_size.tolist(),
            'intrinsic_matrix': self.IntP.tolist(),
            'extrinsic_matrix': self.ExtP.tolist(),
            'distortion': self.DisP.tolist(),
            'calib_img_shape': self.gary_img_shape.tolist(),
            'calib_img_num': self.Image_num,
            'img_path': self.img_path,
            'save_path': self.save_path,
            'save_prefix': self.save_prefix,
            'Calibration_err': self.IntError,
            'Reproject_err': self.Reproject_err
        }
        yaml_file = os.path.join(CONFIGPATH, 'camera_'+self.name+postfix+'.yaml')
        file = open(yaml_file, 'w', encoding='utf-8')
        yaml.dump(camera_model, file)
        file.close()
        logging.info('Write camera model into '+yaml_file)



if __name__ == "__main__":

    test = Camera()

    # test.show_attri()

    # yaml_path = '/Users/zhangyesheng/Desktop/Research/GraduationDesign/StereoVision/StereoCamera/config/camera_Left.yaml'
    # test.init_by_config(yaml_path)

    test.load_images(IMGPATH ,'Calibration')

    # obj_points, img_points = test.calibrate_camera()
    # print(test.R)

    test.show_attri()
    
    # test.write_yaml()
    # test.evaluate_calibration(obj_points, img_points)

    # test.show_img(save_flag=True)

    # test.undistort(save_flag=True)
