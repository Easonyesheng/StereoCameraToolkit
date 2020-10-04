"""The class of binocular camera """
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
import numpy as np
from tqdm import tqdm
import yaml
import logging

from Set.settings import *
from Util.util import *
from ModelCamera import Camera
from ModelLoader import Loader
from ModelEvaluator import Evaluator

class StereoCamera(object):
    """class of binocular camera

        Mainly used to calculate fundamental matrix etc.

    Attributes:
        camera_left: class camera 
        camera_right: class camera

        FM: Fundamental matrix [3x3]
        FE: Estimated Fundamental matrix [3x3]
        EM: Essencial matrix [3x3]
        match_pts: matching points
        R_relate: 
        t_relate: 

        Config: a dictionary can be used to set parameters   

            
    """
    def __init__(self):
        """
        """
        self.camera_left = Camera()
        self.camera_right = Camera()

        self.FM = None
        self.FE = None
        self.match_pts1 = None
        self.match_pts2 = None

        self.EM = None

        self.R_relate = None
        self.t_relate = None

        self.loader = Loader()

        self.Evaluator = Evaluator()
    
    def init_camera_by_config(self, config_left=None, config_right=None):
        """
        """
        self.camera_left.init_by_config(config_left)
        self.camera_right.init_by_config(config_right)

    def load_FM(self, load_FM_mod = 'txt', F_flie='', index = 0):
        """name
            description
        Args:
            F_flie: File path which stores Fundamental matrix
            Index: the i-th F

        Returns:

        """

        if load_FM_mod = 'txt':
            self.FM = self.loader.Load_F_txt(F_flie)
        if load_FM_mod = 'f_list':
            self.FM = self.loader.load_F_form_Fs(F_flie, index)
        if load_FM_mod = 'f_index_list':
            self.FM = self.loader.Load_F_index(F_flie, index)
        if load_FM_mod = 'KITTI':
            self.FM = self.loader.LoadFMGT_KITTI(F_flie)
        
    def __get_normalized_F(self, F, mean, std, size=None):
        """Normalize Fundamental matrix

        """
        if size is None:
            A_resize = np.eye(3)
        else:
            orig_w, orig_h = self.shape
            new_w, new_h = size
            A_resize = np.array([
                [new_w/float(orig_w), 0.,  0.],
                [0., new_h/float(orig_h), 0.],
                [0., 0., 1.]
            ])
        A_center = np.array([
            [1, 0, -mean[0]],
            [0, 1, -mean[1]],
            [0, 0, 1.]
        ])
        A_normvar = np.array([
            [np.sqrt(2.)/std[0], 0, 0],
            [0, np.sqrt(2.)/std[1], 0],
            [0, 0, 1.]
        ])
        A = A_normvar.dot(A_center).dot(A_resize)
        A_inv = np.linalg.inv(A) 
        F = A_inv.T.dot(F).dot(A_inv)
        F /= F[2,2]
        return F

    def __sift_and_find_match(self, img1, img2):
        """
        Args:
            img1
            img2
        Returns:
            pts1
            pts2
        """
        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        good = []
        pts1 = []
        pts2 = []

        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        F,mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

        # mask 是之前的对应点中的内点的标注，为1则是内点。
        # select the inlier points
        # pts1 = pts1[mask.ravel() == 1]
        # pts2 = pts2[mask.ravel() == 1]
        self.match_pts1 = np.int32(pts1)
        self.match_pts2 = np.int32(pts2)
        return match_pts1, match_pts2

    def __matching_points_filter(self, point_len = -1):
        """name
            description
        Args:
            point_len - the pre-set matches length
        Returns:
            flag - True for the filtered matches is enough as the points_len set
        """
        try:
            self.FM.all()
        except AttributeError:
            sys.exit("Warning: Finding good matches without F_gt.")
            
        print("Use F_GT to screening matching points")
        print('Before screening, points length is {:d}'.format(len(match_pts1)))
        leftpoints = []
        rightpoints = []
        sheld = 0.1
        epsilon = 1e-5
        F = self.FM
        # use sym_epi_dist to screen
        for p1, p2 in zip(self.match_pts1, self.match_pts2):
            hp1, hp2 = np.ones((3,1)), np.ones((3,1))
            hp1[:2,0], hp2[:2,0] = p1, p2
            fp, fq = np.dot(F, hp1), np.dot(F.T, hp2)
            sym_jjt = 1./(fp[0]**2 + fp[1]**2 + epsilon) + 1./(fq[0]**2 + fq[1]**2 + epsilon)
            err = ((np.dot(hp2.T, np.dot(F, hp1))**2) * (sym_jjt + epsilon))
            # print(err)
            
            if err < sheld:
                leftpoints.append(p1)
                rightpoints.append(p2)            

            self.match_pts1 = np.array(leftpoints)
            self.match_pts2 = np.array(rightpoints)
            print('After screening, points length is {:d}'.format(len(self.match_pts1)))

        # control the length of matching points
        if point_len != -1 and point_len <= len(self.match_pts1):
            self.match_pts1 = self.match_pts1[:point_lens]
            self.match_pts2 = self.match_pts2[:point_lens]
            print("len=",self.match_pts1.shape)
        
        if self.match_pts1.shape[0] < point_len:
            return False

        return True



    def ExactGoodMatch(self,filter = False,point_len = -1):
        """Get matching points & Use F_GT to get good matching points
            1.use SIFT to exact feature points 
            if filter
            2.calculate metrics use F_GT and screening good matches
            ! use it only you have the F_GT
            :output
                bool - filter success or failed cause the matches are not enough
        """
        img1 = self.camera_left.Image   # left image
        img2 = self.camera_right.Image  # right image

        self.__sift_and_find_match(img1, img2)

        # use F_GT to select good match
        if filter:
            flag = self.__matching_points_filter(point_len=point_len)
            return flag
        
        return True
        
    def EstimateFM(self,method="RANSAC"):
        """Estimate the fundamental matrix 
            :para method: which method you use
                1.RANSAC
                2.LMedS
                3.DL(Deep Learning)
                4.8Points
            :output 
                change self.FE
                return time cost 
        """
        time_start = 0
        time_end = 0

        try: 
            self.match_pts1.all()
        except AttributeError:
            print('Exact matching points')
            self.ExactGoodMatch()

        if method == "RANSAC":
            limit_length = len(self.match_pts1)
            print('Use RANSAC with %d points' %limit_length)
            time_start = time.time()
            self.FE, self.Fmask = cv2.findFundamentalMat(self.match_pts1[:limit_length],
                                                    self.match_pts2[:limit_length],
                                                    cv2.FM_RANSAC)
            time_end = time.time()

        elif method == "LMedS":
            limit_length = len(self.match_pts1)
            print('Use LMEDS with %d points' %len(self.match_pts1))
            time_start = time.time()
            self.FE, self.Fmask = cv2.findFundamentalMat(self.match_pts1[:limit_length],
                                                    self.match_pts2[:limit_length],
                                                    cv2.FM_LMEDS)
            time_end = time.time()
            
        elif method == "8Points":
            print('Use 8 Points algorithm')
            i = -1
            while True:
                # i = np.random.randint(0,len(self.match_pts1)-7)
                i += 1
                time_start = time.time()
                self.FE, self.Fmask = cv2.findFundamentalMat(self.match_pts1[i:i+8],
                                                    self.match_pts2[i:i+8],
                                                    cv2.FM_8POINT, 0.1, 0.99)
                time_end = time.time()
                
                print('Points index: ',i)
                try: 
                    self.FE.all()
                    break
                except AttributeError:
                    continue

            
        elif method == "DL":
            # get the mask
            FE, self.Fmask = cv2.findFundamentalMat(self.match_pts1,
                                                    self.match_pts2,
                                                    cv2.FM_RANSAC, 0.1, 0.99)
            self.DL_F_Es() # need to be completed

        else:
            print("Method Error!")
            return 0
        # print(self.FE)
        F_abs = abs(self.FE)
        # self.shape = np.array([512, 1392])
        self.FE = self.FE / F_abs.max()
        # self.get_normalized_F(self.FE, mean=[0,0], std=[np.sqrt(2.), np.sqrt(2.)], size=self.shape)
        return time_end - time_start