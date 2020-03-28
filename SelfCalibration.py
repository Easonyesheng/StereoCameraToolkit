# -*- coding: utf-8 -*-

"""A system to do stereo camera self-calibration"""

import numpy as np 
import sys
import os
import tensorflow as tf 
import cv2
from PIL import Image
from kitti_ana import KittiAnalyse
from rect import mathching, returnH1_H2, getRectifystereo, compute_epipole
from LoadH5 import parse_K
from math import sqrt, acos

class SelfCalibration:
    """
    自标定系统
        功能：
            1.读入图片
            2.读入参数 -- KITTI(txt) & YFCC(h5)
            3.估计F
            4.F评估 -- 可视化+量化
            5.校正
            
   
    """

    def __init__(self,ImgPath,ParaPath,SavePath,SavePrefix):

        self.ImgPath = ImgPath
        self.ParaPath = ParaPath
        self.SavePath = SavePath
        self.SavePrefix = SavePrefix
        self.F = None
        self.FE = None
        self.imgl = None
        self.imgr = None
        self.Kl = None
        self.Kr = None
        self.dr = None
        self.dl = None

        # create new folders to save
        if not os.path.exists(self.SavePath):
            print("Create Save Path: ",self.SavePath)
            os.makedirs(self.SavePath)



    def load_image_pair(self, img_nameL, img_nameR):
        """Loads pair of images 

        Two view images in the same file -- ImgPath + img_namel/r

        This method loads the two images for which the 3D scene should be
        reconstructed. The two images should show the same real-world scene
        from two different viewpoints.

        :param img_nameL: name of left image
        :param img_nameR: name of right image
            
        """
        self.img_left_name, self.img_right_name = img_nameL,img_nameR
        self.imgl = cv2.imread(os.path.join(self.ImgPath,img_nameL), 0)
        self.imgr = cv2.imread(os.path.join(self.ImgPath,img_nameR), 0)
        # print(self.imgl.shape)
        # self.imgl = cv2.resize(self.imgl,(256,256))
        # self.imgr = cv2.resize(self.imgr,(256,256))

        # make sure images are valid
        if self.imgl is None:
            sys.exit("Image " +os.path.join(self.ImgPath,img_nameL) + " could not be loaded.")
        if self.imgr is None:
            sys.exit("Image " + os.path.join(self.ImgPath,img_nameR) + " could not be loaded.")

        if len(self.imgl.shape) == 2:
            self.imgl = cv2.cvtColor(self.imgl, cv2.COLOR_GRAY2BGR)
            self.imgr = cv2.cvtColor(self.imgr, cv2.COLOR_GRAY2BGR)
    

    def load_image_KITTI(self,index):
        '''load the images in the format of KITTI
            the images are in /ImgPath/image_00(left)/data/00000000xx.pmg
            :para 
                index : the image index

        '''
        # print(len(str(index).zfill(10-len(str(index)))))
        self.img_left_name = os.path.join(self.ImgPath,'image_00/data/'+str(index).zfill(10)+'.png')
        self.img_right_name = os.path.join(self.ImgPath,'image_01/data/'+str(index).zfill(10)+'.png')

        self.imgl = cv2.imread(self.img_left_name)
        self.imgr = cv2.imread(self.img_right_name)

        # make sure images are valid
        if self.imgl is None:
            sys.exit("Image " +self.img_left_name + " could not be loaded.")
        if self.imgr is None:
            sys.exit("Image " + self.img_right_name + " could not be loaded.")

        if len(self.imgl.shape) == 2:
            self.imgl = cv2.cvtColor(self.imgl, cv2.COLOR_GRAY2BGR)
            self.imgr = cv2.cvtColor(self.imgr, cv2.COLOR_GRAY2BGR)


    def load_img_test(self,Index):
        """Load images in manual dataset
            ImgPath should be .../ManualDataset/
            Images are saved in: 
                .../ManualDataset/Left/Index.png
                .../ManualDataset/Right/Index.png
        """
        self.img_left_name = os.path.join(self.ImgPath+'/Left',str(Index)+'.png')
        self.img_right_name = os.path.join(self.ImgPath+'/Right',str(Index)+'.png')

        self.imgl = cv2.imread(self.img_left_name)
        self.imgr = cv2.imread(self.img_right_name)

        # make sure images are valid
        if self.imgl is None:
            sys.exit("Image " +self.img_left_name + " could not be loaded.")
        if self.imgr is None:
            sys.exit("Image " + self.img_right_name + " could not be loaded.")


    def Show(self):
        """Show all the dataset
        """
        print("Path: \nImgPath: %s \nParaPath: %s \nSavePath: %s" %(self.ImgPath,self.ParaPath,self.SavePath))
        print("Images: Showing")
        try: 
            self.imgl.all()
            cv2.startWindowThread()
            cv2.imshow("Left",self.imgl)
            cv2.imshow("Right",self.imgr)
            k = cv2.waitKey(0)
            if k == 27:
                cv2.destroyAllWindows()
        except AttributeError:
            print("Images not loaded")
        
        # print("Parameters: \n Kl:",self.Kl,"  \nKr: ",self.Kr," \ndl: ",self.dl," \ndr:",self.dr)
        try:
            self.Kr.all()
            print("Parameters: \n Kl:",self.Kl,"  \nKr: ",self.Kr," \ndl: ",self.dl," \ndr:",self.dr)
        except:
            print("Parameters are not loaded")

        print("F: \nF_GT:",self.F,"\nFE:",self.FE)

            
    def LoadPara_KITTI(self):
        """Load the camera parameters from KITTI's calib_file

            Load the parameters and undistort the images

            :para Kl: left camera's 3x3 intrinsic camera matrix
            :para Kr: right camera's 3x3 intrinsic camera matrix
            :para dl: vector of distortion coefficients of left camera
            :para dr: vector of distortion coefficients of right camera

        """
        ParaPath = os.listdir(self.ParaPath)[-1]
        print(ParaPath)
        paser = KittiAnalyse(self.ImgPath,os.path.join(self.ParaPath,ParaPath),self.SavePath)
        calib = paser.calib
        Kl,Kr = calib['K_0{}'.format('0')],calib['K_0{}'.format('1')]
        dl,dr = calib['D_0{}'.format('0')],calib['D_0{}'.format('1')]
        Rl,Rr = calib['R_rect_0{}'.format('0')],calib['R_rect_0{}'.format('1')]
        Pl,Pr = calib['P_rect_0{}'.format('0')],calib['P_rect_0{}'.format('1')]

        self.Kl = Kl
        self.Kr = Kr
        self.Kl_inv = np.linalg.inv(Kl)  # store inverse for fast access
        self.Kr_inv = np.linalg.inv(Kr)  # store inverse for fast access
        self.dl = dl
        self.dr = dr
        # for rect
        self.Pl = Pl.reshape((3,4)) 
        self.Pr = Pr.reshape((3,4)) 
        self.Rl = Rl
        self.Rr = Rr 
        # print(Pl)
        # print(Rl)


        # # undistort the images
        # self.imgl = cv2.undistort(self.imgl, self.Kl, self.dl)
        # self.imgr = cv2.undistort(self.imgr, self.Kr, self.dr)
    
    def LoadPara_YFCC(self):
        """Load the camera parameters from YFCC's calib_file

            Load the parameters and undistort the images

            :para Kl: left camera's 3x3 intrinsic camera matrix
            :para Kr: right camera's 3x3 intrinsic camera matrix
            :para dl: vector of distortion coefficients of left camera ; set to [0,0,0,0,0]
            :para dr: vector of distortion coefficients of right camera ; set to [0,0,0,0,0]
        """
        K, T = parse_K(self.ParaPath,self.img_left_name,self.img_right_name)
        # print(type(K))
        self.Kl = K[self.img_left_name]
        self.Kr = K[self.img_right_name]
        self.Kl_inv = np.linalg.inv(self.Kl)  # store inverse for fast access
        self.Kr_inv = np.linalg.inv(self.Kr)  # store inverse for fast access
        self.Tl = T[self.img_left_name]
        self.Tr = T[self.img_right_name]
        self.dl = np.zeros((1,5))
        self.dr = np.zeros((1,5))

    def LoadCorr(self,rightcorr,leftcorr):
        """Load inliers corr infered by OANet
        """
        leftcorr = np.load(leftcorr)
        rightcorr = np.load(rightcorr)
        if leftcorr.shape[1] == 4:
            points_quan = int(leftcorr.shape[0]*2)
            corr_left = np.ones((points_quan,2),dtype=int)
            corr_right = np.ones((points_quan,2),dtype=int)
            corr_left[:points_quan//2,:] = leftcorr[:,:2]
            corr_left[points_quan//2:points_quan] = leftcorr[:,2:4]
            corr_right[:points_quan//2,:] = rightcorr[:,:2]
            corr_right[points_quan//2:points_quan] = rightcorr[:,2:4]
        else:
            corr_left = leftcorr
            corr_right = rightcorr
        self.match_pts1 = corr_left.astype(np.int)
        self.match_pts2 = corr_right.astype(np.int)
        print("OANet get %d matching points" %len(self.match_pts1))

    def Load_F_test(self,FPath):
        """load F to evaluate
            :para
                FPath : the F stored path
            :output
                 as self.FE
        """
        self.FE = np.loadtxt(FPath).reshape((3,3))
        # if self.FE.shape[0] == 1:
        #     self.FE = self.FE.reshape((3,3))
        # if self.FE.shape[0]*self.FE.shape[1] != 9:
        #     print('ERROR: Wrong Shape:'.self.FE.shape)
        

    def LoadFMGT_KITTI(self):
        """Load the fundamental matrix file(.txt)
            KITTI's rectified images!
            Calculate the F by 
            F = [e']P'P^+
            where e' = P'C
            where PC = 0  
        """
        ParaPath = os.listdir(self.ParaPath)[-1]
        paser = KittiAnalyse(self.ImgPath,os.path.join(self.ParaPath,ParaPath),self.SavePath)
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
        self.F = F/ F_abs.max()

        return self.F

        
    

    def get_normalized_F(self, F, mean, std, size=None):
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



    def ExactGoodMatch(self,screening = False,point_lens = -1):
        """Get matching points & Use F_GT to get good matching points
            1.use SIFT to exact feature points 
            if screening
            2.calculate metrics use F_GT and screening good matches
            ! use it only you have the F_GT
            :output
                bool: 
                    True for point_len == 0
        """
        # try:
        #     self.F.all()
        # except AttributeError:
        #     print("ERROR: No F_GT found !")
        #     return

        img1 = self.imgl  #queryimage # left image
        img2 = self.imgr #trainimage # right image

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


        # use F_GT to select good match
        if screening:
            try:
                self.F.all()
            except AttributeError:
                print("There is no F_GT, you can use LoadFMGT_KITTI() to get it.\nWarning: Without screening good matches.")
                return
            print("Use F_GT to screening matching points")
            print('Before screening, points length is {:d}'.format(len(self.match_pts1)))
            leftpoints = []
            rightpoints = []
            sheld = 0.01
            epsilon = 1e-5
            # use sym_epi_dist to screen
            for p1, p2 in zip(self.match_pts1, self.match_pts2):
                hp1, hp2 = np.ones((3,1)), np.ones((3,1))
                hp1[:2,0], hp2[:2,0] = p1, p2
                fp, fq = np.dot(F, hp1), np.dot(F.T, hp2)
                sym_jjt = 1./(fp[0]**2 + fp[1]**2 + epsilon) + 1./(fq[0]**2 + fq[1]**2 + epsilon)
                err = ((np.dot(hp2.T, np.dot(F, hp1))**2) * (sym_jjt + epsilon))
                # print(err)
                
            # # use epi_cons to screen
            # for p1,p2 in zip(self.match_pts1, self.match_pts2):
            #     hp1, hp2 = np.ones((3,1)), np.ones((3,1))
            #     hp1[:2,0], hp2[:2,0] = p1, p2 
            #     err = np.abs(np.dot(hp2.T, np.dot(self.F, hp1)))     
                if err < sheld:
                    leftpoints.append(p1)
                    rightpoints.append(p2)            

            self.match_pts1 = np.array(leftpoints)
            self.match_pts2 = np.array(rightpoints)
            print('After screening, points length is {:d}'.format(len(self.match_pts1)))
        if point_lens != -1 and point_lens <= len(self.match_pts1):
            self.match_pts1 = self.match_pts1[:point_lens]
            self.match_pts2 = self.match_pts2[:point_lens]
            print("len=",self.match_pts1.shape)
        
        if self.match_pts1.shape[0] < point_lens:
            return False

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
        """
        
        try: 
            self.match_pts1.all()
        except AttributeError:
            print('Exact matching points')
            self.ExactGoodMatch()

        if method == "RANSAC":
            limit_length = len(self.match_pts1)
            print('Use RANSAC with %d points' %limit_length)
            self.FE, self.Fmask = cv2.findFundamentalMat(self.match_pts1[:limit_length],
                                                    self.match_pts2[:limit_length],
                                                    cv2.FM_RANSAC)
        elif method == "LMedS":
            limit_length = len(self.match_pts1)
            print('Use LMEDS with %d points' %len(self.match_pts1))
            self.FE, self.Fmask = cv2.findFundamentalMat(self.match_pts1[:limit_length],
                                                    self.match_pts2[:limit_length],
                                                    cv2.FM_LMEDS)
        elif method == "8Points":
            print('Use 8 Points algorithm')
            i = -1
            while True:
                # i = np.random.randint(0,len(self.match_pts1)-7)
                i += 1
                self.FE, self.Fmask = cv2.findFundamentalMat(self.match_pts1[i:i+8],
                                                    self.match_pts2[i:i+8],
                                                    cv2.FM_8POINT, 0.1, 0.99)
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
            return
        # print(self.FE)
        F_abs = abs(self.FE)
        # self.shape = np.array([512, 1392])
        self.FE = self.FE / F_abs.max()
        # self.get_normalized_F(self.FE, mean=[0,0], std=[np.sqrt(2.), np.sqrt(2.)], size=self.shape)
        return

    def DL_F_Es(self):
        """Use DL method to Estimate fundamental matrix

        """
        pass

    def EpipolarConstraint(self,F,pts1,pts2):
        '''Epipolar Constraint
            calculate the epipolar constraint 
            x^T*F*x
            :output 
                err_permatch
        '''
       
        print('Use ',len(pts1),' points to calculate epipolar constraints.')
        assert len(pts1) == len(pts2)
        err = 0.0
        for p1, p2 in zip(pts1, pts2):
            hp1, hp2 = np.ones((3,1)), np.ones((3,1))
            hp1[:2,0], hp2[:2,0] = p1, p2
            err += np.abs(np.dot(hp2.T, np.dot(F, hp1)))
        
        return err / float(len(pts1))
        
    
    def SymEpiDis(self, F, pts1, pts2):
        """Symetric Epipolar distance
            calculate the Symetric Epipolar distance
            To those epipolar lines going through points, calculate the included angle between it and GT line. (cos_all)
        """
        try:
            self.F.all()
        except AttributeError:
            print("There is no F_GT, Use LoadFMGT_KITTI() to get it.")
            self.LoadFMGT_KITTI()

        epsilon = 1e-5
        assert len(pts1) == len(pts2)
        print('Use ',len(pts1),' points to calculate epipolar distance.')
        err = 0.
        sheld = 0.1
        max_dis = 0.
        min_dis = np.Infinity
        cos_all = 0. 
        inliers = 0. 
        for p1, p2 in zip(pts1, pts2):
            hp1, hp2 = np.ones((3,1)), np.ones((3,1))
            hp1[:2,0], hp2[:2,0] = p1, p2
            fp, fq = np.dot(F, hp1), np.dot(F.T, hp2)
            # print(fp[0],'\n',fq[0])
            sym_jjt = 1./(fp[0]**2 + fp[1]**2 + epsilon) + 1./(fq[0]**2 + fq[1]**2 + epsilon)
            dis = ((np.dot(hp2.T, np.dot(F, hp1))**2) * (sym_jjt + epsilon))
            if dis < sheld:
                # if inliers calculate the angle
                inliers+=1
                f_GT_p, f_GT_q = np.dot(self.F, hp1), np.dot(self.F, hp2)
                
                # print(f_GT_p,f_GT_q)
                cos1 = (abs(fp[0]*f_GT_p[0]+fp[1]*f_GT_p[1]))/(sqrt(f_GT_p[0]**2+f_GT_p[1]**2+epsilon)*sqrt(fp[0]**2+fp[1]**2+epsilon))
                cos2 = (abs(fq[0]*f_GT_q[0]+fq[1]*f_GT_q[1]))/(sqrt(f_GT_q[0]**2+f_GT_q[1]**2+epsilon)*sqrt(fq[0]**2+fq[1]**2+epsilon))
                cos_all = (cos1 + cos2)/2 + cos_all
            max_dis = max(max_dis, dis)
            min_dis = min(min_dis, dis)
            err = err + dis
        if inliers == 0:
            angle = -1
        else:
            angle = (acos(cos_all/inliers)/3.1415926)*180
        print('Average angle: ',angle)

        return err / float(len(pts1)), max_dis, min_dis, angle


    def get_F_score(self, FE, pts1, pts2):
        """Get the F-Score
            Definition: the percentage of inliers points accroding to FE in GT matching pts 
            inlier err: sym_dis < 0.01 
        """
        assert len(pts1) == len(pts2)
        print('Use ',len(pts1),' points to calculate F-score.')
        inliers = 0
        epsilon = 1e-5
        for p1, p2 in zip(pts1, pts2):
            hp1, hp2 = np.ones((3,1)), np.ones((3,1))
            hp1[:2,0], hp2[:2,0] = p1, p2
            fp, fq = np.dot(FE, hp1), np.dot(FE.T, hp2)
            sym_jjt = 1./(fp[0]**2 + fp[1]**2 + epsilon) + 1./(fq[0]**2 + fq[1]**2 + epsilon)
            err = ((np.dot(hp2.T, np.dot(FE, hp1))**2) * (sym_jjt + epsilon))
            if err < 0.01:
                inliers += 1

        return inliers/len(pts1)

    def FMEvaluate(self):
        """Evaluate the fundamental matrix
            :output:
                print the metrics
                save as txt file
        """
        file_name = os.path.join(self.SavePath,self.SavePrefix+"F_evaluate.txt")

        epi_cons = self.EpipolarConstraint(self.FE, self.match_pts1, self.match_pts2)
        sym_epi_dis, max_dis, min_dis, angle = self.SymEpiDis(self.FE, self.match_pts1, self.match_pts2)
        L1_loss = np.sum(np.abs(self.F - self.FE))/9
        L2_loss = np.sum(np.power((self.F - self.FE),2))/9

        print("Evaluate the estimated fundamental matrix")
        print("The L1 loss is: {:4f}".format(L1_loss),"\nThe L2 loss is: {:4f}".format(L2_loss))
        print("The quantities of matching points is %d" %len(self.match_pts2))
        print("The epipolar constraint is : " ,float(epi_cons) ,"\nThe symmetry epipolar distance is: " ,float(sym_epi_dis))

        with open(file_name,'w') as f:
            f.writelines("Evaluate the estimated fundamental matrix: "+str(self.SavePrefix)+"\n")
            f.writelines("The L1 loss is: {:4f}".format(L1_loss)+"\nThe L2 loss is: {:4f}".format(L2_loss))
            f.writelines("\nThe quantities of matching points is " +str(len(self.match_pts2))+"\n")
            f.writelines("The epipolar constraint is : " +str(float(epi_cons))+"\nThe symmetry epipolar distance is: " +str(float(sym_epi_dis)))


    def FMEvaluate_aggregate(self):
        """Evaluate the fundamental matrix
            :output
                Metric Dictionary
        """
        # metric_dict = {}
        epi_cons = self.EpipolarConstraint(self.FE, self.match_pts1, self.match_pts2)
        sym_epi_dis, max_dis, min_dis, angle = self.SymEpiDis(self.FE, self.match_pts1, self.match_pts2)
        L1_loss = np.sum(np.abs(self.F - self.FE))/9
        L2_loss = np.sum(np.power((self.F - self.FE),2))/9
        F_score = self.get_F_score(self.FE, self.match_pts1, self.match_pts2)

        metric_dict = {
            'epi_cons' : epi_cons,
            'sym_epi_dis' : sym_epi_dis,
            'max_dis' : max_dis,
            'min_dis' : min_dis,
            'L1_loss' : L1_loss,
            'L2_loss' : L2_loss,
            'F_score' : F_score,
            'angle' : angle
        }
        return metric_dict

    def DrawEpipolarLines(self,i):
        """For F Estimation visulization, drawing the epipolar lines
            1. find epipolar lines
            2. draw lines
            :para
                i : index
        """
        try:
            self.FE.all()
        except AttributeError:
            self.EstimateFM() # use RANSAC as default
        try:
            self.match_pts1.all()
        except AttributeError:
            self.ExactGoodMatch(screening=True,point_lens=18)

        # Find epilines corresponding to points in right image (second image)
        # and drawing its lines on left image
        pts2re = self.match_pts2[:20].reshape(-1, 1, 2)
        lines1 = cv2.computeCorrespondEpilines(pts2re, 2, self.FE)
        lines1 = lines1.reshape(-1, 3)
        img3, img4 = self._draw_epipolar_lines_helper(self.imgl, self.imgr,
                                                      lines1, self.match_pts1,
                                                      self.match_pts2)
 
        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        pts1re = self.match_pts1[:20].reshape(-1, 1, 2)
        lines2 = cv2.computeCorrespondEpilines(pts1re, 1, self.FE)
        lines2 = lines2.reshape(-1, 3)
        img1, img2 = self._draw_epipolar_lines_helper(self.imgr, self.imgl,
                                                      lines2, self.match_pts2,
                                                      self.match_pts1)

        cv2.imwrite(os.path.join(self.SavePath,self.SavePrefix+'_'+str(i)+"_epipolarleft.jpg"),img1)
        # print("Saved in ",os.path.join(self.SavePath,"epipolarleft.jpg"))
        cv2.imwrite(os.path.join(self.SavePath,self.SavePrefix+'_'+str(i)+"_epipolarright.jpg"),img3)

        # cv2.startWindowThread()
        # cv2.imshow("left", img1)
        # cv2.imshow("right", img3)
        # k = cv2.waitKey()
        # if k == 27:
        #     cv2.destroyAllWindows()
       

    def _draw_epipolar_lines_helper(self, img1, img2, lines, pts1, pts2):
        """Helper method to draw epipolar lines and features """
        if img1.shape[2] == 1:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if img2.shape[2] == 1:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
 
        c = img1.shape[1]
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2]/r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0]*c) / r[1]])
            cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            cv2.circle(img1, tuple(pt1), 5, color, -1)
            cv2.circle(img2, tuple(pt2), 5, color, -1)
        return img1, img2



    def LoadE(self,EPath):
        """
        Load the Essentric Matrix got by OANet
        calculate the F by 
            F = Kl_inv^T * E * Kr_inv
        """
        self.E = np.load(EPath).reshape(3,3)
        print('Load E from %s as follow' %EPath)
        print(self.E)
        self.FE = self.E
        # e_gt = np.matmul(np.matmul(np.linalg.inv(K2).T, e_gt), np.linalg.inv(K1))
        # e_gt_unnorm = np.matmul(np.matmul(np.linalg.inv(T2).T, e_gt), np.linalg.inv(T1))
        # e_gt = e_gt_unnorm / np.linalg.norm(e_gt_unnorm)
        # self.FE = np.matmul(np.matmul(self.Kr_inv.T,self.E),self.Kl_inv)
        self.FE /= self.FE[2,2] 
        # print(self.FE)
        # FE = np.matmul(np.matmul(np.linalg.inv(self.Tr).T,self.FE),np.linalg.inv(self.Tl))
        # self.FE = FE / np.linalg.norm(FE)

        
    def _Get_Essential_Matrix(self):
        """Get Essential Matrix from Fundamental Matrix
            E = Kl^T*F*Kr
        """
        self.E = self.Kl.T.dot(self.FE).dot(self.Kr)
    
    def _Get_R_T(self,KITTI = False):
        """Get the [R|T] camera matrix
            After geting the R,T, need to determine whether the points are in front of the images
            :para KITTI : if True means no need to use Fmask to screen cuz points are screened by F_GT
        """
        # decompose essential matrix into R, t (See Hartley and Zisserman 9.13)
        U, S, Vt = np.linalg.svd(self.E)
        W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                      1.0]).reshape(3, 3)
 
        # iterate over all point correspondences used in the estimation of the
        # fundamental matrix
        first_inliers = []
        second_inliers = []
        if not KITTI:
            for i in range(len(self.Fmask)):
                if self.Fmask[i]:
                    # normalize and homogenize the image coordinates
                    first_inliers.append(self.Kl_inv.dot([self.match_pts1[i][0],
                                        self.match_pts1[i][1], 1.0]))
                    second_inliers.append(self.Kr_inv.dot([self.match_pts2[i][0],
                                        self.match_pts2[i][1], 1.0]))
        else:
            for i in range(self.match_pts1.shape[0]):
                first_inliers.append(self.Kl_inv.dot([self.match_pts1[i][0],
                                        self.match_pts1[i][1], 1.0]))
                second_inliers.append(self.Kr_inv.dot([self.match_pts2[i][0],
                                        self.match_pts2[i][1], 1.0]))


        # Determine the correct choice of second camera matrix
        # only in one of the four configurations will all the points be in
        # front of both cameras
        # First choice: R = U * Wt * Vt, T = +u_3 (See Hartley Zisserman 9.19)
        # R = U.dot(W.T).dot(Vt)
        # T = - U[:, 2]
        R = U.dot(W).dot(Vt)
        T = U[:, 2]
        if not self._in_front_of_both_cameras(first_inliers, second_inliers,
                                              R, T):
            # Second choice: R = U * W * Vt, T = -u_3
            T = - U[:, 2]
 
        if not self._in_front_of_both_cameras(first_inliers, second_inliers,
                                              R, T):
            # Third choice: R = U * Wt * Vt, T = u_3
            R = U.dot(W.T).dot(Vt)
            T = U[:, 2]
 
            if not self._in_front_of_both_cameras(first_inliers,
                                                  second_inliers, R, T):
                # Fourth choice: R = U * Wt * Vt, T = -u_3
                T = - U[:, 2]
 
        self.match_inliers1 = first_inliers
        self.match_inliers2 = second_inliers
        self.Rt1 = np.hstack((np.eye(3), np.zeros((3, 1)))) # as defaluted RT 
        self.Rt2 = np.hstack((R, T.reshape(3, 1)))



    def _in_front_of_both_cameras(self, first_points, second_points, rot,
                                  trans):
        """Determines whether point correspondences are in front of both images
        """
        rot_inv = rot
        for first, second in zip(first_points, second_points):
            first_z = np.dot(rot[0, :] - second[0]*rot[2, :],
                             trans) / np.dot(rot[0, :] - second[0]*rot[2, :],
                                             second)
            first_3d_point = np.array([first[0] * first_z,
                                       second[0] * first_z, first_z])
            second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T,
                                                                     trans)
 
            if first_3d_point[2] < 0 or second_3d_point[2] < 0:
                return False
 
        return True

    def RectifyImg(self,KITTI = False,Calib = False):
        """Rectify images using cv2.stereoRectify()
            :para KITTI: for _Get_R_T
            :para Calib: If true means rectify accroding to the calibration file

        """
        try:
            self.FE.all()
        except AttributeError:
            self.EstimateFM() # Using traditional method as default

        try:
            self.E.all()
        except AttributeError:
            self._Get_Essential_Matrix() 

        if not Calib:
            self._Get_R_T(KITTI)

            R = self.Rt2[:, :3]
            T = self.Rt2[:, 3]
            #perform the rectification
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.Kl, self.dl,
                                                            self.Kr, self.dr,
                                                            (self.imgl.shape[1],self.imgl.shape[0]),
                                                            R, T, alpha=0)
            mapx1, mapy1 = cv2.initUndistortRectifyMap(self.Kl, self.dl, R1, P1,
                                                    (self.imgl.shape[1],self.imgl.shape[0]),
                                                    cv2.CV_32FC1)
            mapx2, mapy2 = cv2.initUndistortRectifyMap(self.Kr, self.dr, R2, P2,
                                                    (self.imgr.shape[1],self.imgr.shape[0]),
                                                    cv2.CV_32FC1)
        else:
            R1 = self.Rl
            P1 = self.Pl 
            R2 = self.Rr 
            P2 = self.Pr 

            mapx1, mapy1 = cv2.initUndistortRectifyMap(self.Kl, self.dl, R1, P1,
                                                    (self.imgl.shape[1],self.imgl.shape[0]),
                                                    cv2.CV_32FC1)
            mapx2, mapy2 = cv2.initUndistortRectifyMap(self.Kr, self.dr, R2, P2,
                                                    (self.imgr.shape[1],self.imgr.shape[0]),
                                                    cv2.CV_32FC1)
        img_rect1 = cv2.remap(self.imgl, mapx1, mapy1, cv2.INTER_LINEAR)
        img_rect2 = cv2.remap(self.imgr, mapx2, mapy2, cv2.INTER_LINEAR)

        # save
        cv2.imwrite(os.path.join(self.SavePath,self.SavePrefix+"RectedLeft.jpg"),img_rect1)
        cv2.imwrite(os.path.join(self.SavePath,self.SavePrefix+"RectedRight.jpg"),img_rect2)

        
        # draw the images side by side
        total_size = (max(img_rect1.shape[0], img_rect2.shape[0]),
                      img_rect1.shape[1] + img_rect2.shape[1], 3)
        img = np.zeros(total_size, dtype=np.uint8)
        img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
        img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2
 
        # draw horizontal lines every 25 px accross the side by side image
        for i in range(20, img.shape[0], 25):
            cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0))
 
        cv2.imshow('imgRectified', img)
        print(img_rect2.shape)
        
        k = cv2.waitKey()
        if k == 27:
            cv2.destroyAllWindows()
        


    def RectifyImgUncalibrated(self):
        """Rectify imgs without the parameters
        """
        im1, im2 = self.imgl, self.imgr
        size=im1.shape[1],im1.shape[0]
        points1, points2 = self.match_pts1, self.match_pts2
        H1,H2=returnH1_H2(points1,points2,self.FE,size)
        rectimg1 = cv2.warpPerspective(im1,H1,size)
        rectimg2 = cv2.warpPerspective(im2,H2,size)
        # rectifyim1,rectifyim2=getRectifystereo(H1,H2,im1,im2,size,self.FE)
        
        # cv2.imwrite(os.path.join(self.SavePath,self.SavePrefix+"RectUncalibLeft.jpg"),rectimg1)
        # cv2.imwrite(os.path.join(self.SavePath,self.SavePrefix+"RectUncalibRight.jpg"),rectimg2)

        cv2.startWindowThread()
        # cv2.imshow("left", rectimg1)
        # cv2.imshow("right", rectimg2)

        # draw the images side by side
        total_size = (max(rectimg1.shape[0], rectimg2.shape[0]),
                      rectimg1.shape[1] + rectimg2.shape[1], 3)
        img = np.zeros(total_size, dtype=np.uint8)
        img[:rectimg1.shape[0], :rectimg1.shape[1]] = rectimg1
        img[:rectimg2.shape[0], rectimg1.shape[1]:] = rectimg2
 
        # draw horizontal lines every 25 px accross the side by side image
        for i in range(20, img.shape[0], 25):
            cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0))
 
        cv2.imwrite(os.path.join(self.SavePath,self.SavePrefix+"RectUncalib.jpg"),img)
        
        #show
        # cv2.imshow('imgRectified', img)
        
   
        # k = cv2.waitKey()
        # if k == 27:
        #     cv2.destroyAllWindows()
        

    
if __name__ == "__main__":

    

    
    #DataLoad---------------------
        #--- YFCC100M
    # prefix = 'big_ben_test/'
    # img_nameL = "93734988_13580074803.jpg"
    # img_nameR = "92689035_2333683024.jpg"
    # EPath = "/Users/zhangyesheng/Documents/GitHub/OANet/Rectify/ModelRes/"+prefix+"E.npy"
    # leftcorr = "/Users/zhangyesheng/Documents/GitHub/OANet/Rectify/ModelRes/"+prefix+"leftcorr.npy"
    # rightcorr = "/Users/zhangyesheng/Documents/GitHub/OANet/Rectify/ModelRes/"+prefix+"rightcorr.npy"
    # SavePrefix = '_Model_'

        #--- KITTI
    prefix = 'KITTI_rected/'
    img_nameL = "left.png"
    img_nameR = "right.png"
    EPath = "/Users/zhangyesheng/Documents/GitHub/OANet/Rectify/ModelRes/"+prefix+"E.npy"
    leftcorr = "/Users/zhangyesheng/Documents/GitHub/OANet/Rectify/ModelRes/"+prefix+"leftcorr.npy"
    rightcorr = "/Users/zhangyesheng/Documents/GitHub/OANet/Rectify/ModelRes/"+prefix+"rightcorr.npy"
    SavePrefix = '_OANet_'

    #     #--- indoor
    # prefix = 'indoors/'
    # img_nameL = 'left.jpg'
    # img_nameR = 'right.jpg'
    # EPath = "/Users/zhangyesheng/Documents/GitHub/OANet/Rectify/ModelRes/RectTest/E.npy"
    # leftcorr = "/Users/zhangyesheng/Documents/GitHub/OANet/Rectify/ModelRes/RectTest/leftcorr.npy"
    # rightcorr = "/Users/zhangyesheng/Documents/GitHub/OANet/Rectify/ModelRes/RectTest/rightcorr.npy"
    # SavePrefix = '_indoors_'
    
    #Mac
    ImgPath = "/Users/zhangyesheng/Documents/GitHub/OANet/Rectify/pics/"+prefix
    ParaPath = "/Users/zhangyesheng/Documents/GitHub/OANet/Rectify/calibration/"+prefix
    SavePath = "/Users/zhangyesheng/Documents/GitHub/OANet/Rectify/Res/"+prefix

    #Linux
    # ImgPath = "./pics/
    # ParaPath = "./calibration/"
    # SavePath = "./Res/"
    # img_nameL = "40982537_3102972880.jpg"
    # img_nameR = "42002003_1635942632.jpg"

    test = SelfCalibration(ImgPath,ParaPath,SavePath,SavePrefix)
    
    test.load_image_pair(img_nameL, img_nameR)
    
    

    #-------------Tradition
    # test.LoadPara_KITTI()
    # test.Show()
    # test.FE = test.LoadFMGT_KITTI()
    # print(test.FE)
    # test.ExactGoodMatch(screening=True,point_lens=18)
    test.ExactGoodMatch(screening=False)
    test.EstimateFM(method="RANSAC")
    # test.Show()
    # test.DrawEpipolarLines()


    #--------------DL
    # test.LoadPara_YFCC()
    # test.LoadPara_KITTI()
    # test.LoadE(EPath)
    # test.LoadCorr(rightcorr,leftcorr)
    
    # test.EstimateFM(method="8Points")
    # test.ExactGoodMatch(screening=True,point_lens=18)
    # test.FMEvaluate()

    # test.RectifyImg(Calib=True)
    test.RectifyImgUncalibrated()

