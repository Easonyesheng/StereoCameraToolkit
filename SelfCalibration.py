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

class SelfCalibration:
    """
    自标定系统
        功能：
            1.读入图片
            2.读入参数
            3.估计F
            4.F评估
            5.校正
            
    文件结构
        -图片文件夹
            -左图
            -右图
        -标定文件夹
            -左相机参数 -- K & d 
            -右相机参数
            -基础矩阵
        -保存文件夹
            -校正后左图
            -校正后右图
            -极线图
    """

    def __init__(self,ImgPath,ParaPath,SavePath):

        self.ImgPath = ImgPath
        self.ParaPath = ParaPath
        self.SavePath = SavePath
        self.F = None
        self.FE = None
        self.imgl = None
        self.imgr = None
        self.Kl = None
        self.Kr = None
        self.dr = None
        self.dl = None


    def load_image_pair(self, img_nameL, img_nameR):
            """Loads pair of images
    
                This method loads the two images for which the 3D scene should be
                reconstructed. The two images should show the same real-world scene
                from two different viewpoints.
    
                :param img_nameL: name of left image
                :param img_nameR: name of right image
                
            """
            self.imgl = cv2.imread(os.path.join(self.ImgPath,img_nameL), cv2.CV_8UC3)
            self.imgr = cv2.imread(os.path.join(self.ImgPath,img_nameR), cv2.CV_8UC3)
    
            # make sure images are valid
            if self.imgl is None:
                sys.exit("Image " +os.path.join(self.ImgPath,img_nameL) + " could not be loaded.")
            if self.imgr is None:
                sys.exit("Image " + os.path.join(self.ImgPath,img_nameR) + " could not be loaded.")
    
            if len(self.imgl.shape) == 2:
                self.imgl = cv2.cvtColor(self.imgl, cv2.COLOR_GRAY2BGR)
                self.imgr = cv2.cvtColor(self.imgr, cv2.COLOR_GRAY2BGR)
    
             
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

            
    def LoadPara(self):
        """Load the camera parameters

            Load the parameters and undistort the images

            :para Kl: left camera's 3x3 intrinsic camera matrix
            :para Kr: right camera's 3x3 intrinsic camera matrix
            :para dl: vector of distortion coefficients of left camera
            :para dr: vector of distortion coefficients of right camera

        """
        paser = KittiAnalyse(self.ImgPath,self.ParaPath,self.SavePath)
        calib = paser.calib
        Kl,Kr = calib['K_0{}'.format('0')],calib['K_0{}'.format('1')]
        dl,dr = calib['D_0{}'.format('0')],calib['D_0{}'.format('1')]

        self.Kl = Kl
        self.Kr = Kr
        self.Kl_inv = np.linalg.inv(Kl)  # store inverse for fast access
        self.Kr_inv = np.linalg.inv(Kr)  # store inverse for fast access
        self.dl = dl
        self.dr = dr

        # # undistort the images
        # self.imgl = cv2.undistort(self.imgl, self.Kl, self.dl)
        # self.imgr = cv2.undistort(self.imgr, self.Kr, self.dr)

    def LoadFMGT(self, F_filename = 'F.txt'):
        """Load the fundamental matrix file(.txt)

        """
        F = np.loadtxt(os.path.join(self.ParaPath,F_filename))
        w, h = F.shape
        if w*h != 9:
            print("Fundamental matrix file Error!")
            self.F = None
        if w == h:
            self.F = F
        else:
            self.F = F.reshape(3,3)


    def ExactGoodMatch(self,screening = False):
        """Get matching points & Use F_GT to get good matching points
            1.use SIFT to exact feature points 
            if screening
            2.calculate metrics use F_GT and screening good matches
            ! use it only you have the F_GT
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
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]
        self.match_pts1 = np.int32(pts1)
        self.match_pts2 = np.int32(pts2)


        # use F_GT to select good match
        if screening:
            try:
                self.F.all()
            except AttributeError:
                print("There is no F_GT, you can use LoadFMGT() to get it.\nWarning: Without screening good matches.")
                return

            leftpoints = []
            rightpoints = []

            for p1,p2 in zip(self.match_pts1, self.match_pts2):
                hp1, hp2 = np.ones((3,1)), np.ones((3,1))
                hp1[:2,0], hp2[:2,0] = p1, p2 
                err = np.abs(np.dot(hp2.T, np.dot(self.F, hp1)))       
                if err < 0.5:
                    leftpoints.append(p1)
                    rightpoints.append(p2)
            
            self.match_pts1 = np.array(leftpoints)
            self.match_pts2 = np.array(rightpoints)




    def EstimateFM(self,method="RANSAC"):
        """Estimate the fundamental matrix 
            :para method: which method you use
                1.RANSAC
                2.LMedS
                3.DL(Deep Learning)
            :output 
                change self.FE
        """
        try: 
            self.match_pts1.all()
        except AttributeError:
            self.ExactGoodMatch()

        if method == "RANSAC":
            self.FE, self.Fmask = cv2.findFundamentalMat(self.match_pts1,
                                                    self.match_pts2,
                                                    cv2.FM_RANSAC, 0.1, 0.99)
        elif method == "LMedS":
            self.FE, self.Fmask = cv2.findFundamentalMat(self.match_pts1,
                                                    self.match_pts2,
                                                    cv2.FM_LMEDS, 0.1, 0.99)
        elif method == "DL":
            # get the mask
            FE, self.Fmask = cv2.findFundamentalMat(self.match_pts1,
                                                    self.match_pts2,
                                                    cv2.FM_RANSAC, 0.1, 0.99)
            self.DL_F_Es() # need to be completed

        else:
            print("Method Error!")
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
        """
        epsilon = 1e-5
        assert len(pts1) == len(pts2)
        print('Use ',len(pts1),' points to calculate epipolar distance.')
        err = 0.
        for p1, p2 in zip(pts1, pts2):
            hp1, hp2 = np.ones((3,1)), np.ones((3,1))
            hp1[:2,0], hp2[:2,0] = p1, p2
            fp, fq = np.dot(F, hp1), np.dot(F.T, hp2)
            sym_jjt = 1./(fp[0]**2 + fp[1]**2 + epsilon) + 1./(fq[0]**2 + fq[1]**2 + epsilon)
            err = err + ((np.dot(hp2.T, np.dot(F, hp1))**2) * (sym_jjt + epsilon))

        return err / float(len(pts1))


    def FMEvaluate(self):
        """Evaluate the fundamental matrix
            :output:
                print the metrics
        """

        epi_cons = self.EpipolarConstraint(self.FE, self.match_pts1, self.match_pts2)
        sym_epi_dis = self.SymEpiDis(self.FE, self.match_pts1, self.match_pts2)
        print("Evaluate the estimated fundamental matrix")
        print("The quantities of matching points is %d" %len(self.match_pts2))
        print("The epipolar constraint is : %d" %epi_cons ,"\nThe symmetry epipolar distance is: %d" %sym_epi_dis)
        


    def DrawEpipolarLines(self):
        """For F Estimation visulization, drawing the epipolar lines
            1. find epipolar lines
            2. draw lines
        """
        try:
            self.FE.all()
        except AttributeError:
            self.EstimateFM() # use RANSAC as default
        try:
            self.match_pts1.all()
        except AttributeError:
            self.ExactGoodMatch()

        # Find epilines corresponding to points in right image (second image)
        # and drawing its lines on left image
        pts2re = self.match_pts2.reshape(-1, 1, 2)
        lines1 = cv2.computeCorrespondEpilines(pts2re, 2, self.FE)
        lines1 = lines1.reshape(-1, 3)
        img3, img4 = self._draw_epipolar_lines_helper(self.imgl, self.imgr,
                                                      lines1, self.match_pts1,
                                                      self.match_pts2)
 
        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        pts1re = self.match_pts1.reshape(-1, 1, 2)
        lines2 = cv2.computeCorrespondEpilines(pts1re, 1, self.FE)
        lines2 = lines2.reshape(-1, 3)
        img1, img2 = self._draw_epipolar_lines_helper(self.imgr, self.imgl,
                                                      lines2, self.match_pts2,
                                                      self.match_pts1)

        cv2.imwrite(os.path.join(self.SavePath,"epipolarleft.jpg"),img1)
        # print("Saved in ",os.path.join(self.SavePath,"epipolarleft.jpg"))
        cv2.imwrite(os.path.join(self.SavePath,"epipolarright.jpg"),img3)

        cv2.startWindowThread()
        cv2.imshow("left", img1)
        cv2.imshow("right", img3)
        k = cv2.waitKey()
        if k == 27:
            cv2.destroyAllWindows()
       

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



    def _Get_Essential_Matrix(self):
        """Get Essential Matrix from Fundamental Matrix
            E = Kl^T*F*Kr
        """
        self.E = self.Kl.T.dot(self.FE).dot(self.Kr)
    
    def _Get_R_T(self):
        """Get the [R|T] camera matrix
            After geting the R,T, need to determine whether the points are in front of the images
        """
        # decompose essential matrix into R, t (See Hartley and Zisserman 9.13)
        U, S, Vt = np.linalg.svd(self.E)
        W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                      1.0]).reshape(3, 3)
 
        # iterate over all point correspondences used in the estimation of the
        # fundamental matrix
        first_inliers = []
        second_inliers = []
        for i in range(len(self.Fmask)):
            if self.Fmask[i]:
                # normalize and homogenize the image coordinates
                first_inliers.append(self.Kl_inv.dot([self.match_pts1[i][0],
                                     self.match_pts1[i][1], 1.0]))
                second_inliers.append(self.Kr_inv.dot([self.match_pts2[i][0],
                                      self.match_pts2[i][1], 1.0]))
 
        # Determine the correct choice of second camera matrix
        # only in one of the four configurations will all the points be in
        # front of both cameras
        # First choice: R = U * Wt * Vt, T = +u_3 (See Hartley Zisserman 9.19)
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

    def RectifyImg(self):
        """Rectify images using cv2.stereoRectify()
        """
        try:
            self.FE.all()
        except AttributeError:
            self.EstimateFM() # Using traditional method as default

        self._Get_Essential_Matrix()
        self._Get_R_T()

        R = self.Rt2[:, :3]
        T = self.Rt2[:, 3]
        #perform the rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.Kl, self.dl,
                                                          self.Kr, self.dr,
                                                          self.imgl.shape[:2],
                                                          R, T, alpha=0)
        mapx1, mapy1 = cv2.initUndistortRectifyMap(self.Kl, self.dl, R1, P1,
                                                   [self.imgl.shape[0],self.imgl.shape[1]],
                                                   cv2.CV_32FC1)
        mapx2, mapy2 = cv2.initUndistortRectifyMap(self.Kr, self.dr, R2, P2,
                                                   [self.imgl.shape[0],self.imgl.shape[1]],
                                                   cv2.CV_32FC1)
        img_rect1 = cv2.remap(self.imgl, mapx1, mapy1, cv2.INTER_LINEAR)
        img_rect2 = cv2.remap(self.imgr, mapx2, mapy2, cv2.INTER_LINEAR)

        # save
        cv2.imwrite(os.path.join(self.SavePath,"RectedLeft.jpg"),img_rect1)
        cv2.imwrite(os.path.join(self.SavePath,"RectedRight.jpg"),img_rect2)


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
        
        cv2.imwrite(os.path.join(self.SavePath,"RectUncalibLeft.jpg"),rectimg1)
        cv2.imwrite(os.path.join(self.SavePath,"RectUncalibRight.jpg"),rectimg2)

        cv2.startWindowThread()
        cv2.imshow("left", rectimg1)
        cv2.imshow("right", rectimg2)
        k = cv2.waitKey()
        if k == 27:
            cv2.destroyAllWindows()

    
if __name__ == "__main__":

    ImgPath = "/Users/zhangyesheng/Desktop/GraduationDesign/StereoVision/StereoCamera/pics/"
    ParaPath = "/Users/zhangyesheng/Desktop/GraduationDesign/StereoVision/StereoCamera/calib_cam_to_cam.txt"
    SavePath = "/Users/zhangyesheng/Desktop/GraduationDesign/StereoVision/StereoCamera/Res/"
    img_nameL = "left.jpg"
    img_nameR = "right.jpg"

    test = SelfCalibration(ImgPath,ParaPath,SavePath)
    
    test.load_image_pair(img_nameL, img_nameR)
    
    test.LoadPara()
    # test.Show()

    test.EstimateFM(method="RANSAC")
    # test.Show()
    # test.DrawEpipolarLines()
    test.RectifyImgUncalibrated()

