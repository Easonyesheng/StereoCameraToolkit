# -*- coding: utf-8 -*-
'''cv2.stereoRectifyUncalibrated'''
import cv2
import numpy as np
# from numba import autojit
# @autojit
def mathching(im1,im2):
    surf=cv2.xfeatures2d.SURF_create()
    kp1,des1=surf.detectAndCompute(im1,None)
    kp2,des2=surf.detectAndCompute(im2,None)
    FLANN_INDEX_KDTREE=0
    index_p=dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    searth_p=dict(checks=50)
    flann=cv2.FlannBasedMatcher(index_p,searth_p)
    matches=flann.knnMatch(des1,des2,k=2)
    good =[]
    pts1=[]
    pts2=[]
    for i,(m,n) in enumerate(matches):
        if m.distance<0.6*n.distance:
            good.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
    pts1=np.float32(pts1)
    pts2=np.float32(pts2)
    F,mask=cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
    points1=pts1[mask.ravel()==1]
    points2=pts2[mask.ravel()==1]
    return points1,points2,F

def returnH1_H2(points1,points2,F,size):
    p1=points1.reshape(len(points1)*2,1)#stackoverflow上需要将(m,2)的点变为(m*2,1),因为不变在c++中会产生内存溢出
    p2=points2.reshape(len(points2)*2,1)
    _,H1,H2=cv2.stereoRectifyUncalibrated(p1,p2,F,size) #size是宽，高
    return H1,H2
 
def getRectifystereo(H1,H2,im1,im2,size,F):
    #左极点
    e_l=compute_epipole(F.T)
    #右极点
    e_r=compute_epipole(F)
    w_l,h_l=im1.shape[1],im1.shape[0]
    w_r,h_r=im2.shape[1],im2.shape[0]
    if abs(e_l[0])>w_l and abs(e_l[1])>h_l and abs(e_r[0])>w_r and abs(e_r[1])>h_r:#判断极点在图像外部
        K=np.eye(3,3)
        # left_R=np.linalg.inv(K)@H1@K
        left_R = np.dot(np.linalg.inv(K),np.dot(H1,K))
        # right_R=np.linalg.inv(K)@H2@K
        right_R = np.dot(np.linalg.inv(K),np.dot(H2,K))

        d=np.zeros((5,1))
        map1,map2=cv2.initUndistortRectifyMap(K,d,left_R,K,size,cv2.CV_16SC2)
        map3,map4=cv2.initUndistortRectifyMap(K,d,right_R,K,size,cv2.CV_16SC2)
        rectifyim1=cv2.remap(im1,map1,map2,cv2.INTER_LINEAR)
        rectifyim2=cv2.remap(im2,map3,map4,cv2.INTER_LINEAR)

    else:
        print("极点在图像内部")
        return None,None
    return rectifyim1,rectifyim2

def compute_epipole(F):
    """
    利用F计算右极点，若要计算左极点，则分解F.T
    """
    U, S, V = np.linalg.svd(F)
    e = V[-1]
    return e / e[2]

if __name__=="__main__":
    path = '/Users/zhangyesheng/Desktop/GraduationDesign/StereoVision/self-calibration/Project/Algorithm_self_Calibration/pic/'
    im1=cv2.imread(path+'left.jpg')
    im2=cv2.imread(path+'right.jpg')
    size=im1.shape[1],im1.shape[0]
    points1,points2,F=mathching(im1,im2)
    H1,H2=returnH1_H2(points1,points2,F,size)
    rectifyim1,rectifyim2=getRectifystereo(H1,H2,im1,im2,size,F)
    cv2.imwrite(path+"L_r.jpg",rectifyim1)
    cv2.imwrite(path+"R_r.jpg",rectifyim2)
