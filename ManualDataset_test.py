"""Evaluate on Manual Dataset"""

from SelfCalibration import SelfCalibration
import cv2
import numpy as np 
import os

DatasetPath = '/Users/zhangyesheng/Desktop/Research/GraduationDesign/StereoVision/StereoCamera/ManualDataset'
ParaPath = '/Users/zhangyesheng/Desktop/Research/GraduationDesign/StereoVision/StereoCamera/ManualDataset/Para'
SavePath = '/Users/zhangyesheng/Desktop/Research/GraduationDesign/StereoVision/StereoCamera/ManualDataset/Res'
SavePrefix = 'RANSAC'

epi_cons = 0.
sym_epi_dis = 0.
L1_loss = 0.
L2_loss = 0.

for i in range(200):
    eva = SelfCalibration(DatasetPath,ParaPath,SavePath,'-')
    eva.load_img_test(i)

    eva.EstimateFM(method=SavePrefix)

    eva.LoadFMGT_KITTI()

    eva.ExactGoodMatch(screening=True,point_lens=18)

    dic = eva.FMEvaluate_aggregate()
    epi_cons += dic['epi_cons']
    sym_epi_dis += dic['sym_epi_dis']
    L1_loss += dic['L1_loss']
    L2_loss += dic['L2_loss']

epi_cons /= 200.
sym_epi_dis /= 200. 
L1_loss /= 200. 
L2_loss /= 200. 

file_name = os.path.join(SavePath,SavePrefix+"F_evaluate.txt")
with open(file_name,'w') as f:
    f.writelines("Evaluate the estimated fundamental matrix by "+str(SavePrefix)+" with 200 images\n")
    f.writelines("The L1 loss is: {:4f}".format(L1_loss)+"\nThe L2 loss is: {:4f}\n".format(L2_loss))
    f.writelines("The epipolar constraint is : " +str(float(epi_cons))+"\nThe symmetry epipolar distance is: " +str(float(sym_epi_dis)))
    