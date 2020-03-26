"""Evaluate on Manual Dataset"""

from SelfCalibration import SelfCalibration
import cv2
import numpy as np 
import os

DatasetPath = '/Users/zhangyesheng/Desktop/Research/GraduationDesign/StereoVision/StereoCamera/ManualDataset'
ParaPath = '/Users/zhangyesheng/Desktop/Research/GraduationDesign/StereoVision/StereoCamera/ManualDataset/Para'
SavePath = '/Users/zhangyesheng/Desktop/Research/GraduationDesign/StereoVision/StereoCamera/ManualDataset/Res'
SavePrefix = 'LMedS'

epi_cons = 0.
sym_epi_dis = 0.
L1_loss = 0.
L2_loss = 0.

Index = 1000
nums = 200
mark = 0
heng = '*'*5

for i in range(Index-nums,Index):
    print(heng,str(mark),heng)
    eva = SelfCalibration(DatasetPath,ParaPath,SavePath,'-')
    eva.load_img_test(i)

    eva.EstimateFM(method=SavePrefix)
    # print('FE',eva.FE)
    eva.LoadFMGT_KITTI()
    # print('F_GT',eva.F)   
    flag = eva.ExactGoodMatch(screening=True,point_lens=20)

    if not flag: # if SIFT has hard error: continue
        Index += 1

        continue

    dic = eva.FMEvaluate_aggregate()
    epi_cons += dic['epi_cons']
    sym_epi_dis += dic['sym_epi_dis']
    L1_loss += dic['L1_loss']
    L2_loss += dic['L2_loss']
    mark += 1

epi_cons /= nums
sym_epi_dis /= nums 
L1_loss /= nums
L2_loss /= nums

file_name = os.path.join(SavePath,SavePrefix+"_F_evaluate.txt")
with open(file_name,'w') as f:
    f.writelines("Evaluate the estimated fundamental matrix by "+str(SavePrefix)+" with {:d} images\n".format(nums))
    f.writelines("The L1 loss is: {:4f}".format(L1_loss)+"\nThe L2 loss is: {:4f}\n".format(L2_loss))
    f.writelines("The epipolar constraint is : " +str(float(epi_cons))+"\nThe symmetry epipolar distance is: " +str(float(sym_epi_dis)))
    