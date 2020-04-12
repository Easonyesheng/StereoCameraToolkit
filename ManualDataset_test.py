"""Evaluate on Manual Dataset"""
# threshold: sym_epi_dis < 0.01
from SelfCalibration import SelfCalibration
import cv2
import numpy as np 
import os

DatasetPath = '/Users/zhangyesheng/Desktop/Research/GraduationDesign/StereoVision/StereoCamera/ManualDataset'
ParaPath = '/Users/zhangyesheng/Desktop/Research/GraduationDesign/StereoVision/StereoCamera/ManualDataset/Para'
SavePath = '/Users/zhangyesheng/Desktop/Research/GraduationDesign/StereoVision/StereoCamera/ManualDataset/Res'
FPath = '/Users/zhangyesheng/Desktop/Research/GraduationDesign/Res/+PointNet/point_net_2/0.txt'
SavePrefix = 'GT' 

epi_cons = 0.
sym_epi_dis = 0.
L1_loss = 0.
L2_loss = 0.
max_dis = 0.
min_dis = 0. 
F_score = 0. 
angle = 0. 

Index = 600
nums = 2
mark = 0
heng = '*'*5

for i in range(Index-nums,Index):
    print(heng,str(mark),heng)
    eva = SelfCalibration(DatasetPath,ParaPath,SavePath,SavePrefix)
    eva.load_img_test(i)
    if SavePrefix in ['RANSAC','LMedS','8Points']:
        eva.EstimateFM(method=SavePrefix)
    elif SavePrefix == 'GT':
        eva.FE = eva.LoadFMGT_KITTI()
    else:
        eva.Load_F_test(FPath)
    # print('FE',eva.FE)
    eva.LoadFMGT_KITTI()
    # print('F_GT',eva.F)   

    flag = eva.ExactGoodMatch(screening=True,point_lens=40)

    if not flag: # if screening points length < point_lens : continue
        Index += 1

        continue
    # draw epipolar lines
    eva.DrawEpipolarLines(i) 

    '''Uncomment this part to evaluate'''
#     # evaluate
#     dic = eva.FMEvaluate_aggregate()

#     epi_cons += dic['epi_cons']
#     sym_epi_dis += dic['sym_epi_dis']
#     F_score += dic['F_score']
#     max_dis += dic['max_dis']
#     min_dis += dic['min_dis']
#     L1_loss += dic['L1_loss']
#     L2_loss += dic['L2_loss']
#     if dic['angle'] > 0:
#         angle += dic['angle']
#     mark += 1

# epi_cons /= nums
# sym_epi_dis /= nums 
# F_score /= nums
# F_score *= 100
# max_dis /= nums
# min_dis /= nums
# L1_loss /= nums
# L2_loss /= nums
# angle /= nums
# # print(angle)

# # Save evaluate file as txt
# file_name = os.path.join(SavePath,SavePrefix+"_F_evaluate.txt")
# with open(file_name,'w') as f:
#     f.writelines("Evaluate the estimated fundamental matrix by "+str(SavePrefix)+" with {:d} images\n".format(nums))
#     f.writelines("The L1 loss is: {:4f}".format(L1_loss)+"\nThe L2 loss is: {:4f}\n".format(L2_loss))
#     f.writelines("The epipolar constraint is : " +str(float(epi_cons))+"\nThe symmetry epipolar distance is: " +str(float(sym_epi_dis)))
#     f.writelines('\nThe max symmetry epipolar distance is: '+str(max_dis)+'\nThe min symmetry epipolar distance is: '+str(min_dis))
#     f.writelines('\nThe F-score is: {:4f}%'.format(F_score))
#     f.writelines('\nThe inliers angle cosine is: {:4f} degrees '.format(angle))