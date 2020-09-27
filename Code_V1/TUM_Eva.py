"""
Evaluate on TUM Dataset
"""

import os
from SelfCalibration import SelfCalibration
import cv2
import numpy as np 

ImgPath = r'D:\F_Estimation_private\RealWorldPred'
ParaPath = r''
F_gt_path = r'D:\F_Estimation_private\RealWorldPred\stereo.txt' #+ F_gti.txt
F_est_path = r'D:\F_Estimation_private\RealWorldPred\RealWorld' #+ i.txt
SavePath = r'D:\F_Estimation_private\RealWorldPred\Res'

SavePrefix = 'Real+'
# SavePrefix = 'TUM_0_GT'
# SavePrefix = 'TUM_0_RANSAC'
# SavePrefix = 'TUM_0_8Point'
# SavePrefix = 'TUM_0_LMedS'

left_name = 'left'
right_name = 'right'

epi_cons = 0.
sym_epi_dis = 0.
L1_loss = 0.
L2_loss = 0.
max_dis = 0.
min_dis = 0. 
F_score = 0. 
angle = 0. 
nums = 0
time = 0.

for i in range(1):
# i = 3
    Eva = SelfCalibration(ImgPath, ParaPath, SavePath, SavePrefix)

    left_img = os.path.join(ImgPath,left_name+'\\'+'left'+str(i).zfill(2)+'.jpg')
    right_img = os.path.join(ImgPath,right_name+'\\'+'right'+str(i).zfill(2)+'.jpg')

    Eva.load_image_pair(left_img,right_img)

    # load F_gt
    Eva.F = Eva.Load_F_test(F_gt_path)
    # print(Eva.F)

    # load F_est model
    Eva.Load_F_test(os.path.join(F_est_path,str(i)+'.txt'))

    # # Estimate the F
    # time += Eva.EstimateFM("LMedS")

    Eva.ExactGoodMatch(screening=True)

    # dic = Eva.FMEvaluate_aggregate()

    # epi_cons += dic['epi_cons']
    # sym_epi_dis += dic['sym_epi_dis']
    # F_score += dic['F_score']
    # max_dis += dic['max_dis']
    # min_dis += dic['min_dis']
    # L1_loss += dic['L1_loss']
    # L2_loss += dic['L2_loss']
    # # epi_error += dic['epi_error']
    # if dic['angle'] > 0:
    #     angle += dic['angle']
    # nums += 1


    Eva.DrawEpipolarLines(i)
    
# epi_cons /= nums
# sym_epi_dis /= nums 
# F_score /= nums
# F_score *= 100
# max_dis /= nums
# min_dis /= nums
# L1_loss /= nums
# L2_loss /= nums
# angle /= nums
# time /= nums

# file_name = os.path.join(SavePath,SavePrefix+"_F_evaluate.txt")
# with open(file_name,'w') as f:
#     f.writelines("Evaluate the estimated fundamental matrix by "+str(SavePrefix)+" with {:d} images\n".format(nums))
#     f.writelines("The L1 loss is: {:4f}".format(L1_loss)+"\nThe L2 loss is: {:4f}\n".format(L2_loss))
#     f.writelines("The epipolar constraint is : " +str(float(epi_cons))+"\nThe symmetry epipolar distance is: " +str(float(sym_epi_dis)))
#     f.writelines('\nThe max symmetry epipolar distance is: '+str(max_dis)+'\nThe min symmetry epipolar distance is: '+str(min_dis))
#     f.writelines('\nThe F-score is: {:4f}%'.format(F_score))
#     f.writelines('\nThe inliers angle cosine is: {:4f} degrees '.format(angle))
#     f.writelines('\nThe processing time is {:4f}'.format(time))
#     # f.writelines('\nThe epipolar ralative error is {:2f}'.format(epi_error))
    


