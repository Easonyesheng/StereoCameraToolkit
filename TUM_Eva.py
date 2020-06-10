"""
Evaluate on TUM Dataset
"""

import os
from SelfCalibration import SelfCalibration
import cv2
import numpy as np 

ImgPath = r'D:\F_Estimation_private\deepF_noCorrs\data\Pred\Pred_TUM_06_02'
ParaPath = r''
F_gt_path = r'D:\F_Estimation_private\deepF_noCorrs\data\Pred\Pred_TUM_06_02' #+ F_gti.txt
F_est_path = r'D:\F_Estimation_private\deepF_noCorrs\Pred_Result\TUM' #+ i.txt
SavePath = r'D:\F_Estimation_private\deepF_noCorrs\Pred_Result\TUM\Res'
SavePrefix = 'TUM_0_model'
# SavePrefix = 'TUM_0_GT'
# SavePrefix = 'TUM_0_RANSAC'

i = 3

Eva = SelfCalibration(ImgPath, ParaPath, SavePath, SavePrefix)

left_name = 'Image00'
right_name = 'Image01'
left_img = os.path.join(ImgPath,left_name+'\\'+str(i).zfill(10)+'.jpg')
right_img = os.path.join(ImgPath,right_name+'\\'+str(i).zfill(10)+'.jpg')
Eva.load_image_pair(left_img,right_img)

# load F_gt
Eva.F = Eva.Load_F_test(os.path.join(F_gt_path,'F_gt'+str(i)+'.txt'))
# print(Eva.F)

# # load F_est model
Eva.Load_F_test(os.path.join(F_est_path,str(i)+'.txt'))

# # Estimate the F
# Eva.EstimateFM("RANSAC")


Eva.ExactGoodMatch(True)
Eva.DrawEpipolarLines(i)
Eva.FMEvaluate()
