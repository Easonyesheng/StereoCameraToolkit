"""
Evaluate this dataset
    1. Use GT to perform Visualization
    2. Use LMedS to get F_est to perform Visualization
    3. To Compare
"""

from SelfCalibration import SelfCalibration
import os
import numpy
import cv2

Index = 20
ImgPath = r'D:\F_Estimation_private\deepF_noCorrs\data\TUM\2020_05_27\long_office_household'
SavePath = r'D:\StereoCamera\Res\Eva_TUM'
FTxtFile = r'D:\F_Estimation_private\deepF_noCorrs\data\TUM\2020_05_27\long_office_household\F_gt.txt'
# FTxtFile = r'F:\Dataset\FM_Dataset\TUM\rgbd_dataset_freiburg3_large_cabinet\pairs_with_gt.txt'
ParaPath = ''
SavePrefix = 'Eva_TUM_long_office_household_GT_'

Eva = SelfCalibration(ImgPath, ParaPath, SavePath, SavePrefix)

Eva.load_image_KITTI(Index)

Eva.Load_F_index(FTxtFile,Index)

Eva.ExactGoodMatch(True)

Eva.EstimateFM('RANSAC')
Eva.FE = Eva.F

Eva.DrawEpipolarLines(Index)

Eva.FMEvaluate()