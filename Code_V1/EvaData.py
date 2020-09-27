"""
Evaluate this dataset TUM
    1. Use GT to perform Visualization
    2. Use LMedS to get F_est to perform Visualization
    3. To Compare
"""

from SelfCalibration import SelfCalibration
import os
import numpy
import cv2

Index = 10
# ImgPath = r'D:\F_Estimation_private\deepF_noCorrs\data\TUM\2020_05_27\long_office_household' # after processed
ImgPath = r'F:\Dataset\FM_Dataset\TUM\rgbd_dataset_freiburg3_large_cabinet\Images' # before processed
TxtFile = r'F:\Dataset\FM_Dataset\TUM\rgbd_dataset_freiburg3_large_cabinet\pairs_with_gt.txt' # before processed
SavePath = r'D:\StereoCamera\Res\Eva_TUM'
# FTxtFile = r'D:\F_Estimation_private\deepF_noCorrs\data\TUM\2020_05_27\long_office_household\F_gt.txt'
ParaPath = ''
# SavePrefix = 'Eva_TUM_large_cabinet_GT_'
# SavePrefix = 'Eva_TUM_large_cabinet_LMedS_'
SavePrefix = 'Eva_TUM_large_cabinet_RANSAC_'



Eva = SelfCalibration(ImgPath, ParaPath, SavePath, SavePrefix)

# Eva.load_image_KITTI(Index)
Eva.load_img_TUM(Index,TxtFile)

Eva.Load_F_index(TxtFile,Index)



Eva.EstimateFM('RANSAC')
# Eva.FE = Eva.F

Eva.ExactGoodMatch(True)
Eva.DrawEpipolarLines(Index)


Eva.FMEvaluate()