from SelfCalibration import SelfCalibration
import cv2
import numpy as np 
import os
import sys


DatasetPath = ''
ParaPath = ''
# SavePath = r'D:\F_Estimation_private\RealWorldPred\ManualData\RANSAC'
# SavePath = r'D:\F_Estimation_private\RealWorldPred\ManualData'
SavePath = r'D:\F_Estimation_private\RealWorldPred\ManualData\TUM_L1'

i = int(sys.argv[1])
left_path = r'F:\Dataset\FM_Manua\dataset_demo_0803\dataset_demo_0803\left_dataset_demo_0803\left'+ str(i).zfill(4)+'.jpg'
right_path = r'F:\Dataset\FM_Manua\dataset_demo_0803\dataset_demo_0803\right_dataset_demo_0803\right'+str(i).zfill(4)+'.jpg'

FPath_GT = r'F:\Dataset\FM_Manua\dataset_demo_0803\dataset_demo_0803\F_gt.txt' 
FPath_pred = r'D:\F_Estimation_private\RealWorldPred\ManualData\0.txt'


# SavePrefix = 'RANSAC'+str(i) # 7, 18
SavePrefix = 'TUM_L1_plus_%d' % i 
# SavePrefix = 'GT_%d' % i



eva = SelfCalibration(DatasetPath,ParaPath,SavePath,SavePrefix)

eva.load_image_pair(left_path, right_path)

eva.F = eva.Load_F_test(FPath_GT)


eva.Load_F_test(FPath_pred)

# eva.EstimateFM()

eva.ExactGoodMatch(screening=True)

eva.DrawEpipolarLines(i)

eva.FMEvaluate()