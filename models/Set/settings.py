

import os
import yaml




# Normal
IMGPATH = '/Users/zhangyesheng/Desktop/Dataset/StereoCamera_db/pics/test_calib'
CAMERANAME = 'test'

# IMGPATH = '/Users/zhangyesheng/Desktop/Dataset/StereoCamera_db/pics/calibrator_data_0927/left'
# CAMERANAME = 'left'

# STEREOIMGPATH = '/Users/zhangyesheng/Desktop/Dataset/StereoCamera_db/pics/calibrator_data_0927/' # Anba
# STEREOIMGPATH = '/Users/zhangyesheng/Desktop/Dataset/StereoCamera_db/Res/Calib/undistort/' # Anba undistort
STEREOIMGPATH = '/Users/zhangyesheng/Desktop/Dataset/StereoCamera_db/pics/imagepairs_1010' # HaiKang

# IMGPATH = '/Users/zhangyesheng/Desktop/Dataset/StereoCamera_db/pics/calibrator_data_0927/right'
# CAMERANAME = 'right'

TASK = 'Calibration'
SAVEPATH = '/Users/zhangyesheng/Desktop/Dataset/StereoCamera_db/Res/Calib'
SAVEPREFIX = 'calibration_test'
CHESSBOARDSIZE = [8,12] # HaiKang
# CHESSBOARDSIZE = [6,13] # Anba

LOGFILE = '/Users/zhangyesheng/Desktop/Research/GraduationDesign/StereoVision/StereoCamera/log/log.txt'
CONFIGPATH = '/Users/zhangyesheng/Desktop/Research/GraduationDesign/StereoVision/StereoCamera/config'