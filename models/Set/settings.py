

import os
import yaml




# Normal
IMGPATH = '/Users/zhangyesheng/Desktop/Dataset/StereoCamera_db/pics/test_calib'
CAMERANAME = 'test'

# IMGPATH = '/Users/zhangyesheng/Desktop/Dataset/StereoCamera_db/pics/calibrator_data_0927/left'
# CAMERANAME = 'left'

# STEREOIMGPATH = '/Users/zhangyesheng/Desktop/Dataset/StereoCamera_db/pics/calibrator_data_0927/'
STEREOIMGPATH = '/Users/zhangyesheng/Desktop/Dataset/StereoCamera_db/Res/Calib/undistort/'

# IMGPATH = '/Users/zhangyesheng/Desktop/Dataset/StereoCamera_db/pics/calibrator_data_0927/right'
# CAMERANAME = 'right'

TASK = 'Calibration'
SAVEPATH = '/Users/zhangyesheng/Desktop/Dataset/StereoCamera_db/Res/Calib'
SAVEPREFIX = 'calibration_test'
CHESSBOARDSIZE = [6,13]
LOGFILE = '/Users/zhangyesheng/Desktop/Research/GraduationDesign/StereoVision/StereoCamera/log/log.txt'
CONFIGPATH = '/Users/zhangyesheng/Desktop/Research/GraduationDesign/StereoVision/StereoCamera/config'