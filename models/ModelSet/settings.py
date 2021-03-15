

import os
import yaml




# Normal
IMGPATH = r'D:\DeepCalib\CalibrationNet\Dataset\For_Traditional_Calib_20_3\img'
CAMERANAME = 'test'

# IMGPATH = '/Users/zhangyesheng/Desktop/Dataset/StereoCamera_db/pics/calibrator_data_0927/left'
# CAMERANAME = 'left'

# STEREOIMGPATH = '/Users/zhangyesheng/Desktop/Dataset/StereoCamera_db/pics/calibrator_data_0927/' # Anba
# STEREOIMGPATH = '/Users/zhangyesheng/Desktop/Dataset/StereoCamera_db/Res/Calib/undistort/' # Anba undistort
STEREOIMGPATH = '/Users/zhangyesheng/Desktop/Dataset/StereoCamera_db/pics/imagepairs_1010' # HaiKang

# IMGPATH = '/Users/zhangyesheng/Desktop/Dataset/StereoCamera_db/pics/calibrator_data_0927/right'
# CAMERANAME = 'right'

TASK = 'Calibration'
SAVEPATH = r'D:\StereoCamera\Res'
SAVEPREFIX = 'calibration_test'
# CHESSBOARDSIZE = [8,12] # HaiKang
# CHESSBOARDSIZE = [6,13] # Anba
CHESSBOARDSIZE = [6,8] # Syn


LOGFILE = r'D:\StereoCamera\log\log.txt'
CONFIGPATH = r'D:\StereoCamera\config'
WRITEPATH = r''