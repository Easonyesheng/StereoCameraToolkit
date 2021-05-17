

import os
import yaml




# Normal
# IMGPATH = r'D:\DeepCalib\CalibrationNet\Dataset\ForCalibAc30_newdata\img'
# IMGPATH = r'D:\DeepCalib\CalibrationNet\Dataset\ForCalibAc5_bgdata\img'
# IMGPATH = r'D:\DeepCalib\CalibrationNet\Dataset\data_vs_noise\data_vs_noise_radius1.5_bg\ForCalibAc5_bg\img'
# IMGPATH = r'D:\DeepCalib\CalibrationNet\Dataset\data_real\img' # real
# IMGPATH = r'D:\DeepCalib\CalibrationNet\Dataset\data_vs_light_nobg\ForCalibAc7_light\img\O'
# IMGPATH = r'D:\DeepCalib\CalibrationNet\Dataset\data_vs_noise\data_vs_noise_radius1.5_no_bg\img'
IMGPATH = r'D:\DeepCalib\CalibrationNet\Dataset\data_vs_dist\data_dist045\ForCalibAc0_dist045\dist_img'
GT_path = r'D:\DeepCalib\CalibrationNet\Dataset\data_vs_dist\data_dist045\ForCalibAc0_dist045\GT\0-0.npy'

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
CHESSBOARDSIZE = [6,5] # Syn
# CHESSBOARDSIZE = [5,6] # Syn


LOGFILE = r'D:\StereoCamera\log\log.txt'
CONFIGPATH = r'D:\StereoCamera\config'
WRITEPATH = r''