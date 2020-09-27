"""Rectify"""
# [9.768250931346025e-09,-2.500754865249672e-07,-9.548449895853595e-04;-6.860072261347790e-07,-2.777662381107611e-07,0.106495821734710;0.001526302816024,-0.106009349648440,-1.225012201964717]
from Main import SelfCalibration
import cv2
import numpy as np
import time

ImgPath = ''
ParaPath = ''
SavePath = '/Users/zhangyesheng/Desktop/Research/GraduationDesign/StereoVision/StereoCamera/Res/Rect'
SavePrefix = '1_1'


Rect = SelfCalibration(ImgPath,ParaPath,SavePath,SavePrefix)

# load imgs
left_name = '/Users/zhangyesheng/Desktop/Research/GraduationDesign/StereoVision/StereoCamera/left/2.png'
right_name = '/Users/zhangyesheng/Desktop/Research/GraduationDesign/StereoVision/StereoCamera/right/2.png'
Rect.load_image_pair(left_name, right_name)

# load F
F = np.array([9.768250931346025e-09,-2.500754865249672e-07,-9.548449895853595e-04,-6.860072261347790e-07,-2.777662381107611e-07,0.106495821734710,0.001526302816024,-0.106009349648440,-1.225012201964717])
F.resize((3,3))
Rect.FE = F
Rect.F = F
# Rect.Kl = np.array([2.322705987395302e+03,0,0,-0.167747844550470,2.320091628959279e+03,0,7.307396133519883e+02,5.631600717557235e+02,1])
# Rect.Kl.resize((3,3))
# Rect.Kr = np.array([2.333911905221733e+03,0,0,-2.581942636556887,2.331132299259980e+03,0,7.523197258658565e+02,5.591791897938467e+02,1])
# Rect.Kr.resize((3,3))
# Rect.dl = np.array([-0.003088019278731,9.051004572184039e-04,-0.082022534117694,0.985042599989443,-12.842760788832772])
# Rect.dl.resize((1,5))
# Rect.dr = np.array([-0.003387135997683,-0.002453459880741,-0.058123299595020,-0.475182841732686,6.312811785143997])
# Rect.dr.resize((1,5))
# Rect.R_stereo = np.array([0.999787587893149,-5.630929053392975e-04,-0.020602476093372,4.378413846965960e-04,0.999981399817143,-0.006083445953518,0.020605518428805,0.006073133139285,0.999769238206574])
# Rect.R_stereo.resize((3,3))
# Rect.R_stereo.T

# a1 = -2.467541498923884e+02
# a2 = -2.532822135256471
# a3 = 1.369568704851006
# Rect.T_stereo = np.array([a1,a2,a3])
# Rect.T_stereo.resize((3))


# # get matching points
start = time.time()
Rect.ExactGoodMatch()
end = time.time()
print('Matching points time: %f'%(end-start))
# Rect.Show()
# Rectify

Rect.RectifyImgUncalibrated()
# Rect.RectifyImg()
