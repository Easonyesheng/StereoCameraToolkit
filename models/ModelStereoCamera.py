"""The class of binocular camera """
# function comment template
"""name
    description
Args:
            
        
Returns:

"""
# class comment template
"""The class of 

    description

Attributes:      

"""

import cv2
import numpy as np
from tqdm import tqdm
import yaml
import logging

from settings import *
from Util.util import *


class StereoCamera(object):
    """class of binocular camera

        Mainly used to calculate fundamental matrix etc.

    Attributes:
        camera_left: class camera 
        camera_right: class camera

        FM: Fundamental matrix [3x3]
        EM: Essencial matrix [3x3]
        R_relate: 
        t_relate: 

        Config: a dictionary can be used to set parameters   

            
    """