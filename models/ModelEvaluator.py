"""The class of evaluation """
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
import sys
from tqdm import tqdm

from Util.util import *
from Set.settings import *



class Evaluator(object):
    """The class of evaluation

        A function room.
        Never initialization.

    Attributes:

        save_path: 
        save_prefix: 
    """

    def __init__(self):
        """name
            description
        Args:
            
        Returns:

        """ 
        self.save_path = ''
        self.save_prefix = ''
    
    def evaluate_calibration(self, objpoints, imgpoints, rvecs, tvecs, mtx, dist):
        """name
            use Re-projection Error as metric
        Args:
            
        Returns:

        """ 
        if check_string_is_empty(self.save_path):
            sys.exit("Evaluate without save path! ")

        if check_string_is_empty(self.save_prefix):
            sys.exit("Evaluate without save prefix! ")

        mean_error = 0
        tot_error = 0
        print('Evaluation...')
        for i in tqdm(range(len(objpoints))):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            tot_error += error
        
        mean_error = tot_error/len(objpoints)

        # print("total error: ", mean_error)
        return mean_error