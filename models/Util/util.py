"""Utility """

import numpy as np
import cv2
import os
import logging

def check_string_is_empty(string):
    """name
        check string empty or not
    Args: 

    Returns:

    """
    if string == '':
        return True

    return False

def check_numpy_array(array):
    """name
        check array empty or not
    Args: 

    Returns:
        True - Exist
    """
    try:
        array.all()
    except AttributeError:
        return False
    
    return True

def after_cv_imshow():
    """name

        close all the show window if press 'esc'
        set after cv2.imshow()

    Args:

    Returns:

    """
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()

def save_img_with_prefix(img, path, name):
    """name

        save as 'path/name.jpg'

    Args:

    Returns:

    """
    cv2.imwrite(os.path.join(path,name+'.jpg'), img)

def img_show(img, name):
    """
    """
    cv2.startWindowThread()
    img = img / np.max(img)
    cv2.imshow(name, img)
    after_cv_imshow()

def test_dir_if_not_create(path):
    """name

        save as 'path/name.jpg'

    Args:

    Returns:
    """
    if os.path.isdir(path):
        return True
    else:
        print('Create New Folder:', path)
        os.makedirs(path)
        return True

def log_init(logfilename):
    """name

        save as 'path/name.jpg'

    Args:

    Returns:
    """
    logging.basicConfig(filename=logfilename, level=logging.DEBUG)