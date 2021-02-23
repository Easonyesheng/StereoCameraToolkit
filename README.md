# StereoCameraToolkit

 A Toolkit for monocular/stereo camera (self-)calibration and stereo images processing.

## Refactoring this code now

    Now you can read the /models/*.py for a preview.

## The Code Structure  

    The Basic model is in ModelCamera.py which can perform camera calibration by ModelCalibrator.py and Load imgs or Parameters by ModelLoader.py.  
    The ModelEvaluate can be used to evaluate the accuracy.  
    The ModelStereoCamera are combined with two Camera models with some expanded function.  
