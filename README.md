# StereoCameraToolkit

 A Toolkit for (monocular\binocular)camera calibration and rectification. 
---

- [x] Refactoring this code now

 **Announcement！**
    All the useful codes are in ./models/

## The Code Structure  

- The Basic model is in ModelCamera.py which can perform camera calibration by ModelCalibrator.py and Load imgs or Parameters by ModelLoader.py.  
- The ModelEvaluate can be used to evaluate the accuracy.  
- The ModelStereoCamera is combined with two Camera models for stereo camera with some expanded function.  
