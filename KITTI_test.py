from SelfCalibration import SelfCalibration
import os  


if __name__ == "__main__":

    for i in range(1):
        prefix = 'KITTI_rected'

        #Mac
        ImgPath = "/Users/zhangyesheng/Documents/GitHub/OANet/Rectify/pics/"+prefix +'/'
        ParaPath = "/Users/zhangyesheng/Documents/GitHub/OANet/Rectify/calibration/"+prefix +'/'
        SavePath = "/Users/zhangyesheng/Documents/GitHub/OANet/Rectify/Res/"+prefix+str(i)+'/'

        #--- KITTI_rected
            
        EPath = "/Users/zhangyesheng/Documents/GitHub/OANet/Rectify/ModelRes/"+prefix+str(i)+"/E.npy"
        leftcorr = "/Users/zhangyesheng/Documents/GitHub/OANet/Rectify/ModelRes/"+prefix+str(i)+"/leftcorr.npy"
        rightcorr = "/Users/zhangyesheng/Documents/GitHub/OANet/Rectify/ModelRes/"+prefix+str(i)+"/rightcorr.npy"
        SavePrefix = '_OANet_'+str(i)


        # initialization
        test = SelfCalibration(ImgPath,ParaPath,SavePath,SavePrefix)

        # load images
        test.load_image_KITTI(i) 

        # load calibration file
        test.LoadPara_KITTI()

        # load OANet output -- matching pairs
        test.LoadCorr(rightcorr,leftcorr)

        # Estimate F
        test.EstimateFM(method="8Points")

        # load F_GT
        test.LoadFMGT_KITTI()

        # screen Good Match for Evaluation
        test.ExactGoodMatch(screening=True,point_lens=18)

        # Epipolar lines visualization
        test.DrawEpipolarLines()

        # Evaluate the F matrix
        test.FMEvaluate()
