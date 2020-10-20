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

from ModelUtil.util import *
from ModelSet.settings import *



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
    
    def EpipolarConstraint(self,F,pts1,pts2):
        '''Epipolar Constraint
            calculate the epipolar constraint 
            x^T*F*x
            :output 
                err_permatch
        '''
       
        # print('Use ',len(pts1),' points to calculate epipolar constraints.')
        assert len(pts1) == len(pts2)
        err = 0.0
        for p1, p2 in zip(pts1, pts2):
            hp1, hp2 = np.ones((3,1)), np.ones((3,1))
            hp1[:2,0], hp2[:2,0] = p1, p2
            err += np.abs(np.dot(hp2.T, np.dot(F, hp1)))
        
        return err / float(len(pts1))

    def SymEpiDis(self, F, pts1, pts2):
        """Symetric Epipolar distance
            calculate the Symetric Epipolar distance
            To those epipolar lines going through points, calculate the included angle between it and GT line. (cos_all)
        """

        epsilon = 1e-5
        assert len(pts1) == len(pts2)
        # print('Use ',len(pts1),' points to calculate epipolar distance.')
        err = 0.
        sheld = 0.1
        max_dis = 0.
        min_dis = np.Infinity
        cos_all = 0. 
        inliers = 0. 
        angle = -1
        for p1, p2 in zip(pts1, pts2):
            hp1, hp2 = np.ones((3,1)), np.ones((3,1))
            hp1[:2,0], hp2[:2,0] = p1, p2
            fp, fq = np.dot(F, hp1), np.dot(F.T, hp2)
            # print(fp[0],'\n',fq[0])
            sym_jjt = 1./(fp[0]**2 + fp[1]**2 + epsilon) + 1./(fq[0]**2 + fq[1]**2 + epsilon)
            dis = ((np.dot(hp2.T, np.dot(F, hp1))**2) * (sym_jjt + epsilon))
            if dis < sheld:
                # if inliers calculate the angle
                inliers+=1
                # f_GT_p, f_GT_q = np.dot(self.F, hp1), np.dot(self.F, hp2)
                
                # print(f_GT_p,f_GT_q)
                # cos1 = (abs(fp[0]*f_GT_p[0]+fp[1]*f_GT_p[1]))/(sqrt(f_GT_p[0]**2+f_GT_p[1]**2+epsilon)*sqrt(fp[0]**2+fp[1]**2+epsilon))
                # cos2 = (abs(fq[0]*f_GT_q[0]+fq[1]*f_GT_q[1]))/(sqrt(f_GT_q[0]**2+f_GT_q[1]**2+epsilon)*sqrt(fq[0]**2+fq[1]**2+epsilon))
                # cos_all = (cos1 + cos2)/2 + cos_all
            max_dis = max(max_dis, dis)
            min_dis = min(min_dis, dis)
            err = err + dis
        # if inliers == 0:
        #     angle = -1
        # else:
        #     angle = (acos(cos_all/inliers)/3.1415926)*180
        # print('Average angle: ',angle)

        return err / float(len(pts1)), max_dis, min_dis, angle

    def Evaluate_F(self, F, pts1, pts2, img_num=1):
        """
        """
        epi_cons = 0.0
        sym_epi_dis = 0.0
        max_dis = 0.0
        min_dis = 10000 
        # angle = 0.0

        if img_num == 1:
            epi_cons = self.EpipolarConstraint(F, pts1, pts2)
            sym_epi_dis, max_dis, min_dis, angle = self.SymEpiDis(F, pts1, pts2)
        else:
            for i in range(img_num):
                epi_cons_i = self.EpipolarConstraint(F, pts1[i], pts2[i])
                sym_epi_dis_i, max_dis_i, min_dis_i, angle_i = self.SymEpiDis(F, pts1[i], pts2[i])   
                epi_cons += epi_cons_i
                sym_epi_dis += sym_epi_dis_i
                max_dis = max(max_dis, max_dis_i)
                min_dis = min(max_dis, min_dis_i)
                # angle += angle_i

            epi_cons /= img_num
            sym_epi_dis /= img_num
            # angle /= img_num

        logging.info("Evaluate the fundamental matrix")
        logging.info("The quantities of matching points is %d" %len(pts1))
        logging.info("The epipolar constraint is : "+str(float(epi_cons)) +"\nThe symmetry epipolar distance is: " +str(float(sym_epi_dis)))

        file_name = os.path.join(self.save_path, self.save_prefix+'evaluate_F.txt')
        with open(file_name,'w') as f:
            f.writelines("Evaluate the estimated fundamental matrix: "+str(self.save_prefix)+"\n")
            f.writelines("\nThe quantities of matching points is " +str(len(pts2))+"\n")
            f.writelines("The epipolar constraint is : " +str(float(epi_cons))+"\nThe symmetry epipolar distance is: " +str(float(sym_epi_dis)))

    def evaluate_calibration(self, objpoints, imgpoints, rvecs, tvecs, mtx, dist):
        """name
            use average Re-projection Error as metric
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