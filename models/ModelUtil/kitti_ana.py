"""
The class to analyse the KITTI dataset
Including:
    1.load imgs
    2.paser calib_file
        calculate the F matrix
    3.calculate metrics
        epipolar constraint
        symmetry epipolar distance
    4.Visulazation
        draw the epipolar lines 

"""

import numpy as np 
import os 
import cv2 



class KittiAnalyse(object):

    def __init__(self,imgpath,calib_path,save_path,label = 'RAW'):
        self.imgpath = imgpath
        self.calib_path = calib_path
        self.save_path = save_path
        self.label = label
        self.X = None
        self.calib = self.Paser()
        self.match_pts1 = None
        self.match_pts2 = None 
        self.F = None
        
    
    def load_img_patch(self):
        '''
        load images and use a vessel to save them
        '''
        # get the ralated parameters
        path_dir = os.path.join(self.imgpath, 'image_00\\data\\')
        N = 0
        imgspath = os.listdir(path_dir)
        N = len(imgspath)
        print('How many images: ', N)
        # print('test img: ',os.path.join(path_dir,imgspath[0]))
        imgtest = cv2.imread(os.path.join(path_dir,imgspath[0]),0)
        w,h = imgtest.shape

        # set the vessel
        X = np.zeros((N,w,h,2))

        # load the images
        path_dir_l = os.path.join(self.imgpath, 'image_00\\data\\')
        path_dir_r = os.path.join(self.imgpath, 'image_01\\data\\')

        l_imgs = os.listdir(path_dir_l)
        r_imgs = os.listdir(path_dir_r)
        i = 0
        for l_img, r_img in zip(l_imgs, r_imgs):
            l_img_path = os.path.join(path_dir_l,l_img)
            r_img_path = os.path.join(path_dir_r,r_img)
            img_l = cv2.imread(l_img_path,0)
            img_r = cv2.imread(r_img_path,0)

            imgs= np.zeros((2, w, h)) # H, W interchanged here since numpy takes H,W as input
            imgs[0,:,:] = img_l
            imgs[1,:,:] = img_r

            X[i,:] = np.moveaxis(imgs, [0,1,2], [2,0,1])
            i+=1
        print('vessel shape:',X.shape)
        self.X = X
        self.shape = [w,h]
        return X
    
    def Paser(self):
        '''
        Paser the calib_file 
        return a dictionary
        use it as :
            calib = self.Paser()
            K1, K2 = self.calib['K_0{}'.format(f_cam)], self.calib['K_0{}'.format(t_cam)]
        '''
        d = {}
        with open(self.calib_path) as f:
            for l in f:
                if l.startswith("calib_time"):
                    d["calib_time"] = l[l.index("calib_time")+1:]
                else:
                    [k,v] = l.split(":")
                    k,v = k.strip(), v.strip()
                    #get the numbers out
                    v = [float(x) for x in v.strip().split(" ")]
                    v = np.array(v)
                    if len(v) == 9:
                        v = v.reshape((3,3))
                    elif len(v) == 3:
                        v = v.reshape((3,1))
                    elif len(v) == 5:
                        v = v.reshape((5,1))
                    d[k] = v
        return d

    def F_GT_rected_get(self,f_cam='0',t_cam='1'):
        '''
        get the fundamental matrix of the rectified images
        Calculate the F by 
        F = [e']P'P^+
        where e' = P'C
        where PC = 0
        '''
        P, P_ = self.calib['P_rect_0{}'.format(f_cam)], self.calib['P_rect_0{}'.format(t_cam)]
        P = P.reshape(3,4)
        P_ = P_.reshape(3,4)
        # print('P: ',P)
        P_c = P[:,:3]
        zero = P[:,3:]
        zero = -1*zero
        c = np.linalg.solve(P_c,zero)
        C = np.ones([4,1])
        C[:3,:] = c
        e_ = np.dot(P_,C)
        e_M = np.array([
            [0, -e_[2,0], e_[1,0]],
            [e_[2,0], 0, -e_[0,0]],
            [-e_[1,0], e_[0,0], 0]
        ])
        P = np.matrix(P)
        P_wn = np.linalg.pinv(P)
        F = np.dot(np.dot(e_M, P_),P_wn)
        F = F/F[2,2]
        assert np.linalg.matrix_rank(F) == 2
        self.F = F
        return F

    def F_GT_unrect_get(self):
        '''
        get the fundamental matrix of the unrectified images 
        F = K2^(-T)*R*[t]x*K1^(-1)
        '''
        #assemble the ingredients
        K1, K2 = self.calib['K_0{}'.format(f_cam)], self.calib['K_0{}'.format(t_cam)]
        R1, R2 = self.calib['R_0{}'.format(f_cam)], self.calib['R_0{}'.format(t_cam)]
        # R1, R2 = self.calib['R_rect_0{}'.format(f_cam)], self.calib['R_rect_0{}'.format(t_cam)]
        t1, t2 = self.calib['T_0{}'.format(f_cam)], self.calib['T_0{}'.format(t_cam)]

        print(f"K1: {K1}, K2: {K2}, R1: {R1}, R2: {R2}, t1: {t1}, t2: {t2}")


        R = np.dot(R2, np.linalg.inv(R1))
        t = t2 - t1
        
        T = np.array([
            [0,     -t[2], t[1]],
            [t[2],  0,     -t[0]],
            [-t[1], t[0],  0]
        ])
        #compute
        F = np.dot(np.linalg.inv(K2.T), np.dot(T, np.dot(R, np.linalg.inv(K1))))
        F /= F[2,2]
        assert np.linalg.matrix_rank(F) == 2
        self.F = F
        return F
    
    def F_load(self,F_path):
        '''
        load fundamental matrix saved in txt
        '''
        F = np.loadtxt(F_path)
        if F.shape != [3,3]:
            try:
                F = F.reshape(3,3)
            except ValueError:
                print('F shape is wrong :',F.shape)
                return -1
        F = F / F[2,2]
        self.F = F
        return F

    def F_ES(self):
        '''
        use RANSAC to estimation fundamental matrix
        '''
        pts1, pts2 = self.match_pts1, self.match_pts2
        F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
        F = F / F[2,2]
        self.F = F
        return F


    def drawlines(self,imgl, imgr, lines, pts1, pts2 ):
        '''
        draw lines
        '''
        if len(pts1) > 40:
            pts1 = pts1[20:60]
            pts2 = pts2[20:60]
        
        # imgl = img1.copy()
        # imgr = img2.copy()
        r, c, _ = imgl.shape

        i = 0
        for r, pt1, pt2 in zip(lines,pts1,pts2):
            i+=1
            color = np.random.randint(0,255,3).tolist()
            x0,y0 = map(int, [0,-r[2]/r[1]])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])

            

            imgl = cv2.line(imgl.copy(), (x0,y0), (x1,y1), color ,1)
            imgl = cv2.circle(imgl.copy(), tuple(pt1),5,color,-1)
            imgr = cv2.circle(imgr.copy(), tuple(pt2),5,color,-1)

        return imgl, imgr 

    def exact_match_points_sift(self,img_index):
        '''
        exact the matching points list of the [img_index] images
        use sift
        utility function
        '''
        i = img_index
        img1 = np.zeros((self.X.shape[1],self.X.shape[2],3))
        img2 = np.zeros((self.X.shape[1],self.X.shape[2],3))
        img1[:,:,0] = self.X[i,:,:,0]  #queryimage # left image
        img1[:,:,1] = img1[:,:,0]
        img1[:,:,2] = img1[:,:,1]
        img2[:,:,0] = self.X[i,:,:,1]  #trainimage # right image
        img2[:,:,1] = img2[:,:,0]
        img2[:,:,2] = img2[:,:,1]
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)
        # print(img1.dtype)
        # cv2.imwrite(self.save_path+'OriimgL.jpg',img1)
        # cv2.imwrite(self.save_path+'OriimgR.jpg',img2)
        # img1 = cv2.imread(self.save_path+'OriimgL.jpg',0)
        # img2 = cv2.imread(self.save_path+'OriimgR.jpg')
        # img1 = img1.astype(np.uint8)
        # img2 = img2.astype(np.uint8)

        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        good = []
        pts1 = []
        pts2 = []

        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        self.match_pts1 = np.int32(pts1)
        self.match_pts2 = np.int32(pts2)
        
        return  np.int32(pts1),  np.int32(pts2)

    def get_good_match(self,img_index):
        '''
        use F_GT to get the good match points list
        x'Fx < 0.15
        '''
        F = self.F_GT_rected_get()
        try:
            self.match_pts1.all()
        except AttributeError:
            self.exact_match_points_sift(img_index)

        pts1, pts2 = self.match_pts1, self.match_pts2
        leftpts = []
        rightpts = []
        print('lenth of pts before optimization: ', len(pts1))
        for p1, p2 in zip(pts1, pts2):
            hp1, hp2 = np.ones((3,1)), np.ones((3,1))
            hp1[:2,0], hp2[:2,0] = p1, p2
            err = np.abs(np.dot(hp2.T, np.dot(F, hp1)))
            if err < 0.15:
                leftpts.append(p1)
                rightpts.append(p2)
        print('lenth of pts after optimization: ', len(leftpts))
        self.match_pts1 = np.array(leftpts)
        self.match_pts2 = np.array(rightpts)


    def draw_epipolar_lines(self,img_index ,save_prefix = 'index_0_F_GT_Rect_'):
        '''
        draw the epipolar lines of the [img_index]th image pair
        save to save_path with save_prefix
        defaulted use F_GT_rect, SIFT_keypoints
        '''
        try:
            self.X.all()
        except AttributeError:
            self.load_img_patch()

        img_left = np.zeros((self.X.shape[1],self.X.shape[2],3))
        img_right = np.zeros((self.X.shape[1],self.X.shape[2],3))
        img_left[:,:,0] = self.X[img_index,:,:,0]
        img_left[:,:,1] = img_left[:,:,0]
        img_left[:,:,2] = img_left[:,:,1]
        img_right[:,:,0] = self.X[img_index,:,:,1]
        img_right[:,:,1] = img_right[:,:,0]
        img_right[:,:,2] = img_right[:,:,1]
        
        try: 
            self.F.all()
        except AttributeError:
            self.F_GT_rected_get()
            print("load F_GT")
        
        try:
            self.match_pts1.all()
        except AttributeError:
            self.exact_match_points_sift(img_index)
            print("load matching points")
        
        # if len(self.match_pts1) < 10:
        #     img_index+=1
        #     self.exact_match_points_sift(img_index)

        pts1, pts2 = self.match_pts1, self.match_pts2
        save_path = self.save_path

        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, self.F)
        lines1 = lines1.reshape(-1,3)

        img5, img6 = self.drawlines(img_left,img_right,lines1,pts1,pts2)

        print('Imgs saved in: ',os.path.join(save_path,'xxxx.jpg'))
        cv2.imwrite(os.path.join(save_path,save_prefix+'Left.jpg'),img5)

        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 2, self.F)
        lines2 = lines2.reshape(-1,3)

        
        img3, img4 = self.drawlines(img_right,img_left,lines2,pts2,pts1)

        cv2.imwrite(os.path.join(save_path,save_prefix+'Right.jpg'),img3)


    def metrics_ep_cons(self,img_index):
        '''
        calculate the epipolar constraint of the [img_index]th image pair
        x^T*F*x
        '''

        try: 
            self.F.all()
        except AttributeError:
            self.F_GT_rected_get()
        
        try:
            self.match_pts1.all()
        except AttributeError:
            self.exact_match_points_sift(img_index)
        
        
        
        pts1, pts2 = self.match_pts1, self.match_pts2
        print('Use ',len(pts1),' points to calculate epipolar constraints.')
        assert len(pts1) == len(pts2)
        err = 0.0
        for p1, p2 in zip(pts1, pts2):
            hp1, hp2 = np.ones((3,1)), np.ones((3,1))
            hp1[:2,0], hp2[:2,0] = p1, p2
            err += np.abs(np.dot(hp2.T, np.dot(self.F, hp1)))
        
        return err / float(len(pts1))


    def half_sym_epipolar_dist(self,img_index):
        pass




    def metrics_ep_dist(self,img_index):
        '''
        calcualte the symmetry epipolar distance of the [img_index]th image pair
        dist = [(a_q*x_p+b_q*y_p+c_q)^2/(a_q^2 + b_q^2) + (a_p*x_q+b_p*y_q+c_p)^2/(a_p^2 + b_p^2)]
        '''
        epsilon = 1e-5

        try: 
            self.F.all()
        except AttributeError:
            self.F_GT_rected_get()
        F = self.F
        try:
            self.match_pts1.all()
        except AttributeError:
            self.exact_match_points_sift(img_index)
        
        pts1, pts2 = self.match_pts1, self.match_pts2

        assert len(pts1) == len(pts2)
        print('Use ',len(pts1),' points to calculate epipolar distance.')
        err = 0.
        for p1, p2 in zip(pts1, pts2):
            hp1, hp2 = np.ones((3,1)), np.ones((3,1))
            hp1[:2,0], hp2[:2,0] = p1, p2
            fp, fq = np.dot(F, hp1), np.dot(F.T, hp2)
            sym_jjt = 1./(fp[0]**2 + fp[1]**2 + epsilon) + 1./(fq[0]**2 + fq[1]**2 + epsilon)
            err = err + ((np.dot(hp2.T, np.dot(F, hp1))**2) * (sym_jjt + epsilon))

        return err / float(len(pts1))
    

    def MP_move(self, dist=1):
        '''
        move matching points to test metric
        '''
        for i in range(self.match_pts1.shape[0]):
            self.match_pts2[i,0] += dist
            # print("before move: %d" %self.match_pts2[i,1])
            self.match_pts2[i,1] += dist
            # print("after move: %d" %self.match_pts2[i,1])
            if self.match_pts2[i,0] > self.shape[0]:
                self.match_pts2[i,0] = self.shape[0]
            if self.match_pts2[i,1] > self.shape[1]:
                self.match_pts2[i,1] = self.shape[1]


if __name__ == "__main__":

    # imgpath = r'D:\F_Estimation\deepF_noCorrs\data\kitti\2011_09_26\2011_09_26_drive_0051_sync\\'

    imgpath = r'D:\F_Estimation\UnrectifiedData\2011_09_26_drive_0001_extract\2011_09_26\2011_09_26_drive_0001_extract'
    # imgpath = r"D:\F_Estimation\deepF_noCorrs\data\Pred\Pred1_12_07\\"
    # for RAW DATA all images have the same calib_flie
    calib_path = r'D:\F_Estimation\deepF_noCorrs\data\kitti\2011_09_26\calib_cam_to_cam.txt' 

    save_path = r'D:\F_Estimation\deepF_noCorrs\Code\Util\Kitti_ana\Result\Visualization'

    F_path = r'D:\F_Estimation\deepF_noCorrs\Pred_Result\Simaese_suss\0.txt' # F_Simaese_suss
    # F_path = r'D:\F_Estimation\deepF_noCorrs\Pred_Result\F_GT_Changed_SingleNet.txt' # F_Single
    # F_path = r'D:\F_Estimation\deepF_noCorrs\Pred_Result\Simaese_eclossw-3_goodsave_suss\0.txt' # F_epc_loss_suss
    # F_path = r'D:\F_Estimation\deepF_noCorrs\Pred_Result\Simaese_edW-3_fix\0.txt' # F_epd_loss

    # F_name = 'F_GT_testF'
    # F_name = 'F_Simaese_suss'
    # F_name = 'F_RANSAC'
    # F_name = 'F_GT'
    F_name = 'F_ecloss_suss'

    save_path = os.path.join(save_path,F_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('saved in:',save_path)
    kf = KittiAnalyse(imgpath, calib_path, save_path, label='RAW')
    img_index = 4 
    '''
    D:\F_Estimation\deepF_noCorrs\data\kitti\2011_09_26\2011_09_26_drive_0051_sync\\0000000004.png
    RANSAC 失效
    '''
    kf.load_img_patch()
    kf.get_good_match(img_index) # load matching points and get good match

    
    save_prefix = F_name + '_P_SIFT_index' + str(img_index)+'Unrect'#+ '_Optim' #+ 'traindata'


    # # test F 
    # for i in range(10):
    #     kf.F_GT_rected_get()
    #     F_G = kf.F.copy()
    #     kf.F += i/10.
    #     L1_loss = abs(np.sum(F_G - kf.F) / 9.)

    #     ep_c = kf.metrics_ep_cons(img_index)
    #     ep_d = kf.metrics_ep_dist(img_index)

    #     TxtName = os.path.join(save_path,save_prefix+'.txt')

    #     print("Write in txt file: %s" %TxtName)
    #     with open(TxtName, 'a') as f:
    #         f.truncate()
    #     with open(TxtName, 'a') as f:
    #         f.write('\nUse '+F_name+',ImgIndex='+str(img_index))
    #         f.write('\nF change: '+str(i/10))
    #         f.write('\nL1loss: '+str(L1_loss))
    #         f.write('\nepipolar constraint: '+str(ep_c))
    #         f.write('\nepipolar distance: '+str(ep_d))

    # load F
    print('load %r is :' %save_prefix ,kf.F_load(F_path ))
    
    # # Estimation F
    # kf.exact_match_points_sift(img_index)
    # print("F Estimation with RANSAC: ",kf.F_ES())

    kf.draw_epipolar_lines(img_index, save_prefix)
    ep_c = kf.metrics_ep_cons(img_index)
    ep_d = kf.metrics_ep_dist(img_index)
    print('epipolar constraint: ', ep_c)
    print('epipolar distance: ', ep_d)

    TxtName = os.path.join(save_path,save_prefix+'.txt')

    print("Write in txt file: %s" %TxtName)
    with open(TxtName, 'w') as f:
        f.write('Use '+F_name+',ImgIndex='+str(img_index))
        f.write('\nepipolar constraint: '+str(ep_c))
        f.write('\nepipolar distance: '+str(ep_d))


    # # test Metrics
    # for i in range(0,10):
    #     kf.MP_move(i) 
    #     save_prefix = F_name + '_P_SIFT_index' + str(img_index) + '_Optim'
    #     save_prefix += '_move%s' %i
    #     kf.draw_epipolar_lines(img_index, save_prefix)
    #     ep_c = kf.metrics_ep_cons(img_index)
    #     ep_d = kf.metrics_ep_dist(img_index)
    #     print('epipolar constraint: ', ep_c)
    #     print('epipolar distance: ', ep_d)

    #     TxtName = os.path.join(save_path,save_prefix+'.txt')

    #     print("Write in txt file: %s" %TxtName)
    #     with open(TxtName, 'w') as f:
    #         f.write('Use '+F_name+',ImgIndex='+str(img_index))
    #         f.write('\nepipolar constraint: '+str(ep_c))
    #         f.write('\nepipolar distance: '+str(ep_d))
