# Python实现的自标定系统

-----

## Requirements

python-opencv 3.14  
numpy  
os  
sys  

-----

## 基本流程

```flow
    st=>start: Load Dataset
    op1=>operation: Estimate Fundamental matrix
    op2=>operation: Get essential matrix
    op3=>operation: Get camera matrix[R t]
    op4=>operation: Rectify Images
    ed=>end: Output Images
    st->op1->op2->op3->op4->ed
```

-----

## 读入图片和参数

图片是使用常规的imread，重点是参数的读取。  
这里选取的参数是kitti的标定文件，读取方式如下：  

```python
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
```

-----

## 估计基础矩阵

基础矩阵的估计采用的是传统的算法，可选的是**SIFT+RANSAC**和**SIFT+LMedS**。  
关键代码如下：  

提取特征点算法SIFT:  

```python
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
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F,mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

    # mask 是之前的对应点中的内点的标注，为1则是内点。
    # select the inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    self.match_pts1 = np.int32(pts1)
    self.match_pts2 = np.int32(pts2)

```

由坐标估计基础矩阵：  

```python
        if method == "RANSAC":
            self.FE, self.Fmask = cv2.findFundamentalMat(self.match_pts1,
                                                    self.match_pts2,
                                                    cv2.FM_RANSAC, 0.1, 0.99)
        elif method == "LMedS":
            self.FE, self.Fmask = cv2.findFundamentalMat(self.match_pts1,
                                                    self.match_pts2,
                                                    cv2.FM_LMEDS, 0.1, 0.99)
```

对于基础矩阵估计的精度，有两个评价标准以及一个可视化的方法。  
**Metrics**
**1.对极几何约束**
根据对极几何，基础矩阵和两张图的对应点之间满足以下公式：  
$xFx'=0$  
所以通过计算这个式子，可以大致判断基础矩阵的精度。  
**2.对极线距离**
基础矩阵描述的是左图的点到右图对应极线的变换，而且右图的对应点也应该在这条极线上。  
通过计算，点到极线的距离，就可以大致评价基础矩阵的精度。  

代码如下：

```python
 def EpipolarConstraint(self,F,pts1,pts2):
        '''Epipolar Constraint
            calculate the epipolar constraint 
            x^T*F*x
            :output 
                err_permatch
        '''

        print('Use ',len(pts1),' points to calculate epipolar constraints.')
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
        """
        epsilon = 1e-5
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
```

**可视化**
可视化是通过画出两张图上的对应点和极线来实现的。  
因为极线会交于一点（极点），通过这个可以大致判断基础矩阵的精度。  
代码如下：

```python
def DrawEpipolarLines(self):
    """For F Estimation visulization, drawing the epipolar lines
        1. find epipolar lines
        2. draw lines
    """
    try:
        self.FE.all()
    except AttributeError:
        self.EstimateFM() # use RANSAC as default
    try:
        self.match_pts1.all()
    except AttributeError:
        self.ExactGoodMatch()

    # Find epilines corresponding to points in right image (second image)
    # and drawing its lines on left image
    pts2re = self.match_pts2.reshape(-1, 1, 2)
    lines1 = cv2.computeCorrespondEpilines(pts2re, 2, self.FE)
    lines1 = lines1.reshape(-1, 3)
    img3, img4 = self._draw_epipolar_lines_helper(self.imgl, self.imgr,
                                                    lines1, self.match_pts1,
                                                    self.match_pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    pts1re = self.match_pts1.reshape(-1, 1, 2)
    lines2 = cv2.computeCorrespondEpilines(pts1re, 1, self.FE)
    lines2 = lines2.reshape(-1, 3)
    img1, img2 = self._draw_epipolar_lines_helper(self.imgr, self.imgl,
                                                    lines2, self.match_pts2,
                                                    self.match_pts1)

    cv2.imwrite(os.path.join(self.SavePath,"epipolarleft.jpg"),img1)
    # print("Saved in ",os.path.join(self.SavePath,"epipolarleft.jpg"))
    cv2.imwrite(os.path.join(self.SavePath,"epipolarright.jpg"),img3)

    cv2.startWindowThread()
    cv2.imshow("left", img1)
    cv2.imshow("right", img3)
    k = cv2.waitKey()
    if k == 27:
        cv2.destroyAllWindows()
```

-----

## 由基础矩阵分解本质矩阵

这一步用到的公式是：$E = Kl\times F\times Kr$  
代码如下：  

```python
def _Get_Essential_Matrix(self):
    """Get Essential Matrix from Fundamental Matrix
        E = Kl^T*F*Kr
    """
    self.E = self.Kl.T.dot(self.FE).dot(self.Kr)
```

-----

## 由本质矩阵分解得到相机矩阵[R|T]

这里要用到SVD分解，并且要根据结果来测试四种可能性。具体原理可以去看*Multiple View Geometry in computer vision*.  
代码如下：  

```python
def _Get_R_T(self):
    """Get the [R|T] camera matrix
        After geting the R,T, need to determine whether the points are in front of the images
    """
    # decompose essential matrix into R, t (See Hartley and Zisserman 9.13)
    U, S, Vt = np.linalg.svd(self.E)
    W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                    1.0]).reshape(3, 3)

    # iterate over all point correspondences used in the estimation of the
    # fundamental matrix
    first_inliers = []
    second_inliers = []
    for i in range(len(self.Fmask)):
        if self.Fmask[i]:
            # normalize and homogenize the image coordinates
            first_inliers.append(self.Kl_inv.dot([self.match_pts1[i][0],
                                    self.match_pts1[i][1], 1.0]))
            second_inliers.append(self.Kr_inv.dot([self.match_pts2[i][0],
                                    self.match_pts2[i][1], 1.0]))

    # Determine the correct choice of second camera matrix
    # only in one of the four configurations will all the points be in
    # front of both cameras
    # First choice: R = U * Wt * Vt, T = +u_3 (See Hartley Zisserman 9.19)
    R = U.dot(W).dot(Vt)
    T = U[:, 2]
    if not self._in_front_of_both_cameras(first_inliers, second_inliers,
                                            R, T):
        # Second choice: R = U * W * Vt, T = -u_3
        T = - U[:, 2]

    if not self._in_front_of_both_cameras(first_inliers, second_inliers,
                                            R, T):
        # Third choice: R = U * Wt * Vt, T = u_3
        R = U.dot(W.T).dot(Vt)
        T = U[:, 2]

        if not self._in_front_of_both_cameras(first_inliers,
                                                second_inliers, R, T):
            # Fourth choice: R = U * Wt * Vt, T = -u_3
            T = - U[:, 2]

    self.match_inliers1 = first_inliers
    self.match_inliers2 = second_inliers
    self.Rt1 = np.hstack((np.eye(3), np.zeros((3, 1)))) # as defaluted RT 
    self.Rt2 = np.hstack((R, T.reshape(3, 1)))
```

-----

## 图片校正

这里我提供了两种校正的算法。  
一种是要用到之前的相机参数实现的校正，依赖于cv2.stereoRectify() 
都是固有流程，原理见*Multiple View Geometry in computer vision*  
代码如下：  

```python
def RectifyImg(self):
    """Rectify images using cv2.stereoRectify()
    """
    try:
        self.FE.all()
    except AttributeError:
        self.EstimateFM() # Using traditional method as default

    self._Get_Essential_Matrix()
    self._Get_R_T()

    R = self.Rt2[:, :3]
    T = self.Rt2[:, 3]
    #perform the rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.Kl, self.dl,
                                                        self.Kr, self.dr,
                                                        self.imgl.shape[:2],
                                                        R, T, alpha=0)
    mapx1, mapy1 = cv2.initUndistortRectifyMap(self.Kl, self.dl, R1, P1,
                                                [self.imgl.shape[0],self.imgl.shape[1]],
                                                cv2.CV_32FC1)
    mapx2, mapy2 = cv2.initUndistortRectifyMap(self.Kr, self.dr, R2, P2,
                                                [self.imgl.shape[0],self.imgl.shape[1]],
                                                cv2.CV_32FC1)
    img_rect1 = cv2.remap(self.imgl, mapx1, mapy1, cv2.INTER_LINEAR)
    img_rect2 = cv2.remap(self.imgr, mapx2, mapy2, cv2.INTER_LINEAR)

    # save
    cv2.imwrite(os.path.join(self.SavePath,"RectedLeft.jpg"),img_rect1)
    cv2.imwrite(os.path.join(self.SavePath,"RectedRight.jpg"),img_rect2)
```

另外一种是不需要相机内参的校正算法，依赖于cv2.stereoRectifyUncalibrated()  
代码如下：  

```python
def returnH1_H2(points1,points2,F,size):
    p1=points1.reshape(len(points1)*2,1)#stackoverflow上需要将(m,2)的点变为(m*2,1),因为不变在c++中会产生内存溢出
    p2=points2.reshape(len(points2)*2,1)
    _,H1,H2=cv2.stereoRectifyUncalibrated(p1,p2,F,size) #size是宽，高
    return H1,H2
```

-----

## 结果及说明

校正前：  

校正后：

