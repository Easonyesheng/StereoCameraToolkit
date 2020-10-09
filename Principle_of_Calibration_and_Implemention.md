# 单目相机标定原理及实现

    基于python-opencv。
---

## 原理

整体的原理是张氏标定法，也就是用棋盘图片来完成标定的方法，由张正友教授在98年提出[参考论文](https://www.researchgate.net/publication/3193178_A_Flexible_New_Technique_for_Camera_Calibration)。  

### 手撕公式

    首先是从数学上推导标定的可行性，公式较多！  
标定原理的最底层一定是从相机投影矩阵说起，因为标定的目的就是要求得相机的相关成像参数。那么对于针孔相机模型，其投影的过程可以通过数学模型来描述，不过在此之前，先
要介绍三个坐标系。  

* 世界坐标系
    为三维世界建立的坐标系，点常用大写字母表示，齐次坐标坐标为$Q = [X,Y,Z,1]$。  
* 相机坐标系
    相机坐标系的原点是相机中心所在位置，同时Z轴是相机面朝的方向，点的坐标记为$q = [x,y,1]$。
* 图像坐标系
    相机生成图像平面为坐标系的XY平面，Z轴则朝向相机所指方向。一般与相机坐标系间存在一个尺度的变换（真实尺度到像素）。

相机的投影模型描述的就是点从世界坐标系到图像坐标系的投影，$q = PQ$，P为相机投影矩阵。具体可以写成下图的形式：  
![img](http://easonzhangyesheng.gitee.io/imgnorthpointer/Camera/ProjectionMatrix.png)
图中M是相机的内参，包括x，y轴上的焦距和相机中心的偏移（相机制造时不可避免的偏移）。W是相机的外参，包括从世界坐标系到相机坐标系的旋转和平移矩阵。s则是相机坐标系到图像坐标系的尺度变换。  
所以从理论上讲，如果我们要实现标定（明确相机内外参数），必然需要明确两个坐标系下的点坐标。  
同时，注意到相机内参是不会随坐标系变化的，而相机外参则是对每个世界/图像坐标系都有不同的值。所以用同一坐标系定义下的多组点，或者用不同坐标系定义下的多组点都可以通过方程的堆积来求出相机的内外参。  
但是张正友教授采用的是另一种更简单而高效的思路，就是从平面出发来标定。  
图像平面自然是三维世界中的一个平面，而如果世界坐标系中的点也存在于一个平面，那么从平面到平面的点的投影就可以用另一种方式描述 --  单应变换（Homography Transformation）。  
单应变换用一个3x3的矩阵来表示点到点的投影，这个矩阵可以通过四组点的对应来求。而为了使世界坐标系下的点坐标已知，这里的平面选择的是存在明确角点的国际象棋棋盘（设角点间的实际距离为1，第一个角点坐标为$(0,0,0)$，第二个为$(0,1,0)$，以此类推）。  
![img](http://easonzhangyesheng.gitee.io/imgnorthpointer/Camera/PlaneTrans.png)
此时，原先的点变换可以写为：$q = HQ$，而将棋盘设定为世界坐标系的XY平面，则棋盘上的点z坐标为0，所以有：  
![img](http://easonzhangyesheng.gitee.io/imgnorthpointer/Camera/Ze0.png)
将单应变换和投影变换联立，则有：  
![img](http://easonzhangyesheng.gitee.io/imgnorthpointer/Camera/Homo2ProjM.png)
所以相机的外参就可以用h表示为：  
![img](http://easonzhangyesheng.gitee.io/imgnorthpointer/Camera/getExt.png)
上式中$r_1.r_2$是旋转向量，所以满足正交且模长相等的约束：$r_1^Tr_2 = 0, |r_1| = |r_2|$  
所以可以推出以下公式（称为两个约束）：  
$h_1^T M^{-T}M^{-1}h_2=0 $  
$h_1^T M^{-T}M^{-1}h_1 = h_2^T M^{-T}M^{-1}h_2 $  
注意到两个式子中$M^{-T}M^{-1}$作为整体出现，所以可以用一个矩阵来代替：  
![img](http://easonzhangyesheng.gitee.io/imgnorthpointer/Camera/BM.png)
同时，B可以由相机内参唯一解出：  
![img](http://easonzhangyesheng.gitee.io/imgnorthpointer/Camera/B2Int.png)
注意到B是一个对称矩阵，所以有6个未知元素（但实际只有4个自由度），且约束中有$h_i^T B h_j$作为整体出现，重新排列B的元素构成向量b，则对任意的$h_i, h_j$有：  
![img](http://easonzhangyesheng.gitee.io/imgnorthpointer/Camera/hBh.png)
将两个约束带入，得到：  
![img](http://easonzhangyesheng.gitee.io/imgnorthpointer/Camera/Vb.png)
到这一步我们可以发现，$v_ij$是只和H有关的向量，而b是只和相机内参有关的向量，所以只要有足够多的H就可以通过这个方程组求出b，即求出相机内参。而对于单个H（一个朝向的棋盘）可以提供两个方程，对于b，其自由度为4，所以只要最少2个不同的H就可以求出b，这也是为什么实际操作时需要拍不同朝向的棋盘图片的原因。  
相机内参求出如下：  
![img](http://easonzhangyesheng.gitee.io/imgnorthpointer/Camera/getInt.png)
求出相机内参后，对于特定棋盘的外参也可以求出：  
![img](http://easonzhangyesheng.gitee.io/imgnorthpointer/Camera/getExt.png)
而以上部分是没有考虑相机的两个畸变的（径向畸变和切向畸变），设$(x_p, y_p)$是点的真实位置，$(x_d, y_d)$是畸变后的位置，有替换公式如下：  
![img](http://easonzhangyesheng.gitee.io/imgnorthpointer/Camera/distortion.png)
相当于往原先的方程组中引入了5个未知参数，这样就需要更多的方程，即更多的棋盘图片。  

---

## 实现

实现是基于python-opencv。  
棋盘图片需要选用长宽不等的（例如6x7），便于确认朝向，其次是棋盘的边缘最好留白，这样不会对找角点的函数造成影响。  
具体流程是先用读入棋盘图片，然后用cv2.findChessboardCorners去寻找每张棋盘的角点，如果能够全部找到，就再去寻找亚像素坐标，用cv2.cornerSubPix。然后就是用cv2.calibrateCamera来标定，具体可以看代码中ModelCalibrator.py的Calibrator类的run方法。  

—--

## Code

求Star～ [传送门](https://github.com/Easonyesheng/StereoCameraToolkit)  
