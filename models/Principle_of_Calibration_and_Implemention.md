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
    为三维世界建立的坐标系，点常用大写字母表示，齐次坐标坐标为$[X,Y,Z,1]$。  
* 相机坐标系
    
* 图像坐标系

---

## Code

求Star～ [传送门](https://github.com/Easonyesheng/StereoCameraToolkit)  
