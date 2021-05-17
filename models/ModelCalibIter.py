''' Find good pose combination '''
import os
import glob
import random


from ModelCamera import Camera
from ModelUtil.util import *
from ModelSet.settings import *




# read GT K
res = np.load(GT_path, allow_pickle=True)
res = res.item()
K_gt = res['K']
fx_gt = K_gt[0,0]
fy_gt = K_gt[1,1]
px_gt = K_gt[0,2]
py_gt = K_gt[1,2]


# while 
max_iter = 1000
iter_count = 0
best_img_number = 0
err_thred = 1
err_ip = np.infty
min_err = np.infty
name_min = []
data_name = IMGPATH.split('\\')[-2]

log_init(LOGFILE)
test = Camera('test')
all_img_list = glob.glob(os.path.join(IMGPATH,'*.jpg'))
img_num = len(all_img_list)

while err_ip > err_thred and iter_count < max_iter:
    # construct images 
    best_img_number = random.randint(3,(img_num//3))
    list_index = []
    img_list = []
    names = []

    j = 0
    while j < best_img_number:
        index = random.randint(0,img_num-1)
        if index not in list_index:
            list_index.append(index)
            j+=1
    for i in list_index:
        img_list.append(all_img_list[i])
        names.append(all_img_list[i].split('\\')[-1].split('.')[0])

    test.Image = None
    test.load_images(img_list=img_list ,load_mod_flag='poses')
    test.calibrate_camera(draw_flag=False, show_flag=False, save_flag=False)
    K_pred = test.IntP
    fx_pred = K_pred[0,0]
    fy_pred = K_pred[1,1]
    px_pred = K_pred[0,2]
    py_pred = K_pred[1,2]

    err_f = (abs(fx_pred-fx_gt)+abs(fy_pred- fy_gt))/2
    err_p = (abs(px_pred-px_gt)+abs(py_pred- py_gt))/2
    err_ip = (err_f+err_p)/2

    if err_ip < min_err :
        min_err = err_ip
        name_min = names[:]
    
    logging.info(f'{iter_count} Minimal error of {data_name} currently = {err_ip} minimal = {min_err}')
        
    iter_count += 1

if iter_count==max_iter:
    logging.info(f'No Good Pose! Calibration failed with minimal error = {min_err}.\nThe min error name is {name_min}')
else:
    logging.info(f'Calibration  Done with error = {err_ip} in {iter_count} iter. Select images as follow:')
    for name in names:
        print(name)

test.show_attri()