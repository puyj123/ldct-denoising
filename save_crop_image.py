import os

import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt, patches
import matplotlib.patches as patches

def trunc(mat):
    mat[mat <= -160] = -160
    mat[mat >= 240] = 240
    return mat
#y shape:[512,512]
def save_fig(x,y, pred,fig_name,epoch,img_path):
    x1=100
    x2=200
    y1=200
    y2=300
    x=x.cpu().detach().numpy()
    y, pred = y.cpu().detach().numpy(), pred.cpu().detach().numpy()
    # x = x * (3072.0 + 1024.0) - 1024.0
    # y = y * (3072.0 + 1024.0) - 1024.0
    # pred = pred * (3072.0 + 1024.0) - 1024.0
    # x = trunc(x)
    # y = trunc(y)
    # pred =trunc(pred)
    f, ax = plt.subplots(2, 3, figsize=(60, 20))
    pred_crop=pred[y1:y2,x1:x2]
    x_crop=x[y1:y2,x1:x2]
    y_crop=y[y1:y2,x1:x2]
    # 在灰度图像上绘制红色矩形框
    # ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
    # ax[0].set_title('Quarter-dose', fontsize=30)
    #ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                       #original_result[1],
                                                                       #original_result[2]), fontsize=20)
    ax[1][0].imshow(x_crop, cmap=plt.cm.gray, vmin=-160, vmax=240)
    ax[1][0].set_title('low-dose-crop', fontsize=30)
    ax[1][1].imshow(pred_crop, cmap=plt.cm.gray, vmin=-160, vmax=240)
    ax[1][1].set_title('pred-crop', fontsize=30)
    ax[1][2].imshow(y_crop, cmap=plt.cm.gray, vmin=-160, vmax=240)
    ax[1][2].set_title('full-dose-crop', fontsize=30)
    # ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
    #                                                                    pred_result[1],
    #                                                                    pred_result[2]), fontsize=20)
    ax[0][0].imshow(x,cmap=plt.cm.gray, vmin=-160, vmax=240)
    ax[0][0].set_title('low-dose', fontsize=30)
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
    ax[0][0].add_patch(rect)
    ax[0][1].imshow(pred,cmap=plt.cm.gray, vmin=-160, vmax=240)
    ax[0][1].set_title('pred', fontsize=30)
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
    ax[0][1].add_patch(rect)
    ax[0][2].imshow(y,cmap=plt.cm.gray, vmin=-160, vmax=240)
    ax[0][2].set_title('full-dose', fontsize=30)
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
    ax[0][2].add_patch(rect)
    save_path = os.path.join(img_path,'epoch {}'.format(epoch))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f.savefig(os.path.join( save_path, 'result_{}.png'.format(fig_name)))
    plt.close()

def savefignew(x,y, pred,fig_name,epoch):
    x=x.cpu().detach().numpy()
    y, pred = y.cpu().detach().numpy(), pred.cpu().detach().numpy()
    f, ax = plt.subplots(1, 2, figsize=(60, 20))
    noise = pred-y
    ax[0].imshow(noise,cmap=plt.cm.gray, vmin=-160, vmax=240)
    ax[0].set_title('noise', fontsize=30)
    ax[1].imshow(y,cmap=plt.cm.gray, vmin=-160, vmax=240)
    ax[1].set_title('NDCT', fontsize=30)
    save_path = os.path.join('image','test','redcnn(noise)','epoch {}'.format(epoch))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f.savefig(os.path.join( save_path, 'result_{}.png'.format(fig_name)))
    plt.close()


