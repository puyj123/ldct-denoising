import os.path

import numpy as np
import matplotlib.pyplot as plt


def save_image(image,index,epoch):
    to_image=image.detach().squeeze().cpu().numpy()
    image_path=os.path.join("image","epoch {}".format(epoch))
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    image_path = os.path.join(image_path, "result_{}.png".format(index))
    plt.imsave(image_path,to_image,cmap='gray')
