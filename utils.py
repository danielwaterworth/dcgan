from random import shuffle
import scipy.misc
import numpy as np

def get_image(image_path, resize_w):
    return scipy.misc.imresize(scipy.misc.imread(image_path), (resize_w, resize_w)) / 127.5 - 1
