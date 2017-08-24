from random import shuffle
import scipy.misc
import numpy as np

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return scipy.misc.imresize(scipy.misc.imread(image_path), (resize_w, resize_w)) / 127.5 - 1
