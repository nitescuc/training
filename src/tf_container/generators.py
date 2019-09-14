# Create generators
from sklearn.utils import shuffle
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import random_shift, random_rotation
import numpy as np
import cv2
import os

def linear_bin(a):
    arr = np.zeros(15)
    arr[int(a)] = 1
    return arr
def throttle_bin(a):
    arr = np.zeros(7)
    if a >= 8:
        arr[int(a) - 8] = 1
    return arr

def gen_batch(batch_size, in_data, blur=False, normalization=False, do_random_shift=False, do_random_rotation=False):
    batch_img = np.zeros((batch_size,120,160,1))
    batch_angle = np.zeros((batch_size,15))
    batch_throttle = np.zeros((batch_size,7))
    pointer = 0
    (sh_data)=shuffle(in_data)
    while True:
        i = 0
        while i < batch_size:
            #
            dd = sh_data[pointer]
            d_angle = dd[2]
            d_throttle = dd[1]
            #
            img = img_to_array(load_img(os.path.join(dd[3],dd[4]), grayscale=True))
            if blur:
                img = np.array(cv2.bilateralFilter(img,9,75,75), dtype='uint8').reshape(120,160,1)
            if do_random_shift:
                img = random_shift(img, 0.1, 0.0, row_axis=0, col_axis=1, channel_axis=2) 
            if do_random_rotation:
                img = random_rotation(img, do_random_rotation, row_axis=0, col_axis=1, channel_axis=2) 
            if normalization:
                img = img * 1.0/255.0
            batch_img[i] = img
            batch_angle[i] = linear_bin(d_angle)
            batch_throttle[i] = throttle_bin(d_throttle)
            i = i + 1
            if i >= batch_size: break
                            
            pointer+=1
            if pointer == len(in_data)-1: pointer = 0
        
        yield ({ 'img_in': batch_img }, { 'angle_out': batch_angle, 'throttle_out': batch_throttle })
