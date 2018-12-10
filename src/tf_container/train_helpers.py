import numpy as np
import math
import json
import re
import os

from keras.layers import Input, Dense, merge
from keras.models import Model
from keras.models import load_model
from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import callbacks
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import random_shift

#import cv2

#import Automold as am

numbers = re.compile(r'(\d+)')

def get_data(root,f):
    d = json.load(open(os.path.join(root,f)))
    if ('pilot/throttle' in d):
        return [d['user/mode'],d['user/throttle'],d['user/angle'],root,d['cam/image_array'],d['pilot/throttle'],d['pilot/angle']]
    else:
        return [d['user/mode'],d['user/throttle'],d['user/angle'],root,d['cam/image_array']]
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def load_data(rootDir):
    data = []
    for root, dirs, files in os.walk(rootDir):
        data.extend([get_data(root,f) for f in sorted(files, key=numericalSort) if f.startswith('record') and f.endswith('.json')])
    return data

def trim_img(img):
#    img[0:20] = 0
    return img

def to_discrete(a):
    a = a + 1
    b = round(a / (2/14))
    return int(b)

def get_dataset(data, slide, enhance):
    # Normalize / correct data
    #data = [d for d in data if d[1] > 0.1]

    # ### Loading throttle and angle ###
    angle = [d[2] for d in data]
    throttle = [d[1] for d in data]
    angle_array = np.array(angle)
    throttle_array = np.array(throttle)

    # detect model type (float or discrete)
    max_angle = np.max(angle_array)
    if (max_angle < 2):
        #convert to discrete
        angle_array = np.array([to_discrete(a) for a in angle_array])
        throttle_array = np.array([to_discrete(a) for a in throttle_array])

    # ### Loading images ###
    images = np.array([trim_img(img_to_array(load_img(os.path.join(d[3],d[4]), grayscale=True))) for d in data],'f')
    
    # image before parameters
    images = images[:len(images)-slide]
    angle_array = angle_array[slide:]
    throttle_array = throttle_array[slide:]

    #generate enhanced set
    if enhance == 'shift':
        print('============== Generating enhanced training set: shift ====================')
        images = np.array([random_shift(img, 0.1, 0.0, row_axis=0, col_axis=1, channel_axis=2) for img in images])

    #images = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(120, 160, 1) for img in images])
        
    return images, angle_array, throttle_array

def train(images, angle_array, throttle_array, out_model_path, in_model_path):
    def linear_bin(a):
        arr = np.zeros(15)
        arr[a] = 1
        return arr
    def throttle_bin(a):
        arr = np.zeros(7)
        arr[a - 8] = 1
        return arr

    logs = callbacks.TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True)
    save_best = callbacks.ModelCheckpoint(out_model_path, monitor='angle_out_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    early_stop = callbacks.EarlyStopping(monitor='angle_out_loss', 
                                                    min_delta=.0005, 
                                                    patience=10, 
                                                    verbose=1, 
                                                    mode='auto')
    img_in = Input(shape=(120, 160, 1), name='img_in')                      # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = img_in
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)       # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)       # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Convolution2D(64, (5,5), strides=(2,2), activation='relu')(x)       # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)       # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)       # 64 features, 3px3p kernal window, 1wx1h stride, relu

    # Possibly add MaxPooling (will make it less sensitive to position in image).  Camera angle fixed, so may not to be needed

    x = Flatten(name='flattened')(x)                                        # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)                                    # Classify the data into 100 features, make all negatives 0
    x = Dropout(.1)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(.1)(x)                                                      # Randomly drop out 10% of the neurons (Prevent overfitting)
    #categorical output of the angle
    callbacks_list = [save_best, early_stop, logs]
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)        # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0

    #continous output of throttle
    throttle_out = Dense(7, activation='softmax', name='throttle_out')(x)        # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0
    angle_cat_array = np.array([linear_bin(a) for a in angle_array])
    throttle_cat_array = np.array([throttle_bin(a) for a in throttle_array])
    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    model.compile(optimizer='adam',
                loss={'angle_out': 'categorical_crossentropy', 
                        'throttle_out': 'categorical_crossentropy'},
                loss_weights={'angle_out': 0.9, 'throttle_out': 0.9},
                metrics=["accuracy"])
    if in_model_path:
        model.load_weights(in_model_path)
        print('Using model ' + in_model_path)
    model.fit({'img_in':images},{'angle_out': angle_cat_array, 'throttle_out': throttle_cat_array}, batch_size=32, epochs=100, verbose=1, validation_split=0.2, shuffle=True, callbacks=callbacks_list)
