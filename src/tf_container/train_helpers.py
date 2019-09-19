import numpy as np
import math
import json
import re
import os
import random
import json 
import cv2

from tf_container.NVidia import NVidia
from keras import callbacks
from keras.preprocessing.image import load_img, img_to_array, save_img
from keras.preprocessing.image import random_shift
from sklearn.model_selection import train_test_split

from tf_container.generators import gen_batch

numbers = re.compile(r'(\d+)')

def get_data(root,f):
    d = json.load(open(os.path.join(root,f)))
    if d['user/angle'] == None:
        d['user/angle'] = 0
    return ['user',d['user/throttle'],d['user/angle'],root,d['cam/image_array'], None]
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def load_data(rootDir):
    data = []
    for root, dirs, files in os.walk(rootDir):
        data.extend([get_data(root,f) for f in sorted(files, key=numericalSort) if f.startswith('record') and f.endswith('.json')])

    data = np.array(data)
    angles = np.array(data[:,2], dtype='float32')
    if np.max(angles) < 2:
        print('load_data mapping to discrete')
        data = [to_discrete_data(d) for d in data]
    data = np.array(data)

    remap_throttle(data)

    return data


def trim_img(img):
#    img[0:20] = 0
    return img

def to_discrete(a):
    a = a + 1
    b = round(a / (2/14))
    return int(b)
def to_discrete_data(d):
    d[2] = to_discrete(float(d[2]))
    d[1] = to_discrete(float(d[1]))
    return d

def linear_bin(a):
    arr = np.zeros(15)
    arr[int(a)] = 1
    return arr
def throttle_bin(a):
    arr = np.zeros(7)
    if int(a) >= 8:
        arr[int(a) - 8] = 1
    return arr


def remap_throttle(data):
    break_range = 10
    break_throttle = 9
    slow_throttle = 10
    medium_throttle = 12
    high_throttle = 14

    data[:,1] = slow_throttle
    throttle_array = np.copy(np.array(data[:,1], dtype='float32'))
    angle_array = np.array(data[:,2], dtype='float32')
    start_idx = 0
    end_idx = 0
    start_range = False
    for a_idx in range(0, len(angle_array)):
        val = angle_array[a_idx]
        if a_idx < len(angle_array)-3 and throttle_array[a_idx] < 6 and throttle_array[a_idx+1] < 6 and throttle_array[a_idx+2] < 6:
            start_range = False
            for idx in range(a_idx, min(a_idx+10, len(angle_array))):
                data[idx,1] = break_throttle

        if val >= 5 and val <= 10:
            if not start_range: 
                start_range = True
                start_idx = a_idx
            end_idx = a_idx
        else:
            if (end_idx - start_idx) > 80:
                #print('Long line: ' + str(start_idx) + ',' + str(end_idx))
                for idx in range(start_idx, end_idx - break_range):
                    data[idx,1] = high_throttle
            elif (end_idx - start_idx) > 25:
                #print('Short line: ' + str(start_idx) + ',' + str(end_idx))
                for idx in range(start_idx, end_idx - break_range):
                    data[idx,1] = medium_throttle

            start_idx = 0
            end_idx = 0
            start_range = False
    

def generate_enhanced_dataset(root, destination, images_count):
    print('Generate enhanced dataset')
    do_random_shift = True
    blur = True
    rnd_lines = True
    crop = False

    data = load_data(root)

    # load images
    for idx in range(0, len(data)):
        dd = data[idx]
        img = img_to_array(load_img(os.path.join(dd[3],dd[4]), grayscale=True))
        data[idx,5] = img

    try:
        os.mkdir(destination)
    except:
        pass

    cnt = 0
    # Generate enhanced images
    while cnt < images_count:
        for idx in range(0, len(data)):
            img = data[idx, 5]
            if do_random_shift:
                img = random_shift(img, 0.1, 0.0, row_axis=0, col_axis=1, channel_axis=2) 
            if rnd_lines:
                if random.randint(0,3) == 3:
                    lines_img = np.zeros((120,160), dtype='float32')
                    rnd1 = random.randint(0, 120)
                    rnd2 = random.randint(0, 120)
                    cv2.line(lines_img,(0,rnd1+40),(160,rnd2+40),(256,256,256),2, cv2.LINE_AA)
                    cv2.line(lines_img,(0,rnd1),(160,rnd2),(256,256,256),2, cv2.LINE_AA)
                    cv2.line(lines_img,(0,rnd1-40),(160,rnd2-40),(256,256,256),2, cv2.LINE_AA)

                    img = np.array(cv2.addWeighted(img,0.5,lines_img,0.10,0, dtype=1), dtype='uint8')
            # last transformation
            if blur:
                img = np.array(cv2.bilateralFilter(img,9,75,75), dtype='uint8').reshape(120,160,1)
            # step3
            if crop:
                img[95:120] = 0

            # save image
            img_filename = '{:08d}_cam-image_array_.jpg'.format(cnt)
            save_img(os.path.join(destination, img_filename), img)
            # save data
            ff = open(os.path.join(destination, 'record_{:08d}.json'.format(cnt)), "w")
            json.dump({ "user/mode": "user", "cam/image_array": img_filename, "user/throttle": data[idx, 1], "user/angle": data[idx, 2] }, ff)
            ff.close()

            cnt = cnt + 1
            if cnt > images_count:
                break
    print('Generate enhanced dataset done')


def get_dataset(data, slide, enhance):
    ### Loading throttle and angle ###
    angle = [d[2] for d in data]
    throttle = [d[1] for d in data]
    angle_array = np.array(angle)
    throttle_array = np.array(throttle)

    ### Loading images ###
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

def slide_data(data, slide):
    shifted_data = data
    thr = np.roll(np.array(data)[:,1], (-1)*slide)
    ang = np.roll(np.array(data)[:,2], (-1)*slide)
    for i in range(0,len(shifted_data)):
        shifted_data[i][2] = ang[i]
        shifted_data[i][1] = thr[i]
    return shifted_data

def get_generators(data, slide, do_random_shift=False, do_random_rotation=False, blur=False):
    data = slide_data(data, slide)
    split_train, split_val = train_test_split(data, test_size = 0.10, random_state = 100)
    return gen_batch(32, split_train, do_random_shift=do_random_shift, do_random_rotation=do_random_rotation, blur=blur), gen_batch(32, split_val, False, False), len(split_train)

def get_model(in_model_path):
    model = NVidia()
    model.compile(optimizer='adam',
                loss={'angle_out': 'categorical_crossentropy', 
                        'throttle_out': 'categorical_crossentropy'},
                loss_weights={'angle_out': 0.9, 'throttle_out': 0.9},
                metrics=["accuracy"])
    if in_model_path:
        model.load_weights(in_model_path)
        print('Using model ' + in_model_path)
    return model

def get_callbacks_list(out_model_path):
    logs = callbacks.TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True)
    save_best = callbacks.ModelCheckpoint(out_model_path, monitor='angle_out_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    early_stop = callbacks.EarlyStopping(monitor='angle_out_loss', 
                                                    min_delta=.0005, 
                                                    patience=20, 
                                                    verbose=1, 
                                                    mode='auto')
    #categorical output of the angle
    #callbacks_list = [save_best, early_stop, logs]
    callbacks_list = [save_best, early_stop]
    return callbacks_list

def train(images, angle_array, throttle_array, out_model_path, in_model_path):
    angle_cat_array = np.array([linear_bin(a) for a in angle_array])
    throttle_cat_array = np.array([throttle_bin(a) for a in throttle_array])
    callbacks_list = get_callbacks_list(out_model_path)
    model = get_model(in_model_path)
    model.fit({'img_in':images},{'angle_out': angle_cat_array, 'throttle_out': throttle_cat_array}, batch_size=32, epochs=100, verbose=1, 
        validation_split=0.2, shuffle=True, callbacks=callbacks_list)

def train_generator(train_gen, val_gen, data_len, out_model_path, in_model_path):
    callbacks_list = get_callbacks_list(out_model_path)
    model = get_model(in_model_path)
    model.fit_generator(train_gen, steps_per_epoch = data_len//32, epochs=200, validation_data=val_gen, 
                    verbose=1, validation_steps = 50, callbacks=callbacks_list)
