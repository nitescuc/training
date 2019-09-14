#!/usr/bin/env python3

import container_support as cs
import tf_container.train_helpers as th
import os
import zipfile
import numpy as np
import math
import datetime

from tensorflow.python.client import device_lib

def train():
    env = cs.TrainingEnvironment()

    print(device_lib.list_local_devices())
    os.system('mkdir -p logs')

    # ### Loading the files ###
    # ** You need to copy all your files to the directory where you are runing this notebook into a folder named "data" **

    def unzip_file(root,f):
        zip_ref = zipfile.ZipFile(os.path.join(root,f), 'r')
        zip_ref.extractall(root)
        zip_ref.close()

    # unzip datasets
    for root, dirs, files in os.walk('/opt/ml/input/data/train'):
        for f in files: 
            if f.endswith('.zip'):
                unzip_file(root, f)

    # generate enhanced data
    root = '/opt/ml/input/data/train'
    if env.hyperparameters.get('enhance_image_count', False):
        th.generate_enhanced_dataset(root, '/opt/ml/input/data/train/enhanced', env.hyperparameters.get('enhance_image_count', False))
        root = '/opt/ml/input/data/train/enhanced'

    # decode parameters
    slide = 2
    if env.hyperparameters.get('slide', False):
        slide = env.hyperparameters.get('slide', False)
        print('Slide parameter:' + str(slide))

    in_model_path = None
    if env.hyperparameters.get('input_model', False):
        in_model_path = os.path.join('/opt/ml/input/data/train', env.hyperparameters.get('input_model', False))

    data = th.load_data(root)
    #out_pattern = '/opt/ml/model/model_cat_{epoch:02d}_{angle_out_loss:.2f}_{val_angle_out_loss:.2f}.h5'
    out_pattern = '/opt/ml/model/model-blur-2slide-' + datetime.datetime.now().strftime("%y%m%d_%H%M") + '.h5'

    # ### Start training ###
    if env.hyperparameters.get('use_generator'):
        print('Using generator')
        shift = env.hyperparameters.get('shift', False)
        if shift:
            print('Random shift: yes')
        rotate = env.hyperparameters.get('rotate', False)
        if rotate:
            print('Random rotate: yes')
        train_gen, val_gen, data_len = th.get_generators(data, slide, do_random_shift=shift, do_random_rotation=rotate)
        th.train_generator(train_gen, val_gen, data_len, out_pattern, in_model_path)
    else:
        images, angle_array, throttle_array = th.get_dataset(data, slide, env.hyperparameters.get('enhance', False))
        th.train(images, angle_array, throttle_array, out_pattern, in_model_path)