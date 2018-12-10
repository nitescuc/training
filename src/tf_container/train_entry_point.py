#!/usr/bin/env python3

import container_support as cs
import tf_container.train_helpers as th
import os
import zipfile
import numpy as np
import math

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

    slide = 2
    if env.hyperparameters.get('slide', False):
        slide = env.hyperparameters.get('slide', False)
        print('Slide parameter:' + str(slide))

    data = th.load_data('/opt/ml/input/data/train')
    images, angle_array, throttle_array = th.get_dataset(data, slide, env.hyperparameters.get('enhance', False))

    # ### Start training ###
    in_model_path = None
    if env.hyperparameters.get('input_model', False):
        in_model_path = os.path.join('/opt/ml/input/data/train', env.hyperparameters.get('input_model', False))
    th.train(images, angle_array, throttle_array, '/opt/ml/model/model_cat_{epoch:02d}_{angle_out_loss:.2f}_{val_angle_out_loss:.2f}.h5', in_model_path)