import train_helpers as th
import os

slide = 2
print('Slide parameter:' + str(slide))

in_model_path = None
if False:
    in_model_path = 'model/model_cat.h5'

def train_folder(folder, enhance):
    data = th.load_data(folder)
    images, angle_array, throttle_array = th.get_dataset(data, slide, enhance)
    th.train(images, angle_array, throttle_array, 'model/model_cat.h5', in_model_path)
    #images, angle_array, throttle_array = th.get_dataset(data, slide, enhance)
    #th.train(images, angle_array, throttle_array, 'model/model_cat.h5', 'model/model_cat.h5')

train_folder('data', None)

#in_model_path = 'model/model_cat.h5'
#train_folder('data/home1', None)
#train_folder('data/home1', 'shift')
#train_folder('data/home2', None)
#train_folder('data/home2', 'shift')
#train_folder('data/home3', None)
#train_folder('data/home3', 'shift')
#train_folder('data/tub_1_18-10-13', None)
#train_folder('data/tub_1_18-10-13', 'shift')
