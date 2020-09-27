import train_helpers as th
import os
import datetime

slide = 2
print('Slide parameter:' + str(slide))

in_model_path = None
if False:
    in_model_path = 'model/model_cat.h5'
break_range = 15
model_name = 'model_cat'
crop = 100
clahe = True
shift = True

def train_folder(folder, enhance):
    # load data
    data = th.load_data(folder, break_range)
    #out_pattern = '/opt/ml/model/model_cat_{epoch:02d}_{angle_out_loss:.2f}_{val_angle_out_loss:.2f}.h5'
    options = [model_name, "blur", str(slide) + "slide"]
    if crop:
        options.append("crop" + str(crop))
    if clahe:
        options.append("clahe")
    options.append(datetime.datetime.now().strftime("%y%m%d_%H%M"))
    out_pattern = '/Users/az02289/d2/model/' + '-'.join(options) + '.h5'

    images, angle_array, throttle_array = th.get_dataset(data, slide, enhance)
    th.train(images, angle_array, throttle_array, out_pattern, in_model_path)

train_folder('/Users/az02289/d2/data', 'shift')
