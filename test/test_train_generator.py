from tf_container import train_helpers as th

data = th.load_data('/Users/az02289/d2/data/home')

train_gen, val_gen, data_len = th.get_generators(data, 2)
th.train_generator(train_gen, val_gen, data_len, 'model_cat.h5', '')
