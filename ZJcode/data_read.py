import os
import numpy as np
import scipy.io as sio


def load_data(data_dir, data_name, data_gaussian, is_training):
    if data_gaussian:
        data_name += '_gaussian'
    else:
        data_name += '_minmax'
    if is_training:
        data_name += '_train.mat'
    else:
        data_name += '_test.mat'
    return sio.loadmat(os.path.join(data_dir, data_name))


def shuffle(x, y):
    np.random.seed(4)
    randomorder = np.arange(len(y))
    np.random.shuffle(randomorder)
    x = x[randomorder]
    y = y[randomorder]
    return x, y


def get_xy(data_dir, data_name, is_training, data_gaussian, data_shuffle, batch_size, time_step, time_len):
    dataset = sio.loadmat(os.path.join(data_dir, data_name))
    x = dataset['traffic_flow'] 
    print("before normalization, traffic flow min =", np.min(x), ", max = ", np.max(x))
    
    v_max = np.amax(x)
    x = x/v_max   
    if data_gaussian:
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        x = (x - mean)/std
    print("after normalization, traffic flow min =", np.min(x), ", max = ", np.max(x))

    bs = batch_size   
    ts = time_step
    n_day = 60 // time_len * 24
    n_weekday = n_day * 5
    n_week = n_day * 7
    print(n_week, type(n_week))
    _x = []
    _y = []
    for i in range(0, len(x)-n_week+1, n_week):
        for j in range(i, i+n_weekday):
            _x.append(x[j:j+ts])
            _y.append(x[j+ts])
    _x = np.asarray(_x, dtype=np.float32)
    _y = np.asarray(_y, dtype=np.float32)
    print("_x.shape:", _x.shape)
    print("_y.shape:", _y.shape)

    train_number = np.int(len(_x)/10*8)
    print("train_number:", train_number)
    x_train = _x[:train_number]
    y_train = _y[:train_number]
    x_test = _x[train_number:]
    y_test = _y[train_number:]

    if data_shuffle:
        x_train, y_train = shuffle(x_train, y_train)

    n_station = _y.shape[-1]
    x_train = x_train[:len(x_train)-len(x_train)%bs]
    y_train = y_train[:len(y_train)-len(y_train)%bs]
    x_train = x_train.reshape(-1, bs, ts, n_station)
    y_train = y_train.reshape(-1, bs, n_station)

    print("x_train shape", x_train.shape, "y_train shape", y_train.shape)
    print("x_test shape", x_test.shape, "y_test shape", y_test.shape)
    return x_train, y_train, x_test, y_test, v_max


# def get_xy(data_dir, data_name, is_training, data_gaussian, data_shuffle, batch_size, time_step):
#     dataset = load_data(data_dir, data_name, data_gaussian, is_training)
#     x = dataset['traffic_flow'] 
#     print("data read, traffic flow min =", np.min(x))
#     if data_gaussian:
#         dataset = load_data(data_dir, data_name, False, is_training)
#         x_mm = dataset['traffic_flow'] 
#     bs = batch_size   
#     ts = time_step
#     n_weekday = 12*24*5
#     n_week = 12*24*7
#     _x = []
#     _y = []
#     for i in range(0, len(x)-n_week+1, n_week):
#         for j in range(i, i+n_weekday):
#             _x.append(x[j:j+ts])
#             if data_gaussian:
#                 _y.append(x_mm[j+ts])
#             else:
#                 _y.append(x[j+ts])
#     _x = np.asarray(_x, dtype=np.float32)
#     _y = np.asarray(_y, dtype=np.float32)
#     print("_x.shape:", _x.shape)
#     print("_y.shape:", _y.shape)
#     if is_training: 
#         if data_shuffle:
#             _x, _y = shuffle(_x, _y)
#         n_station = _y.shape[-1]
#         _x = _x[:len(_x)-len(_x)%bs]
#         _y = _y[:len(_y)-len(_y)%bs]
#         _x = _x.reshape(-1, bs, ts, n_station)
#         _y = _y.reshape(-1, bs, n_station)
#     vmax = dataset['vmax']
#     print("data read, traffic flow _x min =", np.min(_x), "max =", np.max(_x))
#     print("data read, traffic flow _y min =", np.min(_y), "max =", np.max(_y))
#     return _x, _y, vmax


# def generate_next(x, batch_size, time_step):
#     bs = batch_size
#     ts = time_step
#     n_weekday = 1440
#     n_week = 1728
#     _x = []
#     _y = []
#     _bs = 0
#     for i in range(0, len(x)-n_week+1, n_week):
#         for j in range(i, i+n_weekday):
#             _x.append(x[j:j+ts])
#             _y.append(x[j+ts])
#             _bs += 1
#             if _bs == bs:
#                 _x = np.array(_x)
#                 _y = np.array(_y)
#                 yield _x, _y
#                 _x = []
#                 _y = []