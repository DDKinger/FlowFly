import os
import numpy as np

def load_data(data_dir, data_gaussian, is_training):
    if data_gaussian:
        data_name = 'TrafficFlow_69_12week_6day_gaussian'
    else:
        data_name = 'TrafficFlow_69_12week_6day_normal'
    if is_training:
        data_name += '_train.mat'
    else:
        data_name += '_test.mat'
    return sio.loadmat(os.path.join(data_dir, data_name))['traffic_flow']

def generate_next(x, batch_size, time_step):
    bs = batch_size
    ts = time_step
    n_weekday = 1440
    n_week = 1728
    _x = []
    _y = []
    _bs = 0
    for i in range(0, len(x)-n_week+1, n_week):
        for j in range(i, i+n_weekday-ts+1):
            _x.append(x[j:j+ts])
            _y.append(x[j+ts])
            _bs += 1
            if _bs == bs:
                _x = np.array(_x)
                _y = np.array(_y)
                yield _x, _y
                _x = []
                _y = []



# def shuffle(x):
#     np.random.seed(400)
#     randomorder = np.arange(len(x))
#     np.random.shuffle(randomorder)
#     x = x[randomorder]
#     return x