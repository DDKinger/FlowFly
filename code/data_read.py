import os

def load_data(data_dir, data_gaussian, is_training):
    if data_gaussian:
        data_name = 'TrafficFlow_69_gaussian'
    else:
        data_name = 'TrafficFlow_69_normal'
    if is_training:
        data_name += '_train.mat'
    else:
        data_name += '_test.mat'
    return sio.loadmat(os.path.join(data_dir, data_name))['traffic_flow']

def generate_next(x, batch_size, time_step):
    bs = batch_size
    ts = time_step
    for i in range(0, len(x)-bs, bs):
        yield x[]


# def shuffle(x):
#     np.random.seed(400)
#     randomorder = np.arange(len(x))
#     np.random.shuffle(randomorder)
#     x = x[randomorder]
#     return x