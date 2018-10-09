import scipy.io as sio
import abc
from functools import partial

# MAXIMUM_FLOW = 1500
NUM_TRAIN = 14400


class AbstractDataset(abc.ABC):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return self.x

    @abc.abstractmethod
    def fmap(self, f):
        pass

    def __len__(self):
        return len(self.x)

    @abc.abstractmethod
    def to_dict(self):
        pass

    # @abc.abstractmethod
    # def denormalization(self):
        # pass

class RawDataset(AbstractDataset):
    def __init__(self, x):
        super().__init__(x)

    def fmap(self, f):
        return RawDataset(*f(x))

    def to_dict(self):
        return {'traffic_flow': self.x}


class MinMaxNormalizedDataset(AbstractDataset):
    def __init__(self, x, x_max):
        super().__init__(x)
        self.x_max = x_max

    def fmap(self, f):
        return MinMaxNormalizedDataset(*f(self.x, self.y), self.x_max)

    def to_dict(self):
        return {'traffic_flow': self.x, 'x_max': self.x_max}


class GaussianNormalizedDataset(AbstractDataset):
    def __init__(self, x, x_mean, x_std, x_max):
        super().__init__(x)
        self.x_mean = x_mean
        self.x_std = x_std
        self.x_max = x_max

    def fmap(self, f):
        return GaussianNormalizedDataset(*f(self.x), self.x_mean, self.x_std, self.x_max)

    def to_dict(self):
        return {
            'x_input': self.x,
            'x_max': self.x_max,
            'x_mean': self.x_mean,
            'x_std': self.x_std,
        }


# def shuffle(x):
#     np.random.seed(400)
#     randomorder = np.arange(len(x))
#     np.random.shuffle(randomorder)
#     x = x[randomorder]
#     return x


def load_data(path):
    data = sio.loadmat(f"{path}.mat")
    return RawDataset(data['traffic_flow'])


def normalization(prefix, processing, is_training):
    full_path = prefix + suffix(processing, is_training)
    if os.path.exists(full_path):
        return load_data(full_path)
    data = processing(load_data(prefix), is_training)
    sio.savemat(full_path, data.to_dict)
    return data


def minmax_normal(data:RawDataset, num_train=NUM_TRAIN, is_training):
    if is_training:
        x_max = np.amax(data.x)
    else:
        x_max = 
    return MinMaxNormalizedDataset(data.x/x_max, x_max)


def gaussian_normal(data:RawDataset, num_train=NUM_TRAIN, is_training):
    data = minmax_normal(data, num_train)
    x_mean = np.mean(data.x[:num_train, :], axis=0)
    x_std = np.std(data.y[:num_train, :], axis=0)
    return GaussianNormalizedDataset(whiten(data.x, x_mean, x_std),
                                     x_mean, x_std, data.x_max)


def whiten(x, mean, std):
    return (x-mean)/std


def suffix(func, is_training):
    if is_training:
        sur = '_train'
    else:
        sur = '_test'
    if func is minmax_normal:
        return sur+'_normal'
    if func is gaussian_normal:
        return sur+'_gaussian'
    raise ValueError(f"Not valid function: {func}")


# def split_dataset(x, y, dataset_type, num_train=NUM_TRAIN):
#     if dataset_type.lower() == 'train':
#         return x[:num_train, :], y[:num_train, :]
#     if dataset_type.lower() == 'test':
#         return x[num_train:, :], y[num_train:, :]
#     raise ValueError(
#         f"Invalid dataset type {dataset_type}, expected train or test.")


def generate_data(prefix, shuffle=True, gaussian=False, is_training):
    if gaussian:
        data = normalization(prefix, gaussian_normal, is_training)
    else:
        data = normalization(prefix, minmax_normal, is_training)

    if is_training:
        data = data.fmap(partial(split_dataset, dataset_type='train'))
        if shuffle:
            data = data.fmap(shuffle)
        print("train data generated")
    else:
        data = data.fmap(partial(split_dataset, dataset_type='test'))
        print("test data generated")

    print("data size:", len(data))
    return data


###
def generate_normal(dataset):
    dataset_normal = dataset+'_normal.mat'
    if not os.path.exists(dataset_normal):
        x_input = sio.loadmat(dataset+'.mat')['x_input']
        y_output = sio.loadmat(dataset)['y_output']
        x_max = np.amax(x_input)
        x_input /= x_max
        y_output /= x_max
        sio.savemat(dataset_x_max, {'x_input': x_input,
                                    'y_output': y_output, 'x_max': x_max})
    else:
        x_input = sio.loadmat(dataset_normal)['x_input']
        y_output = sio.loadmat(dataset_normal)['y_output']
        x_max = sio.loadmat(dataset_x_max)['x_max']
    return x_input, y_output, x_max


def generate_gaussian():
    dataset_gaussian = dataset+'_gaussian.mat'
    if not os.path.exists(dataset_gaussian):
        x_input, y_output, x_max = generate_normal(dataset)
        x_mean = np.mean(x_input[:num_train, :], axis=0)
        x_std = np.std(y_output[:num_train, :], axis=0)
        x_input -= x_mean
        x_input /= x_std
        y_output -= x_mean
        y_output /= x_std
        sio.savemat(dataset_gaussian, {'x_input': x_input, 'y_output': y_output,
                                       'x_max': x_max, 'x_mean': x_mean, 'x_std': x_std})
    else:
        x_input = sio.loadmat(dataset_gaussian)['x_input']
        y_output = sio.loadmat(dataset_gaussian)['y_output']
        x_max = sio.loadmat(dataset_gaussian)['x_max']
        x_mean = sio.loadmat(dataset_gaussian)['x_mean']
        x_std = sio.loadmat(dataset_gaussian)['x_std']
    return x_input, y_output, x_max, x_mean, x_std


def generate_data(dataset, shuffle=True, gaussian=False, is_training):
    if gaussian:
        x_input, y_output, x_max, x_mean, x_std = generate_gaussian(dataset)
    else:
        x_input, y_output, x_max = generate_normal(dataset)
        x_mean = 0
        x_std = 1

    if is_training:
        x = x_input[:num_train, :]
        y = y_output[:num_train, :]
        if shuffle:
            np.random.seed(400)
            randomorder = np.arange(len(x))
            np.random.shuffle(randomorder)
            x = x[randomorder]
            y = y[randomorder]
        print("train data generated")
    else:
        x = x_input[num_train:, :]
        y = y_output[num_train:, :]
        print("test data generated")

    print("data size:", len(x))
    return x, y, x_max, x_mean, x_std
