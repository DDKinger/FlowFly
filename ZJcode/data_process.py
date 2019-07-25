import os
import click
import numpy as np
import scipy.io as sio
import attr


class AbstractDataset:
    pass


@attr.s
class RawDataset(AbstractDataset):
    x = attr.ib()


@attr.s
class MinMaxNormalizedDataset(AbstractDataset):
    x = attr.ib()
    vmax = attr.ib()


@attr.s
class GaussianNormalizedDataset(MinMaxNormalizedDataset):
    x = attr.ib()
    vmax = attr.ib()
    vmean = attr.ib()
    vstd = attr.ib()
    

def minmax(data:RawDataset, dataset, target, num_train):
    x_max = np.amax(data.x)
    data = MinMaxNormalizedDataset(data.x/x_max, x_max)
    # sio.savemat(os.path.join(target, dataset+'_minmax'), rename(attr.asdict(data)))
    data_train = MinMaxNormalizedDataset(data.x[:num_train], x_max)
    sio.savemat(os.path.join(target, dataset+'_minmax_train'), rename(attr.asdict(data_train)))
    data_test = MinMaxNormalizedDataset(data.x[num_train:], x_max)
    sio.savemat(os.path.join(target, dataset+'_minmax_test'), rename(attr.asdict(data_test))) 
    return data


def gaussian(data:RawDataset, dataset, target, num_train):
    data = minmax(data, dataset, target, num_train)
    mean = np.mean(data.x, axis=0)
    std = np.std(data.x, axis=0)
    data = GaussianNormalizedDataset((data.x - mean)/std, data.vmax, mean, std)
    # sio.savemat(os.path.join(target, dataset+'_gaussian'), rename(attr.asdict(data)))
    data_train = GaussianNormalizedDataset(data.x[:num_train], data.vmax, mean, std)
    sio.savemat(os.path.join(target, dataset+'_gaussian_train'), rename(attr.asdict(data_train)))
    data_test = GaussianNormalizedDataset(data.x[num_train:], data.vmax, mean, std)
    sio.savemat(os.path.join(target, dataset+'_gaussian_test'), rename(attr.asdict(data_test))) 
    return data

def rename(data):
    return {NAME_MAP.get(k, k): v for k, v in data.items()}


NAME_MAP = {
    'x': 'traffic_flow'
}


@click.command()
@click.option('--data_dir', default=None, type=str, required=True)
@click.option('--dataset', default=None, type=str, required=True)
@click.option('--output_dir', default='./', type=str)
# @click.option('--method', type=str, default='minmax', help='choose among minmax and gaussian')
@click.option('--num_train', default=None, type=int, required=True) # for 12week_6day, be 17280
def normalization(data_dir, dataset, output_dir, num_train):
    # if proc.lower() != 'minmax' and proc.lower() != 'gaussian':
    #     raise ValueError("Unsupported method, choose from {'minmax', 'gaussian'}.")
    gaussian(RawDataset(sio.loadmat(os.path.join(data_dir, dataset))['traffic_flow']), dataset, output_dir, num_train)


if __name__ == "__main__":
    normalization()
