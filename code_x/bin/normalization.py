import click
import scipy.io as sio
import attr
import numpy as np
import yaml


class AbstractDataset:
    pass


@attr.s
class RawDataset(AbstractDataset):
    x = attr.ib()


@attr.s
class MinMaxNormalizedDataset(AbstractDataset):
    x = attr.ib()
    maxv = attr.ib()


@attr.s
class GaussianNormalizedDataset(MinMaxNormalizedDataset):
    x = attr.ib()
    maxv = attr.ib()
    mean = attr.ib()
    std = attr.ib()


def normalization(source, target, proc):
    data = proc(RawDataset(sio.loadmat(source)['traffic_flow']))
    sio.savemat(target, rename(attr.asdict(data)))


def minmax(data: RawDataset):
    x_max = np.amax(data.x)
    return MinMaxNormalizedDataset(data.x/x_max, x_max)


def gaussian(data: RawDataset):
    data = minmax(data)
    mean = np.mean(data.x, axis=0)
    std = np.std(data.x, axis=0)
    return GaussianNormalizedDataset(data.x - mean/std, mean, std, data.maxv)


def rename(data):
    return {NAME_MAP.get(k, k): v for k, v in data.items()}


METHODS = {
    'gaussian': gaussian,
    'minmax': minmax, }

NAME_MAP = {
    'x': 'traffic_flow',
    'maxv': 'x_max',
    'mean': 'x_mean',
    'std': 'x_std',
}


def load_config(path):
    with open(path, 'r') as fin:
        return yaml.load(fin)


@click.command()
@click.argument("source", type=click.Path(exists=True))
@click.argument("target", type=click.Path())
@click.argument("config", type=click.Path(exists=True))
def cli(source, target, config):
    """
    Normalization dataset.
    """
    cfg = load_config(config)
    normalization(source, target, METHODS[cfg['method'].lower()])


if __name__ == "__main__":
    cli()
