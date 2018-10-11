import click
import scipy.io as sio
import numpy as np
import yaml


def fields():
    return ('traffic_flow', 'x_max', 'x_mean', 'x_std')


def parse_partition(nb_samples, partition):
    if isinstance(partition, int):
        return {'train': partition, 'test': nb_samples-partition}
    if isinstance(partition, float):
        result = {'train': int(nb_samples * partition)}
        result['test'] = nb_samples - result['train']
        return result


def load_data(path):
    data = sio.loadmat(path)
    return {k: v for k, v in data.items() if k in fields()}


def slice_of_dataset_type(partition, dataset_type):
    start = 0
    for k in partition:
        if k == dataset_type:
            return slice(start, start+partition[k])
        start += partition[k]
    raise KeyError(
        f"dataset type {dataset_type} not found in partition {partition}.")


def split(source, target, partition, dataset_type):
    data = load_data(source)
    x = data['traffic_flow']
    sliced = x[slice_of_dataset_type(
        parse_partition(x.shape[0], partition), dataset_type)]
    result = {k: v if k != 'traffic_flow' else sliced for k, v in data.items()}
    sio.savemat(target, result)


def load_config(path):
    with open(path, 'r') as fin:
        return yaml.load(fin)


@click.command()
@click.argument("source", type=click.Path(exists=True))
@click.argument("target", type=click.Path())
@click.option("--dataset", "-d", type=str)
@click.option("--config", "-c", type=click.Path(exists=True))
def cli(source, target, dataset, config):
    """
    Normalization dataset.
    """
    cfg = load_config(config)
    split(source, target, cfg['partition'], dataset)


if __name__ == "__main__":
    cli()