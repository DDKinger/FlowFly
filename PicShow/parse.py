import tensorflow as tf
import itertools
import os

target_dir = "./summary/train/"
path = os.listdir(target_dir)[0]

target_tag = "accuracy_1"

def find_first_event(directory):
    result = list(filter(lambda fn: fn.startswith("event"), os.listdir(directory)))
    if len(result) > 0:
        return result[0]
    else:
        raise ValueError("No event file found")

def fetch_tag(path, target_tag, *islice_args):
    return [
        (e.step, v.simple_value)
        for e in itertools.islice(tf.train.summary_iterator(target_dir + path), *islice_args)
        for v in e.summary.value
        if v.tag == target_tag
    ]

# Fetch first 10 event from event file with given path, with `target_tag`
print(fetch_tag(find_first_event(target_dir), target_tag, 10))
