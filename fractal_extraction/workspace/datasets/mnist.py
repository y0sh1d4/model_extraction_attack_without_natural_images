import os
from typing import Tuple

import tensorflow as tf
import tensorflow_datasets as tfds


def preprocess(x, t) -> Tuple[tf.Tensor, tf.Tensor]:
    x = tf.cast(x, tf.float32)/255.0
    return x, t


def get_dataset(cfg) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    ds_train, ds_test = tfds.load(
            cfg['task_name'],
            split=['train', 'test'],
            shuffle_files=False,
            as_supervised=True,
            data_dir=os.path.join(cfg['ds_path'], cfg['task_name']),
        )

    ds_train = ds_train.map(
            preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        ).cache(
        ).shuffle(
            len(ds_train)
        ).batch(
            cfg['batch_size']
        ).prefetch(
            tf.data.AUTOTUNE
        )

    ds_test = ds_test.map(
            preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        ).cache(
        ).batch(
            cfg['batch_size']
        ).prefetch(
            tf.data.AUTOTUNE
        )

    return ds_train, ds_test
