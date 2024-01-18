from typing import Tuple
import tensorflow as tf


def get_dataset(cfg) -> Tuple[tf.data.Dataset, None]:
    ds_ext = tf.keras.utils.image_dataset_from_directory(
        directory=cfg['ds_path'],
        labels='inferred',
        label_mode='int',
        color_mode='grayscale',
        shuffle=True,
        batch_size=cfg['batch_size'],
        image_size=(28, 28),
        interpolation='bilinear'
    )

    ds_ext = ds_ext.map(
            lambda x, t: (tf.cast(x, tf.float32)/255.0, t),
            num_parallel_calls=tf.data.AUTOTUNE
        ).cache(
        ).prefetch(
            tf.data.AUTOTUNE
        )

    return ds_ext, None
