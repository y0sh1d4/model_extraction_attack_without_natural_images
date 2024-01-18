from .base import strategy_base
import numpy as np
import os
import glob
import tempfile
import tensorflow as tf


def generatorA(img_size, img_ch, num_z=100, num_ch=64):
    init_size = img_size//4
    tf.keras.Sequential([
        tf.keras.layers.Dense(num_z, num_)
    ])


class attack(strategy_base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_query(self):
        try:
            return next(self.ds_ext)[0]
        except StopIteration:
            return None

    def update(self, query, response):
        model = self.train(self.model, self.ds.dataset(self.cfg))
        acc = self.test(model)
        self.model = model.layers[0]
        return acc

    def train(self, model, dataset):
        return substitute_model

    def test(self, model):
        ret = model.evaluate(
            self.ds_test.map(
                lambda x, t:
                (x, tf.one_hot(t, depth=model.output_shape[1]))
                ),
            return_dict=True
            )

        return ret['categorical_accuracy'], ret['loss']
