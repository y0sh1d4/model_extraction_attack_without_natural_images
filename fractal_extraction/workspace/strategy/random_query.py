from .base import strategy_base
from .distillation import Distillation
import numpy as np
import os
import glob
import tempfile
import tensorflow as tf


class extraction_dataset:
    def update(self, query, response):
        if hasattr(self, 'query'):
            self.query = np.concatenate([self.query, query], axis=0)
            self.response = np.concatenate([self.response, response], axis=0)
        else:
            self.query = query
            self.response = response

    def dataset(self, cfg):
        return tf.data.Dataset.from_tensor_slices(
            (self.query, self.response)
            ).shuffle(
                len(self.query)
            ).batch(
                cfg['batch_size']
            ).prefetch(
                tf.data.AUTOTUNE
            )


class attack(strategy_base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ds_ext = iter(self.ds_ext)
        self.ds = extraction_dataset()

    def get_query(self):
        try:
            return next(self.ds_ext)[0]
        except StopIteration:
            return None

    def update(self, query, response):
        self.ds.update(query, response)
        model = self.train(self.model, self.ds.dataset(self.cfg))
        acc = self.test(model)
        self.model = model.layers[0]
        return acc

    # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    # にするとValidation Lossが高止まりしたのであまり良くないかと．
    def train(self, model, dataset):
        with tempfile.TemporaryDirectory() as d:
            # Prepare substitute model to distillate
            substitute_model = Distillation()
            substitute_model.add(model)
            substitute_model.compile(
                optimizer=tf.keras.optimizers.Adam(amsgrad=True),
                loss=tf.keras.losses.KLDivergence(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()]
            )

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.cfg['Early_stopping_patience'],
                    mode='min',
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(d, 'model_{epoch:03d}.h5'),
                    save_best_only=True
                )]

            _ = substitute_model.fit(
                dataset,
                validation_data=self.ds_test.map(
                        lambda x, t:
                        (x, tf.one_hot(t,
                                       depth=substitute_model.output_shape[1]))
                    ),
                epochs=self.cfg['epoch'],
                callbacks=callbacks
                )

            # Load best model
            model_file = sorted(
                glob.glob(
                    os.path.join(d, '*.h5')
                    )
                )[-1]
            substitute_model.load_weights(model_file)

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
