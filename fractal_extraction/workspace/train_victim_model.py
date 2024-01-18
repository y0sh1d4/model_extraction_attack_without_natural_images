import os
import glob
import shutil
import tempfile
import importlib

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn
import matplotlib.pyplot as plt
import hydra
import mlflow
from omegaconf import DictConfig

import mlflow_writer


@hydra.main(config_path='conf', config_name='config')
def run(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg['global_params']['mlflow_path'])

    writer = mlflow_writer.MlflowWriter('train_victim_model')
    writer.log_params_from_omegaconf_dict(cfg['global_params'])
    writer.log_params_from_omegaconf_dict(cfg['victim'])

    get_dataset = getattr(
            importlib.import_module(
                'datasets.'+cfg['victim']['task_type']
            ),
            'get_dataset'
        )
    ds_train, ds_test = get_dataset(cfg['victim'])

    #########
    # Train #
    #########
    model = getattr(
            importlib.import_module(
                'models.'+cfg['victim']['task_type']
            ),
            cfg['victim']['model']
        )()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    with tempfile.TemporaryDirectory() as d:
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(d, 'model_{epoch:03d}.h5'),
                save_best_only=True
            )]

        history = model.fit(
            ds_train,
            validation_data=ds_test,
            epochs=cfg['victim']['epoch'],
            callbacks=callbacks
        )

        os.makedirs(os.path.join(
            cfg['global_params']['victim_model_path']), exist_ok=True)
        model_file = sorted(
            glob.glob(
                os.path.join(d, '*.h5')
                )
            )[-1]
        shutil.copyfile(
                model_file,
                os.path.join(
                    cfg['global_params']['victim_model_path'],
                    cfg['victim']['model']+'.h5'
                )
            )
        writer.log_artifact(os.path.join(
                cfg['global_params']['victim_model_path'],
                cfg['victim']['model']+'.h5'
            ))

    writer.log_history(history)

    ########
    # Test #
    ########
    model = getattr(
            importlib.import_module(
                'models.'+cfg['victim']['task_type']
            ),
            cfg['victim']['model']
        )()
    model.load_weights(os.path.join(
            cfg['global_params']['victim_model_path'],
            cfg['victim']['model']+'.h5'
        ))

    predicts = model.predict(ds_test).argmax(axis=1)
    labels = np.concatenate([v[1].numpy() for v in ds_test], axis=0)
    confmat = confusion_matrix(labels, predicts)
    accuracy = accuracy_score(labels, predicts)
    writer.log_metric('accuracy', accuracy)
    with tempfile.TemporaryDirectory() as d:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        seaborn.heatmap(confmat, ax=ax, cmap='Blues')
        ax.set_title('Classification accuracy: {}'.format(accuracy))
        plt.show()
        fig.savefig(
            os.path.join(d, 'confmat.png'),
            dpi=300,
            bbox_inches='tight',
            pad_inches=0
            )
        writer.log_artifact(os.path.join(d, 'confmat.png'))


if __name__ == '__main__':
    run()
