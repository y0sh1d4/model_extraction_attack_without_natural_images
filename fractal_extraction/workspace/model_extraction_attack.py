import os
import tempfile
import importlib

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import hydra
import mlflow
from omegaconf import DictConfig

import mlflow_writer


@hydra.main(config_path='conf', config_name='config')
def run(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg['global_params']['mlflow_path'])

    writer = mlflow_writer.MlflowWriter('model_extraction_attack')
    writer.log_params_from_omegaconf_dict(cfg)

    # Prepare extraction dataset
    get_dataset = getattr(
            importlib.import_module(
                'datasets.'+cfg['attack']['task_type']
            ),
            'get_dataset'
        )
    tmp = cfg['attack']
    tmp['batch_size'] = tmp['query_size']
    ds_ext, _ = get_dataset(tmp)

    # Prepare test dataset
    get_dataset = getattr(
            importlib.import_module(
                'datasets.'+cfg['victim']['task_type']
            ),
            'get_dataset'
        )
    _, ds_test = get_dataset(cfg['victim'])

    victim_model = getattr(
            importlib.import_module(
                'models.'+cfg['victim']['task_type']
            ),
            cfg['victim']['model']
        )()
    victim_model.load_weights(os.path.join(
            cfg['global_params']['victim_model_path'],
            cfg['victim']['model']+'.h5'
        ))

    substitute_model = getattr(
            importlib.import_module(
                'models.'+cfg['victim']['task_type']
            ),
            cfg['attack']['model']
        )()

    # Attack
    attack = getattr(
            importlib.import_module(
                'strategy.'+cfg['attack']['strategy']
            ),
            'attack'
        )(ds_test, ds_ext, substitute_model, cfg['attack'])

    accuracy = []
    loss = []
    os.makedirs(cfg['global_params']['substitute_model_path'], exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        for i in range(cfg['attack']['round']):
            query = attack.get_query()
            if query is None:
                # Query is exhaust
                break

            response = tf.keras.activations.softmax(
                victim_model(query, training=False))
            acc, lo = attack.update(query, response)

            attack.model.save_weights(
                os.path.join(
                    cfg['global_params']['substitute_model_path'],
                    '{}.h5'.format(i))
                    )
            writer.log_artifact(
                os.path.join(
                    cfg['global_params']['substitute_model_path'],
                    '{}.h5'.format(i)
                    )
                )

            accuracy.append(acc)
            loss.append(lo)
            writer.log_metric('accuracy', acc, step=i)
            writer.log_metric('loss', lo, step=i)
            writer.log_metric('round', i)

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(accuracy)
        fig.savefig(
            os.path.join(d, 'accuracy.png'),
            dpi=300,
            bbox_inches='tight',
            pad_inches=0
            )
        np.save(os.path.join(d, 'accuracy.npy'), accuracy)
        writer.log_artifact(os.path.join(d, 'accuracy.png'))
        writer.log_artifact(os.path.join(d, 'accuracy.npy'))

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(loss)
        fig.savefig(
            os.path.join(d, 'loss.png'),
            dpi=300,
            bbox_inches='tight',
            pad_inches=0
            )
        np.save(os.path.join(d, 'loss.npy'), loss)
        writer.log_artifact(os.path.join(d, 'loss.png'))
        writer.log_artifact(os.path.join(d, 'loss.npy'))

    writer.set_terminated()


if __name__ == '__main__':
    run()
