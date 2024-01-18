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


# Refer: https://keras.io/examples/vision/grad_cam/
def grad_cam(x: tf.Tensor, t: tf.Tensor,
             model: tf.keras.Model, last_conv_name: str) -> np.ndarray:

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_name).output, model.output])

    with tf.GradientTape() as tape:
        last_conv_out, y = grad_model(x)
        pre = y[:, tf.squeeze(t)]

    grad = tape.gradient(pre, last_conv_out)
    pgrad = tf.keras.layers.GlobalAveragePooling2D()(grad)
    heatmap = tf.matmul(last_conv_out, pgrad[..., tf.newaxis])
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.keras.activations.relu(heatmap) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


@hydra.main(config_path='conf', config_name='config')
def run(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg['global_params']['mlflow_path'])

    writer = mlflow_writer.MlflowWriter('gradient_visualization')
    writer.log_params_from_omegaconf_dict(cfg)

    # Prepare test dataset
    get_dataset = getattr(
            importlib.import_module(
                'datasets.'+cfg['victim']['task_type']
            ),
            'get_dataset'
        )
    tmp = cfg['victim']
    tmp['batch_size'] = 1
    _, ds_test = get_dataset(tmp)

    model = getattr(
            importlib.import_module(
                'models.'+cfg[cfg['gv']['target']]['task_type']
            ),
            cfg[cfg['gv']['target']]['model']
        )()
    if cfg['gv']['target'] == 'victim':
        model.load_weights(os.path.join(
                cfg['global_params']['victim_model_path'],
                cfg['victim']['model']+'.h5'
            ))
    else:
        model.load_weights(os.path.join(
                cfg['global_params']['substitute_model_path'],
                cfg['gv']['round']+'.h5'
            ))

    inputs = []
    grads = []
    for x, t in ds_test:
        ret = grad_cam(x, t, model, cfg['gv']['last_conv_name'])
        inputs.append(x.numpy())
        grads.append([ret])
        break

    inputs = np.concatenate(inputs, axis=0)
    grads = np.concatenate(grads, axis=0)
    return
    with tempfile.TemporaryDirectory() as d:
        np.savez(os.path.join(d, 'results.npz'), inputs=inputs, grads=grads)
        writer.log_artifact(os.path.join(d, 'results.npz'))

        fig = plt.figure()
        h = 4
        w = 10
        for i in range(1, h+1):
            for j in range(1, w+1):
                ax = fig.add_subplot(h, w, j+(i-1)*w)
                ax.imshow(inputs[j+(i-1)*w].reshape(28, 28, 1), 'gray')
                ax.set_xticklabels('')
                ax.set_yticklabels('')
        fig.tight_layout()
        fig.savefig(os.path.join(d, 'in.png'), dpi=300)
        writer.log_artifact(os.path.join(d, 'in.png'))

        fig = plt.figure()
        h = 4
        w = 10
        for i in range(1, h+1):
            for j in range(1, w+1):
                ax = fig.add_subplot(h, w, j+(i-1)*w)
                ax.imshow(grads[j+(i-1)*w], 'gray')
                ax.set_xticklabels('')
                ax.set_yticklabels('')
        fig.tight_layout()
        fig.savefig(os.path.join(d, 'grad.png'), dpi=300)
        writer.log_artifact(os.path.join(d, 'grad.png'))

    writer.set_terminated()


if __name__ == '__main__':
    run()
