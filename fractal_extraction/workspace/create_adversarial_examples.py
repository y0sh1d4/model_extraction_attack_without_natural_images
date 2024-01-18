import os
import tempfile
import importlib
import pickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import hydra
import mlflow
from omegaconf import DictConfig
import foolbox

import mlflow_writer


def create_AEs(model, ds_test, attack_method, epsilons):
    sm = foolbox.TensorFlowModel(model, bounds=(0, 1))
    res = {'raw': [], 'clipped': [], 'is_adv': [], 'label': []}
    # Create AEs
    for batch in ds_test:
        _, clipped, is_adv = attack_method(sm, batch[0],
                                           batch[1], epsilons=epsilons)
        res['raw'].append([batch[0]]*len(epsilons))
        res['clipped'].append(clipped)
        res['is_adv'].append(is_adv)
        res['label'].append([batch[1]]*len(epsilons))

    # Aggregate
    result = []
    for i in range(len(epsilons)):
        tmp = {}
        for k in res.keys():
            tmp[k] = np.concatenate(
                [res[k][v][i].numpy() for v in range(len(res[k]))], axis=0)
        result.append(tmp)

    return result


# Victim modelが正解できるサンプルだけを抽出する
def dataset_screening(
    victim_model: tf.keras.Model,
    dataset: tf.data.Dataset
) -> tf.data.Dataset:

    correct_x = []
    correct_t = []
    for batch in dataset:
        x, t = batch
        t = t.numpy()
        y = np.argmax(victim_model(x, training=False).numpy(), axis=1)
        idx = (t == y)
        correct_x.append(x.numpy()[idx])
        correct_t.append(t[idx])
    correct_x = np.concatenate(correct_x, axis=0)
    correct_t = np.concatenate(correct_t, axis=0)

    return tf.data.Dataset.from_tensor_slices((correct_x, correct_t))


def eval_AEs(victim_model, AEs):
    y = []
    for batch in AEs:
        y.append(victim_model(batch[0], training=False).numpy())
    y = np.concatenate(y, axis=0)
    return y


@hydra.main(config_path='conf', config_name='config')
def run(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg['global_params']['mlflow_path'])

    writer = mlflow_writer.MlflowWriter('create_AEs_PGD')
    writer.log_params_from_omegaconf_dict(cfg)

    # Prepare test dataset
    get_dataset = getattr(
            importlib.import_module(
                'datasets.'+cfg['victim']['task_type']
            ),
            'get_dataset'
        )
    _, ds_test = get_dataset(cfg['victim'])

    # Load Victim model
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

    # Datasetのスクリーニング
    ds_test = dataset_screening(
        victim_model,
        ds_test
    )

    if len(ds_test) < cfg['create_AEs']['num_samples']:
        print("# of candidates for AEs is lower than create_AEs.num_samples")
        return

    ds_test = ds_test.take(
        cfg['create_AEs']['num_samples']
        ).batch(
            cfg['create_AEs']['batch_size']
        )

    # AEs生成
    eps = [eval(v) for v in cfg['create_AEs']['epsilons']]

    success_rate = {}
    transferability = {}
    for i in eval(cfg['create_AEs']['round']):
        if not os.path.exists(
            os.path.join(
                cfg['global_params']['substitute_model_path'], str(i)+'.h5')
        ):
            break

        # Load Substitute model
        substitute_model = getattr(
                importlib.import_module(
                    'models.'+cfg['victim']['task_type']
                ),
                cfg['attack']['model']
            )()

        substitute_model.load_weights(os.path.join(
                cfg['global_params']['substitute_model_path'],
                str(i)+'.h5'
            ))

        sm = tf.keras.Sequential()
        sm.add(substitute_model)
        sm.add(tf.keras.layers.Softmax())

        attack_method = foolbox.attacks.PGD(
            abs_stepsize=1/255.0, random_start=True)
        # attack_method = foolbox.attacks.LinfDeepFoolAttack()  # 微妙
        # attack_method = foolbox.attacks.FGSM()  # 結構調子いいけどPGDほどでは？計算めちゃはや
        # attack_method = foolbox.attacks.L2CarliniWagnerAttack()  # ほとんど攻撃成功せず
        tmp = create_AEs(sm, ds_test, attack_method, eps)

        # AEs評価
        for j, e in enumerate(cfg['create_AEs']['epsilons']):
            ds_AEs = tf.data.Dataset.from_tensor_slices(
                    (tmp[j]['clipped'][tmp[j]['is_adv']], )
                ).batch(
                    cfg['create_AEs']['batch_size']
                )
            pre = eval_AEs(victim_model, ds_AEs)
            tmp[j]['victim_prediction'] = pre
            tmp[j]['is_attack_success'] = (
                np.argmax(pre, axis=1) != tmp[j]['label'][[tmp[j]['is_adv']]])

            if e in transferability.keys():
                transferability[e].append(np.mean(tmp[j]['is_attack_success']))
                success_rate[e].append(np.mean(tmp[j]['is_adv']))
            else:
                transferability[e] = [np.mean(tmp[j]['is_attack_success'])]
                success_rate[e] = [np.mean(tmp[j]['is_adv'])]

            writer.log_metric('transferability_'+e,
                              np.mean(tmp[j]['is_attack_success']), step=i)
            writer.log_metric('success_rate_'+e,
                              np.mean(tmp[j]['is_adv']), step=i)
            writer.log_metric('round', i)

        result = {k: v for k, v in zip(cfg['create_AEs']['epsilons'], tmp)}

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, '{}.pkl'.format(i)), 'wb') as f:
                pickle.dump(result, f)
            writer.log_artifact(
                os.path.join(d, '{}.pkl'.format(i)))

    # Save files
    with tempfile.TemporaryDirectory() as d:
        # Success rate
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, e in enumerate(cfg['create_AEs']['epsilons']):
            ax.plot(success_rate[e], label=r'$\epsilon={}$'.format(e))
        ax.legend()
        fig.savefig(os.path.join(d, 'success_rate.png'), dpi=300)
        writer.log_artifact(os.path.join(d, 'success_rate.png'))
        np.savez(os.path.join(d, 'success_rate.npz'), **success_rate)
        writer.log_artifact(os.path.join(d, 'success_rate.npz'))

        # Transferability
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, e in enumerate(cfg['create_AEs']['epsilons']):
            ax.plot(transferability[e], label=r'$\epsilon={}$'.format(e))
        ax.legend()
        fig.savefig(os.path.join(d, 'transferability.png'), dpi=300)
        writer.log_artifact(os.path.join(d, 'transferability.png'))
        np.savez(os.path.join(d, 'transferability.npz'), **transferability)
        writer.log_artifact(os.path.join(d, 'transferability.npz'))

    writer.set_terminated()


if __name__ == '__main__':
    run()
