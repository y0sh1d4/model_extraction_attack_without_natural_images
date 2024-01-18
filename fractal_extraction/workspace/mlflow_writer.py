# from: https://github.com/ymym3412/Hydra-MLflow-experiment-management/blob/master/mlflow_writer.py
from mlflow.tracking import MlflowClient
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig


class MlflowWriter():
    def __init__(self, experiment_name, **kwargs):
        self.client = MlflowClient(**kwargs)
        if experiment_name in [v.name for v in self.client.list_experiments()]:
            self.experiment_id = \
                self.client.get_experiment_by_name(
                    experiment_name).experiment_id
        else:
            self.experiment_id = self.client.create_experiment(experiment_name)

        self.run_id = self.client.create_run(self.experiment_id).info.run_id

    def log_params_from_omegaconf_dict(self, params):
        for param_name, element in params.items():
            self._explore_recursive(param_name, element)

    def _explore_recursive(self, parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    self._explore_recursive(f'{parent_name}.{k}', v)
                else:
                    self.log_param(f'{parent_name}.{k}', v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                self.log_param(f'{parent_name}.{i}', v)
        else:
            self.log_param(parent_name, element)

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value, step=0):
        self.client.log_metric(self.run_id, key, value, step=step)

    def log_history(self, history) -> None:
        for k in history.history.keys():
            for i, v in enumerate(history.history[k]):
                self.log_metric(k, v, step=i)

    def log_artifact(self, local_path):
        self.client.log_artifact(self.run_id, local_path)

    def set_terminated(self):
        self.client.set_terminated(self.run_id)
