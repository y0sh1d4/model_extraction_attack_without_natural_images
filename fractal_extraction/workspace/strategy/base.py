import tensorflow as tf
from abc import abstractmethod


class strategy_base():
    def __init__(self, ds_test, ds_ext, model, cfg):
        self.ds_test = ds_test
        self.ds_ext = ds_ext
        self.model = model
        self.cfg = cfg
        self.accuracy = []

    @abstractmethod
    def get_query() -> tf.Tensor:
        pass

    @abstractmethod
    def update(self, query, response) -> None:
        pass
