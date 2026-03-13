import dataclasses
import abc


class BaseModelConfig(abc):
    @abc.abstractmethod
    def create(self):
        pass


class BaseDataConfig(abc):
    @abc.abstractmethod
    def create(self):
        pass



@dataclasses.dataclass
class Pi0Config(BaseModelConfig):
    action_dim: int
    action_horizon: int
    hidden_size: int = 512

    def create(self):
        from model import Pi0Model
        return Pi0Model(self)







@dataclasses.dataclass
class TrainConfig:
    model: BaseModelConfig
    data: BaseDataConfig
    lr: float
    num_epochs: int