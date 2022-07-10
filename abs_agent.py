import abc


class AbsAgent(abc.ABC):
    # This abstract class is included to faciliate the development of
    # additional agents. It provide the minimum functionality that an
    # agent must provide to be compatible with this project.

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass
