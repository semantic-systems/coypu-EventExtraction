

class BaseComponent(object):
    def __init__(self):
        self.name = self.__name__

    def forward(self, tweet: str):
        raise NotImplementedError
