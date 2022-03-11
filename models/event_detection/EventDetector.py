from schemes import EventDetectorOutput


class EventDetector(object):
    def __init__(self):
        self.name = self.__name__

    def forward(self, tweet: str) -> EventDetectorOutput:
        raise NotImplementedError
