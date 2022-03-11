from schemes import EventArgumentExtractorOutput


class OpenIEExtractor(object):
    def __init__(self):
        self.name = self.__name__

    def forward(self, tweet: str) -> EventArgumentExtractorOutput:
        raise NotImplementedError
