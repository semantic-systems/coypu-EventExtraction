from schemes import EventArgumentExtractorOutput


class OpenIEExtractor(object):
    def __init__(self):
        self.name = self.__name__
        ## instantiate your extractor here

    def forward(self, tweet: str) -> EventArgumentExtractorOutput:
        ## call your extractor here, which takes as input a tweet, and stores output as type EventArgumentExtractorOutput
        raise NotImplementedError
