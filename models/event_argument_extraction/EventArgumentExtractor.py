from schemes import EventArgumentExtractorOutput
from models import BaseComponent


class OpenIEExtractor(BaseComponent):
        ## inherit constructor (__init__) from BaseComponent
        ## instantiate your extractor here

    def forward(self, tweet: str) -> EventArgumentExtractorOutput:
        ## call your extractor here, which takes as input a tweet, and stores output as type EventArgumentExtractorOutput
        raise NotImplementedError

    @property
    def __version__(self):
        return "0.0.1"
