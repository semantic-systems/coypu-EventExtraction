from schemes import EventArgumentExtractorOutput
from models import BaseComponent


class OpenIEExtractor(BaseComponent):
        ## inherit constructor (__init__) from BaseComponent
        ## instantiate your extractor here

    def forward(self, tweet: str) -> EventArgumentExtractorOutput:
        ## call your extractor here, which takes as input a tweet, and stores output as type EventArgumentExtractorOutput
        raise NotImplementedError
