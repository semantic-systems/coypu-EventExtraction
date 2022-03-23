from schemes import EventArgumentExtractorOutput
from models import BaseComponent


class OpenIEExtractor(BaseComponent):
    def __init__(self):
        super(OpenIEExtractor).__init__()
        # inherit constructor (__init__) from BaseComponent
        # instantiate your extractor here

    def forward(self, tweet: str) -> EventArgumentExtractorOutput:
        # call your extractor here, which takes as input a tweet, and stores output as type EventArgumentExtractorOutput
        return EventArgumentExtractorOutput(tweet=tweet,
                                            event_arguments=["arg1", "arg2"],
                                            event_graph=[["arg1", "predicate","arg2"]],
                                            wikidata_links={"arg1": "link1", "arg2": "link1"})

    @property
    def __version__(self):
        return "0.0.1"


class EventArgumentExtractor(BaseComponent):
    def __init__(self, path_to_pretrained_model: str):
        super(EventArgumentExtractor).__init__()
        # TODO: skip for now. This is the interface for our custom model.

    def forward(self, tweet: str) -> EventArgumentExtractorOutput:
        # call your extractor here, which takes as input a tweet, and stores output as type EventArgumentExtractorOutput
        return EventArgumentExtractorOutput(tweet=tweet,
                                            event_arguments=["arg1", "arg2"],
                                            event_graph=[["arg1", "predicate", "arg2"]],
                                            wikidata_links={"arg1": "link1", "arg2": "link1"})

    @property
    def __version__(self):
        return "0.0.1"
