from schemes import EventArgumentExtractorOutput
from models import BaseComponent


class BaseEventArgumentExtractor(BaseComponent):
    def __init__(self):
        super(BaseEventArgumentExtractor).__init__()

    def forward(self, tweet:str) -> EventArgumentExtractorOutput:
        return EventArgumentExtractorOutput(tweet=tweet,
                                            event_arguments=["arg1", "arg2"],
                                            event_graph=[["arg1", "predicate","arg2"]],
                                            wikidata_links={"arg1": None, "arg2": None})


class BaseEventTemporalInformationExtractor(BaseComponent):
    def __init__(self, path_to_pretrained_model: str):
        super(BaseEventTemporalInformationExtractor).__init__()

    def forward(self, tweet: str) -> EventArgumentExtractorOutput:
        return EventArgumentExtractorOutput(tweet=tweet,
                                            event_arguments=["arg1", "arg2"],
                                            event_graph=[["arg1", "predicate","arg2"]],
                                            wikidata_links={"arg1": None, "arg2": None})


class BaseEventGeoSpatialInformationExtractor(BaseComponent):
    def __init__(self, path_to_pretrained_model: str):
        super(BaseEventGeoSpatialInformationExtractor).__init__()

    def forward(self, tweet: str) -> EventArgumentExtractorOutput:
        return EventArgumentExtractorOutput(tweet=tweet,
                                            event_arguments=["arg1", "arg2"],
                                            event_graph=[["arg1", "predicate","arg2"]],
                                            wikidata_links={"arg1": None, "arg2": None})
