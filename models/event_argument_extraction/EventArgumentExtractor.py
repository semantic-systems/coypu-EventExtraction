from schemes import EventArgumentExtractorOutput
from models import BaseComponent


class EventArgumentExtractor(BaseComponent):
    def __init__(self, path_to_pretrained_model: str):
        super(EventArgumentExtractor).__init__()

    def forward(self, tweet: str) -> EventArgumentExtractorOutput:
        return EventArgumentExtractorOutput(tweet=tweet,
                                            event_arguments=["arg1", "arg2"],
                                            event_graph=[["arg1", "predicate","arg2"]],
                                            wikidata_links={"arg1": "link1", "arg2": "link1"})
    @property
    def __version__(self):
        return "1.0.0"
