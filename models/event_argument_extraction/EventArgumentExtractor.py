from schemes import EventArgumentExtractorOutput
from models import BaseComponent


class EventArgumentExtractor(BaseComponent):
    def __init__(self, path_to_pretrained_model: str):
        super(EventArgumentExtractor).__init__()

    def forward(self, tweet: str) -> EventArgumentExtractorOutput:
        raise NotImplementedError

    @property
    def __version__(self):
        return "1.0.0"
