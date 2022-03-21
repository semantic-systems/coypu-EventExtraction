from schemes import EventDetectorOutput
from models import BaseComponent


class EventDetector(BaseComponent):

    def forward(self, tweet: str) -> EventDetectorOutput:
        raise NotImplementedError
