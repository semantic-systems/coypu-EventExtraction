import torch
import json

from pathlib import Path
from typing import List
from dataclasses import asdict
from typing import Optional, Union

from models.event_argument_extraction import OpenIEExtractor
from schemes import EventExtractorOutput, EventDetectorOutput, EventArgumentExtractorOutput

EventDetectorType = Union[torch.nn.Module]
EventArgumentExtractorType = Union[torch.nn.Module, OpenIEExtractor]


class EventExtractor(object):
    def __init__(self,
                 event_detector: EventDetectorType,
                 event_argument_extractor: EventArgumentExtractorType
                 ):
        self.event_detector = event_detector
        self.event_argument_extractor = event_argument_extractor

    def extract_per_tweet(self, tweet: str) -> EventExtractorOutput:
        event_detector_output: EventDetectorOutput = self.event_detector.forward(tweet)
        event_type, event_type_wikidata_links = event_detector_output.event_type, event_detector_output.wikidata_links
        event_argument_extractor_output: EventArgumentExtractorOutput = self.event_argument_extractor.forward(tweet)
        event_arguments, event_graph, event_argument_wikidata_links = event_argument_extractor_output.event_arguments, event_argument_extractor_output.event_graph, event_argument_extractor_output.wikidata_links
        return EventExtractorOutput(
            tweet=tweet,
            event_type=event_type,
            event_arguments=event_arguments,
            event_graph=event_graph,
            wikidata_links={**event_type_wikidata_links, **event_argument_wikidata_links}
        )

    def extract_per_batch(self, tweets: List[str]) -> List[EventExtractorOutput]:
        pass

    def infer(self, tweet: str, output_file_path: str) -> EventExtractorOutput:
        output = self.extract_per_tweet(tweet)
        with open(output_file_path, 'w') as o:
            json.dump(asdict(output), o)
        return output


class Instantiator(object):
    def __init__(self,
                 event_detector_model_path: str,
                 event_argument_extractor_model_path: Optional[str] = None
                 ):
        self.extractor = None
        assert Path(event_detector_model_path).exists()
        self.event_detector: EventDetectorType = self.load_torch_model(event_detector_model_path)
        if Path(event_argument_extractor_model_path).exists():
            self.event_argument_extractor: EventArgumentExtractorType = self.load_torch_model(
                event_argument_extractor_model_path)
        elif event_argument_extractor_model_path == "openie":
            self.event_argument_extractor = OpenIEExtractor()
        else:
            raise ValueError("Please provide a valid event_argument_extractor_model_path.")

    @staticmethod
    def load_torch_model(path: str) -> torch.nn.Module:
        return torch.jit.load(path)

    def __call__(self) -> EventExtractor:
        return EventExtractor(self.event_detector, self.event_argument_extractor)


if __name__ == '__main__':
    tweet = "You are on fire, run!"
    event_detector_model_path = "stores/models/xxx"
    output_file_path = "outputs/output.json"
    event_argument_extractor_model_path = None
    instantiator = Instantiator(event_detector_model_path, event_argument_extractor_model_path)
    event_extractor = instantiator()
    output = event_extractor.infer(tweet, output_file_path)