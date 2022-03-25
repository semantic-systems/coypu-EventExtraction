from datetime import datetime

import torch
import json

from pathlib import Path
from typing import List
from dataclasses import asdict
from typing import Union

from models.event_argument_extraction import OpenIEExtractor, EventArgumentExtractor
from models.event_detection.EventDetector import EventDetector
from parser import parse
from schemes import EventExtractorOutput, EventDetectorOutput, EventArgumentExtractorOutput

EventDetectorType = Union[torch.nn.Module, EventDetector]
EventArgumentExtractorType = Union[torch.nn.Module, OpenIEExtractor, EventArgumentExtractor]


class EventExtractor(object):
    def __init__(self,
                 event_detector: EventDetectorType,
                 event_argument_extractor: EventArgumentExtractorType,
                 ):
        self.event_detector = event_detector
        self.event_argument_extractor = event_argument_extractor

    def extract_per_tweet(self, tweet: str) -> EventExtractorOutput:
        event_detector_output: EventDetectorOutput = self.event_detector.forward(tweet)
        event_type, event_type_wikidata_links = event_detector_output.event_type, event_detector_output.wikidata_links
        event_argument_extractor_output: EventArgumentExtractorOutput = self.event_argument_extractor.forward(tweet)
        event_arguments, event_graph, event_argument_wikidata_links = event_argument_extractor_output.event_arguments, event_argument_extractor_output.event_graph, event_argument_extractor_output.wikidata_links

        wikidata_links = None
        for link in [event_type_wikidata_links, event_argument_wikidata_links]:
            if link is None:
                pass
            elif wikidata_links is None:
                wikidata_links = link
            else:
                wikidata_links = {**event_type_wikidata_links, **event_argument_wikidata_links}

        return EventExtractorOutput(
            tweet=tweet,
            event_type=event_type,
            event_arguments=event_arguments,
            event_graph=event_graph,
            wikidata_links=wikidata_links,
            timestamp=self.get_date_time()
        )

    def extract_per_batch(self, tweets: List[str]) -> List[EventExtractorOutput]:
        pass

    def infer(self, tweet: str) -> EventExtractorOutput:
        output: EventExtractorOutput = self.extract_per_tweet(tweet)
        self.write_json(output)
        return output

    @staticmethod
    def write_json(output: EventExtractorOutput):
        output_json_path = Path("outputs").joinpath("output.json")
        if Path(output_json_path).exists():
            with open(output_json_path, 'r+', encoding='utf-8') as o:
                output_json = json.load(o)
                output_json.append(asdict(output))
                o.seek(0)
                json.dump(output_json, o, indent=4, sort_keys=True)
        else:
            with open(output_json_path, 'w', encoding='utf-8') as o:
                json.dump([asdict(output)], o, indent=4, sort_keys=True)

    @staticmethod
    def get_date_time() -> str:
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        return dt_string


class Instantiator(object):
    def __init__(self,
                 event_detector_model_path: str,
                 event_argument_extractor_model_path: str
                 ):
        self.set_up_directory()
        self.extractor = None
        assert Path(event_detector_model_path).exists()
        self.event_detector: EventDetectorType = self.load_event_detector(event_detector_model_path)
        self.event_argument_extractor = None
        if Path(event_argument_extractor_model_path).exists():
            self.event_argument_extractor: EventArgumentExtractorType = self.load_event_argument_extractor(
                event_argument_extractor_model_path)
        elif event_argument_extractor_model_path == "openie":
            self.event_argument_extractor = OpenIEExtractor()
        else:
            raise ValueError("Please provide a valid event_argument_extractor_model_path.")

    @staticmethod
    def load_event_detector(path: str) -> EventDetector:
        return EventDetector(path)

    @staticmethod
    def load_event_argument_extractor(path: str) -> EventArgumentExtractor:
        return EventArgumentExtractor(path)

    @staticmethod
    def set_up_directory():
        if not Path("stores/models").exists():
            Path("stores/models").mkdir()
        if not Path("outputs").exists():
            Path("outputs").mkdir()

    def __call__(self) -> EventExtractor:
        return EventExtractor(self.event_detector, self.event_argument_extractor)


if __name__ == '__main__':
    args = parse()
    event_detector_model_path = args.event_detector_path
    event_argument_extractor_model_path = args.event_argument_extractor_path
    instantiator = Instantiator(event_detector_model_path, event_argument_extractor_model_path)
    event_extractor = instantiator()
    while True:
        tweet = input('Please enter a tweet: ')
        output = event_extractor.infer(tweet)
        print(f""
              f"Event type: {output.event_type}\n"
              f"Event arguments: {output.event_arguments}\n"
              f"Event graph: {output.event_graph}\n"
              f"Wikidata links: {output.wikidata_links}\n")
