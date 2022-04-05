import torch
import yaml

from pathlib import Path
from typing import List, Dict, Union
from datetime import datetime

from models.event_argument_extraction import OpenIEExtractor, EventArgumentExtractor
from models.event_detection.EventDetector import EventDetector
from parser import parse
from schemes import EventExtractorOutput, EventDetectorOutput, EventArgumentExtractorOutput, Config, \
    ModelConfig, PublicMetaConfig, LocalMetaConfig

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
        return output

    @staticmethod
    def get_date_time() -> str:
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        return dt_string


class Instantiator(object):
    def __init__(self,
                 config: Config
                 ):
        self.config = config
        self.extractor = None
        self.event_detector: EventDetectorType = self.load_event_detector(self.event_type_detector_path)
        self.event_argument_extractor = None
        if Path(self.event_argument_extractor_path).exists():
            self.event_argument_extractor: EventArgumentExtractorType = self.load_event_argument_extractor(
                self.event_argument_extractor_path)
        elif self.event_argument_extractor_path == "openie":
            self.event_argument_extractor = OpenIEExtractor()
        else:
            raise ValueError("Please provide a valid event_argument_extractor_model_path.")

    @property
    def event_type_detector_path(self) -> str:
        if self.config.event_type_detector.type == "custom":
            path = str(Path(self.config.event_type_detector.meta.directory_to_store)
                       .joinpath("pretrained_event_detector.pt").absolute())
        elif self.config.event_type_detector.type == "public":
            path = self.config.event_type_detector.meta.package
        else:
            raise ValueError("Please provide the model type as custom or public.")
        return path

    @property
    def event_argument_extractor_path(self) -> str:
        if self.config.event_argument_extractor.type == "custom":
            path = str(Path(self.config.event_argument_extractor.meta.directory_to_store)
                       .joinpath("pretrained_event_detector.pt").absolute())
        elif self.config.event_argument_extractor.type == "public":
            path = self.config.event_argument_extractor.meta.package
        else:
            raise ValueError("Please provide the model type as custom or public.")
        return path

    @staticmethod
    def load_event_detector(path: str) -> EventDetector:
        return EventDetector(path)

    @staticmethod
    def load_event_argument_extractor(path: str) -> EventArgumentExtractor:
        return EventArgumentExtractor(path)

    def __call__(self) -> EventExtractor:
        return EventExtractor(self.event_detector, self.event_argument_extractor)


def validate(config: Dict) -> Config:
    empty_config = {}
    keys = config.keys()
    for key in keys:
        if config.get(key, None) is not None:
            if config.get(key).get("type") == "custom":
                meta_config = LocalMetaConfig(**config.get(key).get("meta"))
                model_config = ModelConfig(type="custom", meta=meta_config)
            elif config.get(key).get("type") == "public":
                meta_config = PublicMetaConfig(**config.get(key).get("meta"))
                model_config = ModelConfig(type="public", meta=meta_config)
            else:
                raise ValueError("Please select type from custom or public.")
            empty_config[key] = model_config
        else:
            empty_config[key] = None
    return Config(**empty_config)


if __name__ == '__main__':
    args = parse()
    config_path: str = str(Path(args.config).absolute())
    with open(config_path, "r") as f:
        config: Dict = yaml.safe_load(f)
        config: Config = validate(config)
    instantiator = Instantiator(config)
    event_extractor = instantiator()
    while True:
        tweet = input('Please enter a tweet: ')
        output = event_extractor.infer(tweet)
        print(f""
              f"Event type: {output.event_type}\n"
              f"Event arguments: {output.event_arguments}\n"
              f"Event graph: {output.event_graph}\n"
              f"Wikidata links: {output.wikidata_links}\n")
