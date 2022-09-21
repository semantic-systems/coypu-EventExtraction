import os

import torch
import yaml

from pathlib import Path
from typing import List, Dict, Union, Optional
from datetime import datetime

from models.event_argument_extraction import EventArgumentExtractor
from models.event_detection.EventDetector import EventDetector
from parser import parse
from schemes import EventExtractorOutput, EventDetectorOutput, EventArgumentExtractorOutput, Config, \
    ModelConfig, PublicMetaConfig, LocalMetaConfig
from stores.download import download_from_google_drive

EventDetectorType = Union[torch.nn.Module, EventDetector]
EventArgumentExtractorType = Union[torch.nn.Module, EventArgumentExtractor]
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class EventExtractor(object):
    def __init__(self,
                 event_detector: EventDetectorType,
                 event_argument_extractor: Optional[EventArgumentExtractorType] = None,
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
                 event_type_detector_path: str,
                 event_argument_extractor_path: str
                 ):
        self.extractor = None
        self.event_detector: EventDetectorType = self.load_event_detector(event_type_detector_path)
        self.event_argument_extractor: EventArgumentExtractorType = self.load_event_argument_extractor(
            event_argument_extractor_path)

    @staticmethod
    def load_event_detector(path: str) -> EventDetector:
        return EventDetector(path)

    @staticmethod
    def load_event_argument_extractor(path: str) -> EventArgumentExtractor:
        return EventArgumentExtractor(path)

    def __call__(self) -> EventExtractor:
        return EventExtractor(self.event_detector, self.event_argument_extractor)
