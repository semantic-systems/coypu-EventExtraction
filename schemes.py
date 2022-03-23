from typing import Set, List, Dict
from dataclasses import dataclass


@dataclass
class EventDetectorOutput:
    tweet: str
    event_type: str
    wikidata_links: Dict[str, str]


@dataclass
class EventArgumentExtractorOutput:
    tweet: str
    event_arguments: List[str]
    event_graph: List[List[str]]
    wikidata_links: Dict[str, str]


@dataclass
class EventExtractorOutput:
    tweet: str
    event_type: str
    event_arguments: List[str]
    event_graph: List[List[str]]
    wikidata_links: Dict[str, str]
    timestamp: str
