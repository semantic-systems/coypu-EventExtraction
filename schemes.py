from typing import Set, List, Dict
from dataclasses import dataclass


@dataclass
class EventDetectorOutput():
    tweet: str
    event_type: str
    wikidata_links: Dict[str, str]


@dataclass
class EventArgumentExtractorOutput():
    tweet: str
    event_arguments: Set[str]
    event_graph: Set[List[str]]
    wikidata_links: Dict[str, str]


@dataclass
class EventExtractorOutput():
    tweet: str
    event_type: str
    event_arguments: Set[str]
    event_graph: Set[List[str]]
    wikidata_links: Dict[str, str]
