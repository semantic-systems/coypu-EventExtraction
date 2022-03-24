from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class EventDetectorOutput:
    tweet: str
    event_type: Optional[str]
    wikidata_links: Optional[Dict[str, str]]


@dataclass
class EventArgumentExtractorOutput:
    tweet: str
    event_arguments: Optional[List[str]]
    event_graph: Optional[List[List[str]]]
    wikidata_links: Optional[Dict[str, str]]


@dataclass
class EventExtractorOutput:
    tweet: str
    event_type: Optional[str]
    event_arguments: Optional[List[str]]
    event_graph: Optional[List[List[str]]]
    wikidata_links: Optional[Dict[str, str]]
    timestamp: str
