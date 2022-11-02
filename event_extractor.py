import os
import torch
from typing import List, Dict, Union, Optional
from datetime import datetime

from models.event_argument_extraction.FalconEventArgumentExtractor import FalconEventArgumentExtractor
from models.event_argument_extraction.RebelEventArgumentExtractor import RebelEventArgumentExtractor
from models.event_detection.EventDetector import EventDetector
from schemes import EventExtractorOutput, EventDetectorOutput, EventArgumentExtractorOutput, Config, \
    ModelConfig, PublicMetaConfig, LocalMetaConfig

EventDetectorType = Union[torch.nn.Module, EventDetector]
EventArgumentExtractorType = Union[torch.nn.Module, FalconEventArgumentExtractor, RebelEventArgumentExtractor]
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
        return output.event_type, output.event_arguments, output.event_graph, output.wikidata_links, output.timestamp

    @staticmethod
    def get_date_time() -> str:
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        return dt_string


if __name__ == '__main__':
    event_detector = EventDetector()
    event_argument_extractor = RebelEventArgumentExtractor()
    event_extractor = EventExtractor(event_detector=event_detector, event_argument_extractor=event_argument_extractor)
    import gradio as gr
    demo = gr.Interface(fn=event_extractor.infer,
                        inputs=gr.Textbox(placeholder="Enter a sentence here..."),
                        outputs=["text", "json", "json"],
                        examples=[["A preliminary 6.20 magnitude #earthquake has occurred near Taft, Eastern Visayas, #Philippines."],
                                   ["A shooting has been reported at Saugus High School in Santa Clarita just north of Los Angeles."],
                                   ["Six Vicpol officers have tested positive this month #COVID19"],
                                   ["One person was missing following a large explosion at an apparent industrial building in Houston Friday. The blast damaged nearby buildings and homes."]
                                   ])
    demo.launch(share=True, show_error=True)
