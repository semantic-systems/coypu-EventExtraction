import torch
from typing import List, Union, Optional
from models.event_argument_extraction.FalconEventArgumentExtractor import FalconEventArgumentExtractor
from models.event_argument_extraction.OpenTapiocaArgumentExtractor import OpenTapiocaArgumentExtractor
from models.event_argument_extraction.RebelEventArgumentExtractor import RebelEventArgumentExtractor
from models.event_detection.EventDetector import EventDetector
from models.graph_generator.rdf_generator import RDFGenerator
from schemes import EventExtractorOutput, EventDetectorOutput, EventArgumentExtractorOutput

EventDetectorType = Union[torch.nn.Module, EventDetector]
EventArgumentExtractorType = Union[torch.nn.Module, FalconEventArgumentExtractor, RebelEventArgumentExtractor,
                                   OpenTapiocaArgumentExtractor]


class EventExtractor(object):
    def __init__(self,
                 event_detector: EventDetectorType,
                 event_argument_extractor: Optional[EventArgumentExtractorType] = None,
                 ):
        self.event_detector = event_detector
        self.event_argument_extractor = event_argument_extractor

    def extract_per_tweet(self, tweet: str) -> EventExtractorOutput:
        event_detector_output: EventDetectorOutput = self.event_detector.forward(tweet)
        event_argument_extractor_output: EventArgumentExtractorOutput = self.event_argument_extractor.forward(tweet)
        rdf_graph = RDFGenerator().convert(event_detector_output, event_argument_extractor_output)
        return EventExtractorOutput(
            tweet=tweet,
            event_type=event_detector_output.event_type,
            event_arguments=event_argument_extractor_output.event_arguments,
            event_graph=rdf_graph
        )

    def extract_per_batch(self, tweets: List[str]) -> List[EventExtractorOutput]:
        pass

    def infer(self, tweet: str) -> tuple:
        output: EventExtractorOutput = self.extract_per_tweet(tweet)
        return output.event_type, output.event_arguments, output.event_graph


if __name__ == '__main__':
    event_detector = EventDetector()
    event_argument_extractor = OpenTapiocaArgumentExtractor()
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
