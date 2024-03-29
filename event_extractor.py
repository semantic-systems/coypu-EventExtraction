import torch
from typing import List, Union, Optional

from models.event_argument_extraction.FalconEventArgumentExtractor import FalconEventArgumentExtractor
from models.event_argument_extraction.OpenTapiocaArgumentExtractor import OpenTapiocaArgumentExtractor
from models.event_argument_extraction.RebelEventArgumentExtractor import RebelEventArgumentExtractor
from models.entity_linking.BLINKEntityLinker import BLINKEntityLinker
from models.event_detection.EventDetector import EventDetector
from models.graph_generator.rdf_generator import RDFGenerator
from schemes import EventExtractorOutput, EventDetectorOutput, EventArgumentExtractorOutput

EventDetectorType = Union[torch.nn.Module, EventDetector]
EventArgumentExtractorType = Union[torch.nn.Module, FalconEventArgumentExtractor, RebelEventArgumentExtractor,
                                   OpenTapiocaArgumentExtractor, BLINKEntityLinker]


class EventExtractor(object):
    def __init__(self,
                 event_detector: EventDetectorType,
                 event_argument_extractor: Optional[EventArgumentExtractorType] = None,
                 ):
        self.event_detector = event_detector
        self.event_argument_extractor = event_argument_extractor

    def extract_per_batch(self, tweets: List[str]) -> EventExtractorOutput:
        if isinstance(tweets, list):
            first_tweets = tweets[0]
        elif isinstance(tweets, str):
            first_tweets = tweets
        else:
            raise TypeError("Tweets must be in list or str.")
        event_detector_output: EventDetectorOutput = self.event_detector.forward(tweets)
        event_types = event_detector_output.event_type
        event_argument_extractor_output: EventArgumentExtractorOutput = self.event_argument_extractor.forward(first_tweets)
        first_event_detector_output = event_detector_output
        first_event_detector_output.tweet = first_event_detector_output.tweet[0]
        first_event_detector_output.event_type = first_event_detector_output.event_type[0]
        first_event_detector_output.wikidata_link = first_event_detector_output.wikidata_link[0]
        rdf_graph = RDFGenerator().convert(first_event_detector_output, event_argument_extractor_output)

        return EventExtractorOutput(
            tweet=tweets,
            event_type=event_types,
            event_arguments=event_argument_extractor_output.event_arguments,
            event_graph=rdf_graph
        )

    def infer(self, tweets: List[str]) -> tuple:
        output: EventExtractorOutput = self.extract_per_batch(tweets)
        return output.event_type, output.event_arguments, output.event_graph


if __name__ == '__main__':
    event_detector = EventDetector()
    event_argument_extractor = BLINKEntityLinker()
    event_extractor = EventExtractor(event_detector=event_detector, event_argument_extractor=event_argument_extractor)
    import gradio as gr

    title = "Hitec Event Extraction Demo v2"
    description = "Event Detector: crisis-related LM + supervised contrastive learning on TREC-IS dataset. \n\n " \
                  "Event Argument Extractor: wikidata-based knowledge extraction module.\n\n " \
                  "(paste the extracted event graph [here](https://json-ld.org/playground) to see the visualization!)"
    article = '<img src="https://lh5.googleusercontent.com/5BFOPuJcVnIYF4P5bmvxTAYoD2cLsqFa_FZUyyUqYCivGq3XdfEPQJ_EvOMdwHD84957qvcitITbAMlKZi5E1nTBjrLXeG7rxPl2FbjJ_I8Ka7gnn7lX7ce1ZZUMlfzjn5R9hztf_q2K3T_lWMDki6AdtFuHT-YeikcG0j8QXCeWzONg6-t3e4NC4gmF" width=150px><img src="https://lh5.googleusercontent.com/vq_hxMF8lqmXdxIQJJGwq9SPRrE15SFRwCWHBWZXlFS_10LTAuzHmcx-1NUYznbXwH02KZgjpwz5eYM-j5m3RUOhW90rDV9uFj-30dBg7kS6irstL3_VHuHkOe5jskDM2-rTLAKhchDlrH3hAG8W9o9XIlzxWD-tQoj83oyrv6NVmsjhmAPR6ohDsmED" width=150px>'
    with gr.Blocks() as d:
        with gr.Row():
            input_box = gr.Textbox(placeholder="Enter a sentence here...", label="Input Tweet")
            output_box_event_type = gr.Textbox(label="Event type:")
            output_box_entities = gr.JSON(label="Extracted entities:")
            output_box_graph = gr.JSON(label="Event graph:")
    demo = gr.Interface(fn=event_extractor.infer,
                        inputs=input_box,
                        outputs= [output_box_event_type, output_box_entities, output_box_graph],
                        examples=[["A preliminary 6.20 magnitude #earthquake has occurred near Taft, Eastern Visayas, #Philippines."],
                                   ["A shooting has been reported at Saugus High School in Santa Clarita just north of Los Angeles."],
                                   ["Six Vicpol officers have tested positive this month #COVID19"],
                                   ["One person was missing following a large explosion at an apparent industrial building in Houston Friday. The blast damaged nearby buildings and homes."]
                                   ],
                        title=title,
                        description=description,
                        article=article)
    demo.launch(share=True, show_error=True)
