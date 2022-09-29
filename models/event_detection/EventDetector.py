import os

import torch
from torch import tensor
from typing import Optional, Tuple
from dataclasses import dataclass
from schemes import EventDetectorOutput
from models import BaseComponent
from models.event_detection.src.models.SingleLabelSequenceClassification import SingleLabelSequenceClassification
from stores.ontologies.event_type_wikidata_links_trecis import EVENT_TYPE_WIKIDATA_LINKS
from transformers import logging
logging.set_verbosity_error()


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class InputFeature:
    input_ids: tensor
    attention_mask: tensor
    labels: Optional[tensor] = None


@dataclass
class SingleLabelClassificationForwardOutput:
    loss: Optional[tensor] = None
    prediction_logits: tensor = None
    encoded_features: Optional[tensor] = None
    attentions: Optional[Tuple[tensor]] = None


class EventDetector(BaseComponent):
    def __init__(self, path_to_pretrained_model: str):
        super(EventDetector).__init__()
        checkpoint = torch.load(path_to_pretrained_model, map_location=torch.device('cpu'))
        print("cwd", os.getcwd())
        checkpoint['config']['model']["from_pretrained"] = "../../../../data/language_models/CoyPu-CrisisLM-v1"
        self.model = SingleLabelSequenceClassification(checkpoint['config'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.index_label_map = checkpoint['index_label_map']
        self.model.to(torch.device('cpu'))

    def forward(self, tweet: str) -> EventDetectorOutput:
        tokenized_text = self.model.tokenizer(tweet, padding=True, truncation=True, return_tensors="pt")
        input_ids: tensor = tokenized_text["input_ids"].to(self.model.device)
        attention_masks: tensor = tokenized_text["attention_mask"].to(self.model.device)
        labels = None
        input_feature: InputFeature = InputFeature(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
        output: SingleLabelClassificationForwardOutput = self.model.forward(input_feature, "test")
        prediction = output.prediction_logits.argmax(1).item()
        event_type = self.index_label_map[str(prediction)]
        wikidata_link = EVENT_TYPE_WIKIDATA_LINKS.get(event_type)
        if wikidata_link is not None:
            wikidata_link = "http://www.wikidata.org/entity/" + wikidata_link
        return EventDetectorOutput(tweet=tweet, event_type=event_type, wikidata_links={event_type: wikidata_link})

    @property
    def __version__(self):
        return "2.0.0"
