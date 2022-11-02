import requests
from schemes import EventArgumentExtractorOutput
from models import BaseComponent
from transformers import pipeline


class BaseEventArgumentExtractor(BaseComponent):
    def __init__(self):
        super(BaseEventArgumentExtractor).__init__()

    def forward(self, tweet:str) -> EventArgumentExtractorOutput:
        return EventArgumentExtractorOutput(tweet=tweet,
                                            event_arguments=["arg1", "arg2"],
                                            event_graph=[["arg1", "predicate","arg2"]],
                                            wikidata_links={"arg1": None, "arg2": None})

    @staticmethod
    def get_tuples(predict_result):
        tuple = {"e1": "", 'predicate': "", 'tuples': []}
        for res in predict_result:
            if res["entity_group"] == 'P':
                tuple["predicate"] = res["word"]
            else:
                if res["entity_group"] == 'A0':
                    tuple["e1"] = res["word"]
                else:
                    tuple["tuples"].append([tuple["e1"], tuple["predicate"], res['word']])
        return tuple['tuples']


class BaseEventTemporalInformationExtractor(BaseComponent):
    def __init__(self, path_to_pretrained_model: str):
        super(BaseEventTemporalInformationExtractor).__init__()

    def forward(self, tweet: str) -> EventArgumentExtractorOutput:
        return EventArgumentExtractorOutput(tweet=tweet,
                                            event_arguments=["arg1", "arg2"],
                                            event_graph=[["arg1", "predicate","arg2"]],
                                            wikidata_links={"arg1": None, "arg2": None})


class BaseEventGeoSpatialInformationExtractor(BaseComponent):
    def __init__(self, path_to_pretrained_model: str):
        super(BaseEventGeoSpatialInformationExtractor).__init__()

    def forward(self, tweet: str) -> EventArgumentExtractorOutput:
        return EventArgumentExtractorOutput(tweet=tweet,
                                            event_arguments=["arg1", "arg2"],
                                            event_graph=[["arg1", "predicate","arg2"]],
                                            wikidata_links={"arg1": None, "arg2": None})


class EventArgumentExtractor(BaseEventArgumentExtractor):
    def __init__(self, path_to_pretrained_model: str):
        super(EventArgumentExtractor).__init__()
        self.token_classifier = pipeline(
            "token-classification", model=path_to_pretrained_model, aggregation_strategy="simple")

    def forward(self, tweet: str) -> EventArgumentExtractorOutput:
        result = self.token_classifier(tweet)
        tuples = self.get_tuples(result)
        entities = list(set([tuple[0] for tuple in tuples] + [tuple[-1] for tuple in tuples]))
        return EventArgumentExtractorOutput(tweet=tweet,
                                            event_arguments=entities,
                                            event_graph=tuples,
                                            wikidata_links={entity: None for entity in entities})


class FalconEventArgumentExtractor(BaseEventArgumentExtractor):
    def __init__(self, path_to_pretrained_model: str):
        super(FalconEventArgumentExtractor).__init__()

    def forward(self, tweet: str) -> EventArgumentExtractorOutput:
        json_data = {
            'text': tweet,
        }
        headers = {
            'Content-Type': 'application/json',
        }
        response = requests.post('https://labs.tib.eu/falcon/falcon2/api?mode=long', json=json_data, headers=headers)
        entities = response.json().get("entities_wikidata", None)
        relations = response.json().get("relations_wikidata", None)
        if entities is not None:
            entity = [entity["surface form"] for entity in entities] + [relation["surface form"] for relation in relations]
            wikidata_links = [entity["URI"] for entity in entities] + [relation["URI"] for relation in relations]
            return EventArgumentExtractorOutput(tweet=tweet,
                                                event_arguments=entity,
                                                event_graph=None,
                                                wikidata_links={e: wikidata_links[i] for i, e in enumerate(entity)})
        else:
            return EventArgumentExtractorOutput(tweet=tweet,
                                                event_arguments=None,
                                                event_graph=None,
                                                wikidata_links=None)

