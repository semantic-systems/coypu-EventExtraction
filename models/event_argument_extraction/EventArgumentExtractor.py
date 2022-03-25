from schemes import EventArgumentExtractorOutput
from models import BaseComponent
from models.event_argument_extraction.ie import txt2graph


class OpenIEExtractor(BaseComponent):
    def __init__(self):
        super(OpenIEExtractor).__init__()
        self.ArgExtractor = txt2graph(1/3)

    def forward(self, tweet: str) -> EventArgumentExtractorOutput:
        triples = self.ArgExtractor.AnnoText(tweet)
        if not triples:
            event_arguments, event_graph, wikidata_links = None, None, None
        # the situation when there is only one relation in a tweet.
        else:
            event_arguments, event_graph = list(), list()
            wikidata_links = dict()
            for triple in triples:
                if triple['subject'] not in event_arguments:
                    event_arguments.append(triple['subject'])
                if triple['object'] not in event_arguments:
                    event_arguments.append(triple['object'])
                event_graph.append([triple['subject'], triple['relation'], triple['object']])
                wikidata_links[triple['subject']] = triple.get('sublink', {}).get('pid')
                wikidata_links[triple['object']] = triple.get('oblink', {}).get('pid')
                wikidata_links = {k: v for k, v in wikidata_links.items() if v is not None}

        return EventArgumentExtractorOutput(tweet=tweet,
                                            event_arguments=event_arguments,
                                            event_graph=event_graph,
                                            wikidata_links=wikidata_links)

    @property
    def __version__(self):
        return "1.0.0"


class EventArgumentExtractor(BaseComponent):
    def __init__(self, path_to_pretrained_model: str):
        super(EventArgumentExtractor).__init__()

    def forward(self, tweet: str) -> EventArgumentExtractorOutput:
        raise NotImplementedError

    @property
    def __version__(self):
        return "1.0.0"
