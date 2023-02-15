import json
import logging
import urllib.request
import uuid
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path

import requests_cache
from rdflib import Graph, Literal, Namespace, URIRef, OWL
from rdflib.namespace import FOAF, RDF, RDFS
from rdflib.plugins.parsers.notation3 import BadSyntax
from requests import RequestException

from schemes import EventDetectorOutput, EventArgumentExtractorOutput

logger = logging.getLogger(__name__)


class RDFGenerator(object):

    @staticmethod
    def _get_wikidata_coypu_mappings(coy_ontology_address: str) -> Dict[URIRef, URIRef]:
        session = requests_cache.CachedSession('coy_ontology_wikidata_mappings')
        g = Graph()
        try:
            wikidata_coy_mappings_doc = session.get(coy_ontology_address).content
            g.load(wikidata_coy_mappings_doc, format='ttl')
        except BadSyntax as e:
            logger.error(
                f'Content of retrieved document from {coy_ontology_address} '
                f'seemingly is not in valid turtle format'
            )
        except RequestException as e:
            logger.error(
                f'The CoyPu-Wikidata mappings could not be retrieved from '
                f'{coy_ontology_address}: {str(e)}'
            )

        # WARNING: A key assumption here is, that there aren't multiple mappings
        # for a single CoyPu class. In case there are were multiple mappings,
        # e.g.
        #   coy:A owl:equivalentClass wd:Z . coy:A owl:equivalentClass wd:Y
        # only one of them would be kept!
        wikidata_to_coy_mappings = \
            {s: o for s, p, o in g if p == OWL.equivalentClass}

        return dict(wikidata_to_coy_mappings)

    def __init__(
            self,
            coy_ontology_address: str = 'http://gitlab.com/coypu-project/coy-ontology/-/raw/main/ontology/mapping/coy_wikidata.ttl'
    ):
        self._wikidata_to_coy_mappings = \
            self._get_wikidata_coypu_mappings(coy_ontology_address)

    def convert(self,
                event_detector_output: EventDetectorOutput,
                event_argument_extractor_output: EventArgumentExtractorOutput,
                path_to_jsonld: Optional[str] = "../data/output.jsonld"):
        if not Path("../data").exists():
            Path("../data").mkdir()
        # first fix naming space
        coy = Namespace("https://schema.coypu.org/global#")

        g = Graph()
        g.bind("rdfs", RDFS)
        g.bind("rdf", RDF)
        g.bind("coy", coy)
        g.bind("foaf", FOAF)

        current = URIRef(f"https://data.coypu.org/event/mod/{uuid.uuid4()}")

        for linked_entity in event_argument_extractor_output.event_arguments:
            ID = URIRef("http://www.wikidata.org/entity/"+linked_entity.id)
            if linked_entity.label == "PERSON":
                # relations need to be found somewhere else
                g.add((current, coy.hasImpactOn, ID))
            # FIXME: coy:hasLocality does not exist
            # Maybe coy:hasLocation is meant instead
            if linked_entity.label == "LOC":
                g.add((current, coy.hasLocality, ID))
            if linked_entity.label == "ORG":
                g.add((current, coy.hasImpactOn, ID))
        # event type(s)
        g.add((current, RDF.type, coy.Event))
        # event_detector_output.wikidata_link: e.g. 'Q7944'
        wikidata_uri = URIRef("http://www.wikidata.org/entity/"+event_detector_output.wikidata_link)

        coy_ontology_uri = self._wikidata_to_coy_mappings.get(wikidata_uri)
        if coy_ontology_uri:
            g.add((current, RDF.type, coy_ontology_uri))

        # As https://schema.coypu.org/global#hasEventType is deprecated we use
        # rdf:Type here instead, as suggested in the property annotation.
        # g.add((current, coy.hasEventType, wikidata_uri))
        g.add((current, RDF.type, wikidata_uri))

        g.add((current, RDFS.comment, Literal(event_detector_output.tweet)))
        g.add((current, coy.hasPublisher, Literal("HiTec")))
        g.add((current, coy.hasTimestamp, Literal(self.get_date_time())))
        g.serialize(format='json-ld', indent=4, destination=path_to_jsonld)
        return json.loads(g.serialize(format='json-ld', indent=4))[0]

    @staticmethod
    def get_date_time() -> str:
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        return dt_string
