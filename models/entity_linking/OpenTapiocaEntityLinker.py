from typing import List
import spacy
from Linker import Linker
from rdflib import Graph, Literal, BNode, Namespace, URIRef
from rdflib.namespace import FOAF, RDF, RDFS

class OpenTapiocaEntityLinker(Linker):
    def __init__(self):
        super(OpenTapiocaEntityLinker, self).__init__()

    @staticmethod
    def instantiate():
        model = spacy.blank("en")
        model.add_pipe('opentapioca')
        return model

    def forward(self, text: str) -> List:
        model_output = self.model(text)
        return [(span.text, span.kb_id_, span.label_) for span in model_output.ents]

class OpenTapiocaText2Graph(Linker):
    def __init__(self):
        super(OpenTapiocaText2Graph, self).__init__()

    @staticmethod
    def instantiate():
        model = spacy.blank("en")
        model.add_pipe('opentapioca')
        return model

    def forward(self, text: str) -> List:
        model_output = self.model(text)
        # only has people, organization and locations
        return [(span.text, span.kb_id_, span.label_, span._.description) for span in model_output.ents]

    # get output from 
    def list2Graph(self, text, type = str):
        info = self.forward(text)
        # first fix naming space
        coy = Namespace("https://schema.coypu.org/global#")


        g = Graph()
        g.bind("rdfs", RDFS)
        g.bind("rdf", RDF)
        g.bind("coy", coy)
        g.bind("foaf", FOAF)

        current = BNode()  # a GUID is generated

        for item in info:
            ID  = URIRef("http://www.wikidata.org/entity/"+item[1])
            if item[2] == "PERSON":
                # relations need to be found somewhere else
                g.add((current, coy.hasImpactOn, ID))
            if item[2] == "LOC":
                g.add((current, coy.hasLocality, ID))
            if item[2] == "ORG":
                g.add((current, coy.hasImpactOn, ID))
        # event type
        g.add((current, RDF.type, Literal(type)))
        g.add((current, coy.hasEventType, Literal(type)))
        g.add((current, RDFS.comment, Literal(text)))
        g.add((current, coy.hasPublisher, Literal("HiTec")))
        g.serialize(format='json-ld', indent=4, destination="test.jsonld")
        
        



if __name__ == "__main__":
    test = OpenTapiocaText2Graph()
#    print(test.forward('Christian Drosten works in Germany for Apple company during the covid outbreak.'))
    test.list2Graph("Christian Drosten works in Germany for Apple company during the covid outbreak.")