from EntLink.api import init, Related, filter
import spacy

class Linker(object):
    def __init__(self):
        self.path_to_ckpt = ''
        self.tokenizer = None
        self.model = None
        self.model_para = dict()

    def instantiate(self):
        pass
         
    def forward(self):
        pass


class RelationLinker(object):

    def instantiate(self):
        init()
         
    def forward(self, text:str) -> list:
        return filter(Related(text))


class EntityLinker(object):


    def instantiate(self):
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe('opentapioca')
         
    def forward(self, text:str) -> list:
        doc = self.nlp(text)
        out = list()
        for span in doc.ents:
            out.append((span.text, span.kb_id_, span.label_))
        return out 


if __name__ == "__main__":
    test = EntityLinker()
    test.instantiate()
    print(test.forward('Germany'))