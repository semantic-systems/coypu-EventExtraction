from openie import StanfordOpenIE
import json
from models.event_argument_extraction.api import init, Related


# CoreNLP API 
class txt2graph(object):
    def __init__(self,ratio):
        self.properties = {
                            'openie.affinity_probability_cap': ratio,
                            }
        self.client = StanfordOpenIE(properties=self.properties)
        self.linker = entLink()

    def AnnoText(self, txt, save = False):

        triples = list()
        for triple in self.client.annotate(txt):
            sublink, oblink = self.linker.link(triple['subject']), self.linker.link(triple['object'])
            if sublink:
                triple['sublink'] = sublink[0]
            if oblink:
                triple['oblink'] = oblink[0]
            triples.append(triple)
        
        if save:
            with open('triples.json', mode = 'w', encoding='utf-8') as f:
                json.dump(triples, f)
                print('Triples have been saved in triples.json')

        return triples



class entLink(object):
    def __init__(self):
        init()   

    def link(self, text):
        return Related(text)

        

if __name__ == '__main__':
    # Initialie a imformation extraction module.
    extractor = txt2graph(1/3)
    extractor.AnnoText('Fire breaks out in Hawaii', True)