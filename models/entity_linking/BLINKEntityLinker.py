from typing import List
from Linker import Linker
import blink.main_dense as main_dense
import argparse
from flair.data import Sentence
from flair.models import SequenceTagger
import ipdb
import sys
import io
# please install blink as BLINK with 
# pip install -e git+https://github.com/facebookresearch/BLINK#egg=BLINK

# the above code shall generate a directory called scr in the current dir and install BLINK as a package.
# change two files in src:
# in download_blink_model.sh: change line13 to : ROOD_DIR=$(dirname "$0")
# replace main_dense.py by the script i uploaded

# go to `src/blink`, execute to download all needed files (this will take a while):
# chmod +x download_blink_models.sh
#./download_blink_models.sh

class BLINKEntityLinker(Linker):
    def __init__(self):
        super(BLINKEntityLinker, self).__init__()

    @staticmethod
    def instantiate():
        models_path = "src/blink/models/"
        config = {
            "test_entities": None,
            "test_mentions": None,
            "interactive": False,
            "top_k": 10,
            "biencoder_model": models_path+"biencoder_wiki_large.bin",
            "biencoder_config": models_path+"biencoder_wiki_large.json",
            "entity_catalogue": models_path+"entity.jsonl",
            "entity_encoding": models_path+"all_entities_large.t7",
            "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
            "crossencoder_config": models_path+"crossencoder_wiki_large.json",
            "fast": True, # set this to be true if speed is a concern
            "output_path": "logs/" # logging directory
        }

        args = argparse.Namespace(**config)
        model = main_dense.load_models(args, logger=None)

        tagger = SequenceTagger.load('ner')
        models = [model, tagger]
        return models

    def forward(self, text: str) -> List:
        out = list()
        NerOut = self.runFlair(text)
        
        for ner_pair in NerOut:
            mention, label = ner_pair[0], ner_pair[1]
            WikiIDs = self.runBLINK(ner_pair[0], text)
            ipdb.set_trace()
            WikiID, WikiStr =  WikiIDs[0][0][1],WikiIDs[0][0][0]
            out.append((mention, WikiID, WikiStr,label))
            
        return out
    
    # ner tool used by BLINK
    # output a list of mention and label pairs saved in list
    def runFlair(self, text: str) -> List:
        sentence = Sentence(text)
        tagger = self.model[1]
        tagger.predict(sentence)

        output = list()
        for entity in sentence.get_spans('ner'):
            old_stdout = sys.stdout 
            sys.stdout = buffer = io.StringIO()
            print(entity)
            sys.stdout = old_stdout # Put the old stream back in place
            whatWasPrinted = buffer.getvalue() # Return a str containing the entire contents of the buffer.
            print(whatWasPrinted)
            mention = entity.text
            label =  whatWasPrinted.split('Labels')[-1].split(': ')[-1].split(' (')[0]
            output.append([mention, label])

        return output
        

    # entity linking module
    # output a list of possible wikiIDs.
    # you could adjust the top_k para to get more possible candidate
    def runBLINK(self, mention: str, text: str):
        models_path = "src/blink/models/"
        config = {
            "test_entities": None,
            "test_mentions": None,
            "interactive": False,
            "top_k": 1,
            "biencoder_model": models_path+"biencoder_wiki_large.bin",
            "biencoder_config": models_path+"biencoder_wiki_large.json",
            "entity_catalogue": models_path+"entity.jsonl",
            "entity_encoding": models_path+"all_entities_large.t7",
            "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
            "crossencoder_config": models_path+"crossencoder_wiki_large.json",
            "fast": True, # set this to be true if speed is a concern
            "output_path": "logs/" # logging directory
        }

        args = argparse.Namespace(**config)
        data_to_link = [{
                            "id": 0,
                            "label": "unknown",
                            "label_id": -1,
                            "context_left": text.split(mention)[0].lower(),
                            "mention": mention.lower(),
                            "context_right": text.split(mention)[0].lower(),
                        }
                        ]

        _, _, _, _, _, predictions, _, = main_dense.run(args, None, *self.model[0], test_data=data_to_link)
        return predictions


if __name__ == "__main__":
    test = BLINKEntityLinker()
    print(test.forward("Shakespeare s account of the Roman general Julius Caesar's murder by his friend Brutus is a meditation on duty."))
