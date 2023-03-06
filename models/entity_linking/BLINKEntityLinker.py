from models.entity_linking.Linker import Linker
from torch.utils.data import DataLoader, SequentialSampler
from flair.data import Sentence
from flair.models import SequenceTagger
import argparse
import src.blink.blink.main_dense as main_dense
from src.blink.blink.main_dense import _run_biencoder
from src.blink.blink.biencoder.data_process import process_mention_data
from schemes import LinkedEntity, EventArgumentExtractorOutput

# please install blink as BLINK with 
# pip install -e git+https://github.com/facebookresearch/BLINK#egg=BLINK

# the above code shall generate a directory called scr in the current dir and install BLINK as a package.
# change two files in src:
# in download_blink_model.sh: change line13 to : ROOD_DIR=$(dirname "$0")
# replace main_dense.py by the script i uploaded

# go to `src/blink`, execute to download all needed files (this will take a while):
# chmod +x download_blink_models.sh
#./download_blink_models.sh


MODEL_PATH = "/data/blink/models/" #"./src/blink/models/"


class FlairNER(object):
    def __init__(self):
        self.model = SequenceTagger.load("ner")

    def predict(self, sentence):
        sentence = Sentence(sentence)
        mentions = []
        labels = []
        start_positions = []
        end_positions = []
        confidences = []
        self.model.predict(sentence)
        for entity in sentence.get_spans('ner'):
            labels.append(entity.tag)
            mentions.append(entity.text)
            start_positions.append(entity.start_position)
            end_positions.append(entity.end_position)
            confidences.append(round(entity.score, 3))

        return {"sentence": sentence, "mentions": mentions, "labels": labels, "start_positions": start_positions,
                "end_positions": end_positions, "confidences": confidences}


class BLINKEntityLinker(Linker):
    def __init__(self):
        super(BLINKEntityLinker, self).__init__()
        self.ner = FlairNER()
        self.instantiate_blink_models()
        self.id2url = {v: "https://en.wikipedia.org/wiki?curid=%s" % k for k, v in self.wikipedia_id2local_id.items()}

    def instantiate_blink_models(self):
        models_path = MODEL_PATH
        config = {
            "test_entities": None,
            "test_mentions": None,
            "interactive": False,
            "top_k": 1,
            "biencoder_model": models_path + "biencoder_wiki_large.bin",
            "biencoder_config": models_path + "biencoder_wiki_large.json",
            "entity_catalogue": models_path + "entity.jsonl",
            "entity_encoding": models_path + "all_entities_large.t7",
            "crossencoder_model": models_path + "crossencoder_wiki_large.bin",
            "crossencoder_config": models_path + "crossencoder_wiki_large.json",
            "fast": True,  # set this to be true if speed is a concern
            "output_path": "logs/"  # logging directory
        }

        args = argparse.Namespace(**config)
        self.biencoder, self.biencoder_params, self.crossencoder, self.crossencoder_params, self.candidate_encoding, \
        self.title2id, self.id2title, self.id2text, self.wikipedia_id2local_id, self.faiss_indexer = main_dense.load_models(args, logger=None)

    def forward(self, text):
        samples = self.annotate(text)
        dataloader = self.process_biencoder_dataloader(
            samples, self.biencoder.tokenizer, self.biencoder_params
        )
        top_k = 1
        labels, nns, scores = _run_biencoder(
            self.biencoder, dataloader, self.candidate_encoding, top_k, self.faiss_indexer
        )
        entities = []
        event_arguments = []
        for entity_list, sample in zip(nns, samples):
            e_id = entity_list[0]
            e_title = self.id2title[e_id]
            e_text = self.id2text[e_id]
            e_url = self.id2url[e_id]
            entities.append((e_id, e_title, e_text, e_url))
            linked_entity = LinkedEntity(entity=sample["mention"], id=f"Q{e_id}", label=sample["label"],
                                         description=e_text)
            event_arguments.append(linked_entity)
        return EventArgumentExtractorOutput(tweet=text, event_arguments=event_arguments)

    def annotate(self, input_sentence):
        ner_output_data = self.ner.predict(input_sentence)
        mentions = ner_output_data["mentions"]
        labels = ner_output_data["labels"]
        start_positions = ner_output_data["start_positions"]
        end_positions = ner_output_data["end_positions"]
        confidences = ner_output_data["confidences"]
        samples = []
        for i, mention in enumerate(mentions):
            record = {}
            record["label"] = "unknown"
            record["label_id"] = -1
            # LOWERCASE EVERYTHING !
            record["context_left"] = input_sentence[: start_positions[i]].lower()
            record["context_right"] = input_sentence[end_positions[i]:].lower()
            record["mention"] = mentions[i].lower()
            record["start_pos"] = int(start_positions[i])
            record["end_pos"] = int(end_positions[i])
            record["sent_idx"] = 0
            record["label"] = labels[i]
            record["confidence"] = confidences[i]
            samples.append(record)
        return samples

    def process_biencoder_dataloader(self, samples, tokenizer, biencoder_params):
        _, tensor_data = process_mention_data(
            samples,
            tokenizer,
            biencoder_params["max_context_length"],
            biencoder_params["max_cand_length"],
            silent=True,
            logger=None,
            debug=biencoder_params["debug"],
        )
        sampler = SequentialSampler(tensor_data)
        dataloader = DataLoader(
            tensor_data, sampler=sampler, batch_size=biencoder_params["eval_batch_size"]
        )
        return dataloader


if __name__ == "__main__":
    test = BLINKEntityLinker()
    text = input("insert text:")
    print(test.forward(text))
