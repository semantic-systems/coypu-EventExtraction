import os
import re
import gdown
import gradio as gr
import pandas as pd
import plotly
import torch
from torch import tensor
from typing import Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from schemes import EventDetectorOutput
from models import BaseComponent
from models.event_detection.src.models.SingleLabelSequenceClassification import SingleLabelSequenceClassification
from stores.ontologies.event_type_wikidata_links_trecis import EVENT_TYPE_WIKIDATA_LINKS
from transformers import logging
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from gdeltdoc import GdeltDoc, Filters
from typing import Dict, Union, List


State = Union[None, Dict[str, Union[str, Dict, List]]]
logging.set_verbosity_error()


os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEFAULT_OUTPUT_PATH = '/data/event_detector/'
DEFAULT_LANGUAGE_MODELS_PATH = '/data/language_models/'


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
    def __init__(
            self,
            path_to_pretrained_model: str = os.path.join(DEFAULT_OUTPUT_PATH, "crisisbert_w_oos_linear.pt")
    ):
        super(EventDetector).__init__()
        # self.prepare(path_to_pretrained_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path_to_pretrained_model, map_location=self.device)
        checkpoint['config']['model']["from_pretrained"] = \
            os.path.join(DEFAULT_LANGUAGE_MODELS_PATH, "CoyPu-CrisisLM-v1")
        self.model = SingleLabelSequenceClassification(checkpoint['config'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.index_label_map = checkpoint['index_label_map']
        self.model.to(self.device)
        self.pca = PCA(n_components=2)

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
        return EventDetectorOutput(tweet=tweet, event_type=event_type, wikidata_link=wikidata_link)

    def forward_batch(self, data: dict) -> str:
        sentences = data["title"]
        if len(sentences) == 0:
            description = f"No English articles found with the provided keywords!\n"
            return description
        tokenized_text = self.model.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        input_ids: tensor = tokenized_text["input_ids"].to(self.model.device)
        attention_masks: tensor = tokenized_text["attention_mask"].to(self.model.device)
        labels = None
        input_feature: InputFeature = InputFeature(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
        output: SingleLabelClassificationForwardOutput = self.model.forward(input_feature, "test")
        prediction_logits = output.prediction_logits
        prediction_indices = output.prediction_logits.argmax(1).detach().numpy()
        event_types = [self.index_label_map[str(prediction)] for prediction in prediction_indices]

        prediction_logits_array = self.reduce_with_PCA(prediction_logits.detach().numpy())
        df = pd.DataFrame({"PC 1": prediction_logits_array[:, 0], "PC 2": prediction_logits_array[:, 1],
                           "prediction_logits": prediction_indices, "event type": event_types, "sentences":sentences,
                           "url": data["url"], "timestamp": data["timestamp"]})

        db = DBSCAN(eps=3, min_samples=2).fit(prediction_logits_array)
        labels = db.labels_
        df["clustered label"] = labels
        df["clustered label"] = df["clustered label"].astype(str)
        fig_cls = px.scatter(df, x="PC 1", y="PC 2", color="event type", hover_data=['sentences', "url"]) #,width=1000, height=700)
        fig_cls.update_traces(marker_size=10)
        fig_cluster = px.scatter(df, x="PC 1", y="PC 2", color="clustered label", hover_data=['sentences', "url"]) #, width=1065, height=700)
        fig_cluster.update_traces(marker_size=10)
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        fig_cls.update_layout(
            title="Event Visualization with Classification Result",
            title_x=0.5,
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            font=dict(
                size=12
            ),
            hoverlabel=dict(
                font_size=12,
            ),
            margin=dict(l=15, r=15, t=15, b=15),
            paper_bgcolor="#E8E8DC"
        )
        fig_cluster.update_layout(
            title="Event Visualization with Clustering Result",
            title_x=0.5,
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            font=dict(
                size=12
            ),
            hoverlabel=dict(
                font_size=12,
            ),
            margin=dict(l=15, r=15, t=15, b=15),
            paper_bgcolor="#E8E8DC"
        )
        fig_cls.write_json("./fig_cls.json")
        fig_cluster.write_json("./fig_cluster.json")
        description = f"Estimated number of clusters: {n_clusters_}\n"
        return description

    def reduce_with_PCA(self, features):
        return self.pca.fit_transform(features)

    @property
    def __version__(self):
        return "2.0.0"

    @staticmethod
    def prepare(path_to_pretrained_model):
        if not Path(path_to_pretrained_model).exists():
            if not Path(DEFAULT_OUTPUT_PATH).exists():
                Path(DEFAULT_OUTPUT_PATH).mkdir()
            gdown.download(url="https://drive.google.com/file/d/1Hj_s7UfKYOMszQYAYLy0iNFN1qD1wxrH/view?usp=sharing&confirm=t",
                           output=path_to_pretrained_model, fuzzy=True)
        path_str = os.path.join(DEFAULT_LANGUAGE_MODELS_PATH, "CoyPu-CrisisLM-v1")
        if not Path(path_str).exists():
            if not Path(DEFAULT_LANGUAGE_MODELS_PATH).exists():
                Path(DEFAULT_LANGUAGE_MODELS_PATH).mkdir()
            gdown.download_folder(
                url="https://drive.google.com/drive/folders/1u6Mthkr4ffVNSjPn3F_B49axTsCHwRv8?usp=sharing&confirm=t",
                output=path_str)


def get_event_timeline_plot(df):
    fig = go.Figure(go.Scatter(
        x=list(df.keys()),
        y=list(df.values())
    ))

    fig.update_xaxes(
        rangeslider_visible=True,
        tickformatstops=[
            # dict(dtickrange=[None, 1000], value="%H:%M:%S.%L ms"),
            # dict(dtickrange=[1000, 60000], value="%H:%M:%S s"),
            dict(dtickrange=[60000, 3600000], value="%H:%M m"),
            dict(dtickrange=[3600000, 86400000], value="%H:%M h"),
            dict(dtickrange=[86400000, 604800000], value="%e. %b d"),
            dict(dtickrange=[604800000, "M1"], value="%e. %b w"),
            dict(dtickrange=["M1", "M12"], value="%b '%y M"),
            dict(dtickrange=["M12", None], value="%Y Y")
        ]
    )
    return fig


class GdeltFunctions:
    def __init__(self):
        self.api = GdeltDoc()

    @staticmethod
    def datetime_range(start, end, delta):
        current = start
        while current < end:
            yield current
            current += delta

    def get_feed(self, query: str, lang: str = "English") -> Tuple[Dict, str]:
        '''Searches for feeds with the given query and returns one randomly'''
        data = {}
        if query == '':
            query = ""

        start_date = datetime.today() - timedelta(1)
        end_date = datetime.today() #- timedelta(1)
        f = Filters(
            keyword=query,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            num_records=250
        )

        # Search for articles matching the filters
        articles = self.api.article_search(f)
        try:
            english_articles = articles[articles['language'] == lang]
        except:
            english_articles = []
        description = f"{len(articles)} articles found with the keyword {query}, among which {len(english_articles)} articles are in English.\n"
        # timestamp = [datetime.strftime(date, "%Y%m%dT%H%M%SZ") for date in english_articles["seendate"].values]
        data["title"] = [self.clean_feed(title) for title in english_articles['title'].values]
        data["timestamp"] = english_articles["seendate"].values
        data["url"] = english_articles["url"].values
        # midnight = datetime.combine(datetime.today(), datetime.min.time())
        # yesterday_midnight = midnight - timedelta(days=1)
        # timeline = [dt for dt in self.datetime_range(yesterday_midnight, midnight, timedelta(minutes=15))]
        # timeline_count = {time.strftime("%Y-%m-%d %H:%M:%S"): 0 for time in timeline}
        # for timestamp in data["timestamp"]:
        #     for i, t in enumerate(timeline):
        #         if i < len(timeline)-1:
        #             time_object = datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ")
        #             if timeline[i] <= time_object < timeline[i+1]:
        #                 timeline_count[timeline[i].strftime("%Y-%m-%d %H:%M:%S")] += 1
        if len(english_articles) == 0:
            description = ""
        return data, description

    # stolen from previous code ;)
    @staticmethod
    def clean_feed(feed: str):
        '''
        Utility function to clean feed text by removing links, special characters
        using simple regex statements.
        '''
        if feed.startswith("RT @") :
            feed = feed.replace("RT ", "")
        feed = re.sub(" . ",".", feed)
        feed = re.sub(" : ",":", feed)
        feed = re.sub(" / ","/", feed)
        feed = re.sub(" , ",", ", feed)
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", feed).split())


if __name__ == "__main__":
    api = GdeltFunctions()
    model = EventDetector()
    # data, _ = api.get_feed("Hamburg")
    # model.forward_batch(data)

    def run(keyword):
        descriptions = ""
        data, description = api.get_feed(keyword)
        descriptions += description
        fig_cls_dict, fig_cluster_dict, description = model.forward_batch(data)
        descriptions += description
        descriptions += "\n Note:\n " \
                        "- oos refers to an out-of-scope class.\n" \
                        "- PC 1 and PC 2 refers to the first and the second principal components of the sentence embeddings, when reduced to two dimensions.\n"
        fig_cls = plotly.io.read_json("./fig_cls.json")
        fig_cluster = plotly.io.read_json("./fig_cluster.json")
        return descriptions, fig_cls, fig_cluster

    with gr.Blocks() as d:
        with gr.Row():
            gr.Markdown(
                value="This process takes about 10 seconds to complete. The following steps are executed if you click the submit botton. \n "
                      "- It queries 250 articles from GDELT (the max. for a single query in GDELT).\n"
                      "- Only articles written in English are selected for further analyses. \n "
                      "- Titles of the articles will be fed into the event type detector. \n"
                      "- A unsupervised clustering algorithm takes sentence embeddings as input and output cluster assignments. \n"
                      "- Visualization of both the classified result and clustering result will be displayed.")
        with gr.Column():
            input_box = gr.Textbox(placeholder="Enter a keyword here...", label="Keyword to query GDELT")
            output_box_description = gr.Markdown(label="Description")
            plot_cls = gr.Plot(label="Classification Result").style()
            plot_cluster = gr.Plot(label="Clustering Result").style()
            # plot_timeline = gr.Plot(label="Event Timeline").style()

    demo = gr.Interface(fn=run,
                        inputs=input_box,
                        outputs=[output_box_description, plot_cls, plot_cluster])
    demo.launch(share=True, show_error=True)
