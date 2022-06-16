# coypu-EventExtraction
Event Extraction module for deployment.

This repo contains both the event type detection and event argument extraction modules for inference. Currently, for the first two MVPs, these two modules are independently implemented, meaning it is error pruned - event type and arguments could be mismatched. 
The plan is to merge this two modules into one in the last MVP.

- [x] MVP 1: custom (initial) Event Detector and OpenIE Event Argument Extractor
- [ ] MVP 2: custom (better) Event Detector and custom/OpenIE Event Argument Extractor
- [ ] MVP 3: HiTeC Event Extractor

# Requirement
- python >= 3.7

Use your favorite virtual environment management tool and 
```
pip install -r requirements.txt
```

# Usage
## Interactive CMD interface
You can use the interactive command-line interface to enter tweet and retrieve the extracted event information.

In order to do so, you must first download the pretrained event detector checkpoint from 
google drive, with the following command:
```
python stores/download.py -link <link_to_google_drive> -path <path_to_dir_to_store_the_model>
```
Note that the default value for both `-link` and `-path` is provided as the following,
```
link: https://drive.google.com/file/d/1cZd9dxValoqwy_85ZTQMtnZW7m1mJ1wQ/view?usp=sharing 
path: ./../data/
```
Therefore you can simply run
```
python stores/download.py
```

Then, you can enter interactive mode by
```
python event_extractor.py -c <path_to_config>
```
for example,
```
python event_extractor.py -c ./config/version_one.yml
```
Note that the default value for `-c` has already been provided, which is `./config/version_one.yml`
Therefore you can simply run
```
python event_extractor.py 
```

## Deployment
Currently, only the basic version of event extractor is included. 
- **Event Detector**: an initial version on Google Drive, please run the download script to store the 
checkpoint locally. (A better trained version will be updated frequently)
- **Event Argument Extractor**: a public package OpenIE is used.

For deployment, please first instantiate the EventExtractor with the EventDetector and 
EventArgumentExtractor components with the above mentioned basic version with the following arguments. 
```
    config_path = <path_to_yaml_config>
    with open(config_path, "r") as f:
        config: Dict = yaml.safe_load(f)
        config: Config = validate(config)
    instantiator = Instantiator(config)
    event_extractor = instantiator()
```
Note that there is a `__call__` function implemented in the Instantiator to instantiate the EventExtractor.

With this, one can extract information from a single tweet with the `infer(tweet: str)` method.
```
    output = event_extractor.infer(tweet)
```
The output of the `infer` method will be the following dataclass,
```
@dataclass
class EventExtractorOutput:
    tweet: str
    event_type: Optional[str]
    event_arguments: Optional[List[str]]
    event_graph: Optional[List[List[str]]]
    wikidata_links: Optional[Dict[str, str]]
    timestamp: str
```
You can access to each attribute of the dataclass using dot syntax (e.g., `output.event_type`).
