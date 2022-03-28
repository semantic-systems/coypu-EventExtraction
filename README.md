# coypu-EventExtraction
Event Extraction module for deployment.

This repo contains both the event type detection and event argument extraction modules for inference. Currently, for the first two MVPs, these two modules are independently implemented, meaning it is error pruned - event type and arguments could be mismatched. 
The plan is to merge this two modules into one in the last MVP.

- [x] MVP 1: custom (initial) Event Detector and OpenIE Event Argument Extractor
- [ ] MVP 2: custom (better) Event Detector and custom/OpenIE Event Argument Extractor
- [ ] MVP 3: HiTeC Event Extractor

# Requirement
- Jave
- python = 3.7

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
python stores/download.py
```
Then, you can enter interactive mode by
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
    event_detector_model_path = "stores/models/pretrained_event_detector.pt"
    event_argument_extractor_model_path = "openie"
    instantiator = Instantiator(event_detector_model_path, event_argument_extractor_model_path)
    event_extractor = instantiator()
```
Note that there is a `__call__` function implemented in the Instantiator to instantiate the EventExtractor.

With this, one can extract information from a single tweet with the `infer(tweet: str)` method.
```
    output = event_extractor.infer(tweet, output_file_path)
```
