# coypu-EventExtraction
Event Extraction module for deployment.

This repo contains both the event type detection and event argument extraction modules for inference. Currently, for the first two MVPs, these two modules are independently implemented, meaning it is error pruned - event type and arguments could be mismatched. 
The plan is to merge this two modules into one in the last MVP.

- [ ] MVP 1
- [ ] MVP 2
- [ ] MVP 3

# Usage
for deployment, one must first instantiate the event extractor with the event detector and event 
argument extractor components. For any torch models, the pretrained networks will be stored somewhere (tbd).
The Instantiator will have access to the model storage and load the models from there. For the OpenIE model (as in the first version of 
event argument extractor), there will be no storage. 
```
    tweet = "You are on fire, run!"
    event_detector_model_path = "stores/models/xxx"
    output_file_path = "outputs/output.json"
    event_argument_extractor_model_path = "openie"
    instantiator = Instantiator(event_detector_model_path, event_argument_extractor_model_path)
    event_extractor = instantiator()
    output = event_extractor.infer(tweet, output_file_path)
```