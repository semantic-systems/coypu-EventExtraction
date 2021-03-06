from pathlib import Path

import pytest

from models.event_argument_extraction import OpenIEExtractor
from models.event_detection import EventDetector
from event_extractor import EventExtractor


@pytest.fixture()
def pretrained_model_google_drive_url():
    return "https://docs.google.com/document/d/1p89rZA3vlLAcMCQmjtpSEwujLbbMt92fZWk8a6T76O8/edit?usp=sharing"


@pytest.fixture()
def pretrained_model_output_path(tmpdir):
    return tmpdir.mkdir("outputs")


@pytest.fixture()
def downloaded_pretrained_model_path(pretrained_model_output_path):
    return str(pretrained_model_output_path)+"/mocked_checkpoint.pt"


@pytest.fixture()
def valid_pretrained_event_detector_path():
    return str(Path(__file__).parent.parent.joinpath("stores/models/pretrained_event_detector.pt"))


@pytest.fixture()
def invalid_pretrained_event_detector_path():
    return "/mocked_checkpoint.pt"


@pytest.fixture()
def valid_pretrained_event_argument_extractor_path():
    return str(Path(__file__).parent.parent.joinpath("stores/models/pretrained_event_argument_extractor.pt"))


@pytest.fixture()
def invalid_pretrained_event_argument_extractor_path():
    return "/mocked_checkpoint.pt"


@pytest.fixture()
def event_type_detector_instance(valid_pretrained_event_detector_path):
    return EventDetector(valid_pretrained_event_detector_path)


@pytest.fixture()
def event_argument_extractor_openie_instance():
    return OpenIEExtractor()


@pytest.fixture()
def event_extractor_instance(event_type_detector_instance, event_argument_extractor_openie_instance):
    return EventExtractor(event_type_detector_instance, event_argument_extractor_openie_instance)
