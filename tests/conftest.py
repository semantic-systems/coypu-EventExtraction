import pytest


@pytest.fixture()
def pretrained_model_google_drive_url():
    return "https://docs.google.com/document/d/1p89rZA3vlLAcMCQmjtpSEwujLbbMt92fZWk8a6T76O8/edit?usp=sharing"


@pytest.fixture()
def pretrained_model_output_path(tmpdir):
    return tmpdir.mkdir("outputs")
