import pytest


@pytest.fixture()
def pretrained_model_google_drive_url():
    return "..."


@pytest.fixture()
def pretrained_model_output_path(tmpdir):
    return tmpdir.mkdir("outputs")
