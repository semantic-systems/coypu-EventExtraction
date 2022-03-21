from stores.download import download_from_google_drive


def test_download_pretrained_model_from_google_drive(pretrained_model_google_drive_url, pretrained_model_output_path):
    download_from_google_drive(pretrained_model_google_drive_url, pretrained_model_output_path)
    assert len(pretrained_model_google_drive_url.listdir()) == 1
    # TODO: test by name
