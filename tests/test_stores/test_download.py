from stores.download import download_from_google_drive


def test_download_pretrained_model_from_google_drive(pretrained_model_google_drive_url,
                                                     pretrained_model_output_path,
                                                     downloaded_pretrained_model_path):
    download_from_google_drive(pretrained_model_google_drive_url, downloaded_pretrained_model_path)
    assert len(pretrained_model_output_path.listdir()) == 1
