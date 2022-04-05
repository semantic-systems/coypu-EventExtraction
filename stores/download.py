import gdown
import argparse
from pathlib import Path
from dataclasses import dataclass


@dataclass
class LocalMetaConfig:
    google_drive_link: str
    directory_to_store: str


def parse():
    parser = argparse.ArgumentParser(description='Please input the google drive link to download the model checkpoint.'
                                                 'As well as where the model should be stored. ')

    parser.add_argument("-link", "--link",
                        help='Google Drive link to download the model checkpoint.',
                        default="https://drive.google.com/file/d/1cZd9dxValoqwy_85ZTQMtnZW7m1mJ1wQ/view?usp=sharing")
    parser.add_argument("-path", "--path",
                        help='Path to a directory where the model checkpoint is saved.'
                             'This should NOT be located within this repo because we want to store big files withinin '
                             '/data',
                        default="./../data")
    parser.parse_args()
    return parser.parse_args()


def download_from_google_drive(config: LocalMetaConfig):
    if not Path(config.directory_to_store).exists():
        Path(config.directory_to_store).mkdir(parents=True, exist_ok=False)
    gdown.download(config.google_drive_link,
                   str(Path(config.directory_to_store).joinpath("pretrained_event_detector.pt").absolute()),
                   fuzzy=True)


if __name__ == '__main__':
    args = parse()
    google_drive_link = args.link
    directory_to_store = args.path
    config = LocalMetaConfig(google_drive_link=google_drive_link, directory_to_store=directory_to_store)
    download_from_google_drive(config)
