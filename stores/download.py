import gdown
from pathlib import Path


def download_from_google_drive(url: str, output: str):
    gdown.download(url, output, fuzzy=True)


if __name__ == '__main__':
    url = "https://drive.google.com/file/d/1cZd9dxValoqwy_85ZTQMtnZW7m1mJ1wQ/view?usp=sharing"
    output = str(Path(__file__).parent.joinpath("models").absolute()) + "/pretrained_event_detector.pt"
    download_from_google_drive(url, output)