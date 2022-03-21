import gdown
from pathlib import Path


def download_from_google_drive(url: str, output: str):
    gdown.download(url, output, fuzzy=True)
    print(f"{url} saved to {output}")


if __name__ == '__main__':
    url = "https://drive.google.com/file/d/18vOZP0kT9XRVxkC-C5fQp0JiirY5sRFl/view?usp=sharing"
    output = str(Path(__file__).parent.joinpath("models").absolute()) + "pretrained_event_detector.pt"
    download_from_google_drive(url, output)