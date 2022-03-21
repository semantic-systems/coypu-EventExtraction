import gdown


def download_from_google_drive(url: str, output: str):
    gdown.download(url, output)
    print(f"{url} saved to {output}")
