import argparse


def parse():
    parser = argparse.ArgumentParser(description='Please input the path to the event detector and event argument '
                                                 'extractor checkpoint.'
                                                 'This will open an interactive cmd interface for Event Extraction.')

    parser.add_argument("-ed", "--event_detector_path",
                        help='Path to a pretrained event detector checkpoint. If your checkpoint in on Google Cloud,'
                             'you can use the ./stores/download.py script to download it locally.',
                        default="stores/models/pretrained_event_detector.pt")
    parser.add_argument("-eae", "--event_argument_extractor_path",
                        help='Path to a pretrained event argument extractor checkpoint.'
                             'Alternatively, you can use OpenIEExtractor which is a public package.',
                        default="openie")
    parser.parse_args()
    return parser.parse_args()