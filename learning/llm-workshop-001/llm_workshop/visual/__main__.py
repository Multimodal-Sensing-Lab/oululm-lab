"""Launch the local Python desktop teaching lab; no network or web server."""
import argparse
from .app import launch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--topic', type=int, choices=range(9), default=0)
args = parser.parse_args()
launch(args.topic)
