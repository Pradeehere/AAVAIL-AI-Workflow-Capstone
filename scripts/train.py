#!/usr/bin/env python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pprint import pprint
from aavail.model import train_all

if __name__ == "__main__":
    pprint(train_all("data/cs-train"))
