#!/usr/bin/env python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pprint import pprint
from aavail.monitor import evaluate_production

if __name__ == "__main__":
    pprint(evaluate_production("all", "data/cs-train", "data/cs-production"))
