#!/usr/bin/env python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aavail.eda import run_eda

if __name__ == "__main__":
    print(run_eda("data/cs-train"))
