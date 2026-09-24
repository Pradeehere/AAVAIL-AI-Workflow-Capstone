#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import requests

BASE = "https://raw.githubusercontent.com/aavail/ai-workflow-capstone/master"
TRAIN_MONTHS = [
    "2017-11", "2017-12",
    *[f"2018-{m:02d}" for m in range(1, 13)],
    *[f"2019-{m:02d}" for m in range(1, 8)],
]
PROD_MONTHS = [f"2019-{m:02d}" for m in range(8, 13)]


def download(folder: str, months: list[str], dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    for ym in months:
        name = f"invoices-{ym}.json"
        url = f"{BASE}/{folder}/{name}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        (dest / name).write_bytes(response.content)
        print("downloaded", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    root = Path(args.root)
    download("cs-train", TRAIN_MONTHS, root / "cs-train")
    download("cs-production", PROD_MONTHS, root / "cs-production")

if __name__ == "__main__":
    main()
