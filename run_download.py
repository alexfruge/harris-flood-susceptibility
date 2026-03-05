#!/usr/bin/env python3
"""
run_download.py — Entry point 1: Download all raw datasets for Harris County.

Usage
-----
    uv run python run_download.py               # download everything
    uv run python run_download.py --skip rainfall soil  # skip slow sources
    uv run python run_download.py --only dem streams    # only these sources

Exit codes
----------
    0  all downloads succeeded
    1  one or more downloads failed
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make src importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent))

from src.data.download import run_all

VALID_SOURCES = ["dem", "landcover", "soil", "streams", "rainfall", "flood_labels"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download all raw data for Harris County flood susceptibility project."
    )
    p.add_argument(
        "--skip", nargs="+", choices=VALID_SOURCES, default=[],
        metavar="SOURCE",
        help="Sources to skip (space-separated).  E.g. --skip rainfall",
    )
    p.add_argument(
        "--only", nargs="+", choices=VALID_SOURCES, default=[],
        metavar="SOURCE",
        help="Only run these sources (overrides --skip).",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.only:
        skip = [s for s in VALID_SOURCES if s not in args.only]
    else:
        skip = args.skip

    results = run_all(skip=skip)

    failed = [name for name, path in results.items() if path is None and name not in skip]
    if failed:
        logging.error("The following downloads FAILED: %s", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())