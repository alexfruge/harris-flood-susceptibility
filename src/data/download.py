"""
download.py — Master downloader: runs all six data-source modules in sequence.
Called by run_download.py (project entry point 1).
"""
from __future__ import annotations
import logging, sys
from pathlib import Path

log = logging.getLogger(__name__)


def run_all(skip: list[str] | None = None) -> dict[str, Path | None]:
    """Download all data sources. Pass skip=['rainfall'] to omit slow sources."""
    skip = skip or []
    results: dict[str, Path | None] = {}
    sources = {
        "dem":          _run_dem,
        "landcover":    _run_landcover,
        "soil":         _run_soil,
        "streams":      _run_streams,
        "rainfall":     _run_rainfall,
        "flood_labels": _run_flood_labels,
    }
    for name, fn in sources.items():
        if name in skip:
            log.info("Skipping %s (requested).", name)
            results[name] = None
            continue
        log.info("=" * 55)
        log.info("Starting download: %s", name.upper())
        log.info("=" * 55)
        try:
            results[name] = fn()
            log.info("OK  %s -> %s", name, results[name])
        except Exception as exc:
            log.error("FAIL %s: %s", name, exc, exc_info=True)
            results[name] = None

    log.info("\n%s\nDOWNLOAD SUMMARY\n%s", "="*55, "="*55)
    for name, path in results.items():
        log.info("  %-15s %s", name, f"OK -> {path}" if path else "FAILED/SKIPPED")
    return results


def _run_dem():
    from src.data import dem
    return dem.run()

def _run_landcover():
    from src.data import landcover
    return landcover.run()

def _run_soil():
    from src.data import soil
    soil.build_ksat_layer()
    from src.config import RAW_SOIL_DIR
    return RAW_SOIL_DIR / "ssurgo_harris_ksat.geojson"

def _run_streams():
    from src.data import streams
    return streams.run()

def _run_rainfall():
    from src.data import rainfall
    return rainfall.run()

def _run_flood_labels():
    from src.data import flood_labels
    return flood_labels.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run_all()
