"""
run_model.py — CLI entry point for the Harris County flood susceptibility modelling stage.

Usage
-----
# Run the full modelling pipeline
uv run python run_model.py

# Skip sampling (e.g. samples already saved) and go straight to training
uv run python run_model.py --skip sample

# Run only evaluation
uv run python run_model.py --only evaluate

# Use the SVM for prediction rather than RF (default)
uv run python run_model.py --only predict --pred-model svm

Available step names
--------------------
  sample    extract pixel samples from feature stack + labels
  train     spatial CV, SMOTE, hyperparameter tuning, save models
  evaluate  metrics, confusion matrix, ROC-AUC, feature importance
  predict   run best model across full raster → susceptibility_map.tif
  maps      folium + matplotlib visualisations
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)

# Allow importing src.* from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

STEPS_ORDERED = ["sample", "train", "evaluate", "predict", "maps"]


def _run_step(name: str, pred_model: str = "rf") -> None:
    log.info("━" * 60)
    log.info("STEP: %s", name.upper())
    log.info("━" * 60)
    t0 = time.perf_counter()

    if name == "sample":
        from src.modeling.sample import run
        run()

    elif name == "train":
        from src.modeling.train import run
        run()

    elif name == "evaluate":
        from src.modeling.evaluate import run
        run()

    elif name == "predict":
        from src.modeling.predict import run
        run(use_model=pred_model)

    elif name == "maps":
        from src.visualization.maps import run
        run()

    else:
        raise ValueError(f"Unknown step: {name!r}. Choose from {STEPS_ORDERED}.")

    elapsed = time.perf_counter() - t0
    log.info("  ✓ %s completed in %.1f s", name, elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harris County flood susceptibility — modelling pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--skip",
        metavar="STEP",
        nargs="+",
        choices=STEPS_ORDERED,
        help="Steps to skip (space-separated).",
    )
    group.add_argument(
        "--only",
        metavar="STEP",
        nargs="+",
        choices=STEPS_ORDERED,
        help="Run only these steps (space-separated, in pipeline order).",
    )
    parser.add_argument(
        "--pred-model",
        choices=["rf", "svm"],
        default="rf",
        help="Model to use for the predict step (default: rf).",
    )

    args = parser.parse_args()

    # Resolve which steps to run
    if args.only:
        steps = [s for s in STEPS_ORDERED if s in args.only]
    elif args.skip:
        steps = [s for s in STEPS_ORDERED if s not in args.skip]
    else:
        steps = list(STEPS_ORDERED)

    if not steps:
        log.error("No steps to run — check --skip / --only arguments.")
        sys.exit(1)

    log.info("Pipeline steps: %s", " → ".join(steps))
    total_t0 = time.perf_counter()

    for step in steps:
        _run_step(step, pred_model=args.pred_model)

    total_elapsed = time.perf_counter() - total_t0
    log.info("━" * 60)
    log.info("All steps complete in %.1f s", total_elapsed)
    log.info("━" * 60)


if __name__ == "__main__":
    main()
