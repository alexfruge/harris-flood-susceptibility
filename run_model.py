#!/usr/bin/env python3
"""
run_model.py — Entry point 3: Train, evaluate, predict, and visualize.

Steps
-----
1. Train Random Forest (+ SMOTE) and SVM baseline
2. Spatial cross-validation, metrics, feature importance
3. Predict full county susceptibility map
4. Generate static PNG + interactive Folium HTML map

Usage
-----
    uv run python run_model.py
    uv run python run_model.py --skip-svm        # RF only
    uv run python run_model.py --skip-eval       # skip CV (faster)
    uv run python run_model.py --skip-train      # use existing models
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser(description="Train and evaluate flood susceptibility models.")
    p.add_argument("--skip-train",   action="store_true", help="Skip training, load existing models.")
    p.add_argument("--skip-svm",     action="store_true", help="Skip SVM baseline.")
    p.add_argument("--skip-eval",    action="store_true", help="Skip cross-validation.")
    p.add_argument("--skip-predict", action="store_true", help="Skip prediction raster.")
    p.add_argument("--skip-maps",    action="store_true", help="Skip visualization.")
    p.add_argument("--no-smote",     action="store_true", help="Disable SMOTE for RF training.")
    p.add_argument("--log-level",    default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("run_model")
    errors = []

    # Step 1: Train
    if not args.skip_train:
        log.info("=" * 60)
        log.info("STEP 1: Training models")
        log.info("=" * 60)
        try:
            from src.modeling.train import load_feature_matrix, train_random_forest, train_svm
            X, y, _ = load_feature_matrix()
            train_random_forest(X, y, use_smote=not args.no_smote)
            if not args.skip_svm:
                train_svm(X, y)
        except Exception as exc:
            log.error("Training failed: %s", exc, exc_info=True)
            errors.append("train")

    # Step 2: Evaluate
    if not args.skip_eval:
        log.info("=" * 60)
        log.info("STEP 2: Spatial cross-validation")
        log.info("=" * 60)
        try:
            from src.modeling.evaluate import run as run_eval
            metrics = run_eval()
            log.info("Metrics summary:\n%s", metrics.groupby("model")[["f1_macro","kappa"]].mean().to_string())
        except Exception as exc:
            log.error("Evaluation failed: %s", exc, exc_info=True)
            errors.append("eval")

    # Step 3: Predict
    if not args.skip_predict:
        log.info("=" * 60)
        log.info("STEP 3: Predicting county-wide susceptibility")
        log.info("=" * 60)
        try:
            from src.modeling.predict import run as run_predict
            map_path, proba_path = run_predict()
            log.info("Map: %s", map_path)
        except Exception as exc:
            log.error("Prediction failed: %s", exc, exc_info=True)
            errors.append("predict")

    # Step 4: Visualize
    if not args.skip_maps:
        log.info("=" * 60)
        log.info("STEP 4: Generating maps")
        log.info("=" * 60)
        try:
            from src.visualization.maps import run as run_maps
            results = run_maps()
            for k, v in results.items():
                log.info("  %s -> %s", k, v)
        except Exception as exc:
            log.error("Visualization failed: %s", exc, exc_info=True)
            errors.append("maps")

    if errors:
        log.error("run_model completed with errors: %s", errors)
        return 1
    log.info("All modeling steps complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())