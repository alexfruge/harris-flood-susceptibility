"""
predict.py — Tile-by-tile prediction across the full feature stack.

Memory strategy
---------------
rasterio windowed reads process the raster in strips of `BLOCK_ROWS` rows at a time,
keeping peak RAM proportional to strip height rather than the full raster.

Output
------
outputs/susceptibility_map.tif : uint8, same CRS / transform / grid as feature_stack.tif
    0 = NoData
    1 = Low
    2 = Moderate
    3 = High
"""

import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    FEATURE_STACK, SUSCEPTIBILITY_MAP,
    RF_MODEL_PATH, SVM_MODEL_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

NODATA_FEATURES = -9999.0
NODATA_OUT      = 0
BLOCK_ROWS      = 256        # number of rows per processing strip


def predict_raster(
    model_path:       Path = RF_MODEL_PATH,
    feature_path:     Path = FEATURE_STACK,
    output_path:      Path = SUSCEPTIBILITY_MAP,
    block_rows:       int  = BLOCK_ROWS,
) -> None:
    """
    Stream-predict class labels for every valid pixel in feature_path.

    Parameters
    ----------
    model_path    : path to a joblib-saved scikit-learn / imbalanced-learn pipeline
    feature_path  : 7-band feature stack GeoTIFF
    output_path   : destination for uint8 susceptibility map
    block_rows    : height of each processing strip (tune to available RAM)
    """
    log.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(feature_path) as src:
        profile = src.profile.copy()
        profile.update(
            dtype   = rasterio.uint8,
            count   = 1,
            nodata  = NODATA_OUT,
            compress= "lzw",
            predictor= 2,
        )
        height = src.height
        width  = src.width
        n_bands = src.count
        feat_nodata = src.nodata if src.nodata is not None else NODATA_FEATURES

        log.info("Raster: %d rows × %d cols, %d bands", height, width, n_bands)
        log.info("Writing susceptibility map → %s", output_path)

        with rasterio.open(output_path, "w", **profile) as dst:
            n_strips = (height + block_rows - 1) // block_rows

            for strip_idx in range(n_strips):
                row_off = strip_idx * block_rows
                actual_rows = min(block_rows, height - row_off)
                window = Window(
                    col_off = 0,
                    row_off = row_off,
                    width   = width,
                    height  = actual_rows,
                )

                # Read strip: shape (n_bands, actual_rows, width)
                strip = src.read(window=window).astype(np.float32)

                # Valid pixel mask: all bands non-nodata
                valid_mask = np.all(strip != float(feat_nodata), axis=0)   # (H, W)

                # Allocate output strip (nodata = 0)
                out_strip = np.zeros((actual_rows, width), dtype=np.uint8)

                n_valid = int(valid_mask.sum())
                if n_valid > 0:
                    # Flatten valid pixels → (n_valid, n_bands)
                    X_strip = strip[:, valid_mask].T   # (n_valid, 7)
                    preds   = model.predict(X_strip).astype(np.uint8)
                    out_strip[valid_mask] = preds

                dst.write(out_strip[np.newaxis, :, :], window=window)

                if (strip_idx + 1) % max(1, n_strips // 10) == 0 or strip_idx == n_strips - 1:
                    pct = 100.0 * (strip_idx + 1) / n_strips
                    log.info("  Progress: strip %d/%d  (%.0f %%)", strip_idx + 1, n_strips, pct)

    log.info("Susceptibility map written: %s", output_path)


def run(use_model: str = "rf") -> None:
    """
    Parameters
    ----------
    use_model : "rf" (default) or "svm"
    """
    model_path = RF_MODEL_PATH if use_model.lower() == "rf" else SVM_MODEL_PATH
    predict_raster(model_path=model_path)
    log.info("Prediction complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict flood susceptibility map.")
    parser.add_argument(
        "--model", choices=["rf", "svm"], default="rf",
        help="Which trained model to use (default: rf)",
    )
    args = parser.parse_args()
    # run(use_model=args.model)
    run(use_model="rf")
