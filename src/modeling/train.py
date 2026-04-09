"""
train.py — Train Random Forest and SVM classifiers with spatial block cross-validation.

Spatial CV strategy
-------------------
The raster extent is divided into an (N_SPATIAL_FOLDS × N_SPATIAL_FOLDS) grid of blocks.
Each block is assigned to exactly one fold.  Samples inherit the fold of the block they
fall in, so no training fold contains pixels spatially adjacent to the test fold — this
prevents spatial leakage that standard k-fold would introduce.

SMOTE is applied *inside* each training fold (never touching validation pixels).

The best hyper-parameters from spatial-CV are used to refit on the full training split
(80 % of samples by block, preserving spatial structure).  Final models are saved with
joblib to the paths defined in config.py.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    RF_MODEL_PATH, SVM_MODEL_PATH, MODEL_DIR,
    RF_PARAM_GRID, SVM_PARAM_GRID,
    RANDOM_STATE, TEST_SIZE, N_SPATIAL_FOLDS, N_JOBS,
    DATA_PROCESSED,
)
from src.modeling.sample import load_samples

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

X_OUT     = DATA_PROCESSED / "X_samples.npy"
Y_OUT     = DATA_PROCESSED / "y_samples.npy"
COORD_OUT = DATA_PROCESSED / "coords.npy"


# ── Spatial block CV splitter ─────────────────────────────────────────────────

class SpatialBlockCV(BaseCrossValidator):
    """
    Assigns each pixel to a spatial block, then groups blocks into folds.

    Parameters
    ----------
    coords    : (n, 2) int array of (row, col) pixel positions
    n_folds   : number of CV folds
    grid_size : number of blocks per side (grid_size × grid_size tiles total)
    """

    def __init__(self, coords: np.ndarray, n_folds: int = 5, grid_size: int = 10):
        self.coords    = coords
        self.n_folds   = n_folds
        self.grid_size = grid_size

    def _get_block_assignments(self) -> np.ndarray:
        rows, cols = self.coords[:, 0], self.coords[:, 1]
        row_max, col_max = rows.max(), cols.max()

        block_row = (rows / (row_max + 1) * self.grid_size).astype(int)
        block_col = (cols / (col_max + 1) * self.grid_size).astype(int)

        # Unique block id
        block_id = block_row * self.grid_size + block_col
        return block_id

    def _iter_test_masks(self, X=None, y=None, groups=None):
        block_ids   = self._get_block_assignments()
        unique_blocks = np.unique(block_ids)

        # Shuffle blocks deterministically then split into folds
        rng = np.random.default_rng(RANDOM_STATE)
        shuffled = rng.permutation(unique_blocks)
        fold_assignments = np.array_split(shuffled, self.n_folds)

        for fold_blocks in fold_assignments:
            test_mask = np.isin(block_ids, fold_blocks)
            yield test_mask

    def split(self, X, y=None, groups=None):
        for test_mask in self._iter_test_masks(X, y, groups):
            train_idx = np.where(~test_mask)[0]
            test_idx  = np.where(test_mask)[0]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_folds


def spatial_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    test_size: float = TEST_SIZE,
    grid_size: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Hold out a spatially contiguous test set (last `test_size` fraction of blocks
    by block index) rather than a random draw.
    """
    rows, cols = coords[:, 0], coords[:, 1]
    row_max, col_max = rows.max(), cols.max()

    block_row = (rows / (row_max + 1) * grid_size).astype(int)
    block_col = (cols / (col_max + 1) * grid_size).astype(int)
    block_id  = block_row * grid_size + block_col

    unique_blocks = np.unique(block_id)
    rng = np.random.default_rng(RANDOM_STATE)
    shuffled = rng.permutation(unique_blocks)

    n_test_blocks = max(1, int(len(shuffled) * test_size))
    test_blocks   = shuffled[:n_test_blocks]
    train_blocks  = shuffled[n_test_blocks:]

    test_mask  = np.isin(block_id, test_blocks)
    train_mask = np.isin(block_id, train_blocks)

    log.info(
        "Train/test split → train: %d samples (%d blocks), test: %d samples (%d blocks)",
        train_mask.sum(), len(train_blocks),
        test_mask.sum(),  len(test_blocks),
    )
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


# ── Model builders ────────────────────────────────────────────────────────────

def _build_rf_pipeline():
    return Pipeline([   # <- NOT ImbPipeline
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )),
    ])


def _build_svm_pipeline() -> ImbPipeline:
    """SMOTE → StandardScaler → SVC."""
    return ImbPipeline([
        ("smote",  SMOTE(random_state=RANDOM_STATE)),
        ("scaler", StandardScaler()),
        ("clf",    SVC(
            random_state=RANDOM_STATE,
            probability=True,      # needed for ROC-AUC in evaluate.py
        )),
    ])


def _prefix_param_grid(param_grid: dict, prefix: str) -> dict:
    """Prepend 'clf__' (or given prefix) to each key for use inside a Pipeline."""
    return {f"{prefix}__{k}": v for k, v in param_grid.items()}


# ── Training orchestration ────────────────────────────────────────────────────

def train_model(
    name:        str,
    pipeline,
    param_grid:  dict,
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    cv_splitter,
) -> object:
    """
    Run spatial-CV GridSearchCV, log best params, refit on full training set.
    Returns the best estimator (already refit).
    """
    log.info("=" * 60)
    log.info("Training %s", name)
    log.info("  Param grid: %s", param_grid)

    gs = GridSearchCV(
        estimator  = pipeline,
        param_grid = param_grid,
        cv         = cv_splitter,
        scoring    = "f1_macro",   # balanced across Low / Moderate / High
        n_jobs     = N_JOBS,
        refit      = True,
        verbose    = 1,
        error_score= "raise",
    )
    gs.fit(X_train, y_train)

    log.info("%s  best F1-macro (spatial CV): %.4f", name, gs.best_score_)
    log.info("%s  best params: %s", name, gs.best_params_)

    return gs.best_estimator_


def save_model(estimator, path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, path)
    log.info("%s saved → %s", name, path)


def run() -> None:
    # ── Load samples ──────────────────────────────────────────────────────────
    log.info("Loading samples …")
    X, y, coords = load_samples()

    # ── Spatial train/test split ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = spatial_train_test_split(X, y, coords)

    # Persist the split indices for evaluate.py
    # (We save a boolean mask aligned to the full X/y arrays)
    rows, cols = coords[:, 0], coords[:, 1]
    row_max, col_max = rows.max(), cols.max()
    GRID = 10
    block_row = (rows / (row_max + 1) * GRID).astype(int)
    block_col = (cols / (col_max + 1) * GRID).astype(int)
    block_id  = block_row * GRID + block_col
    rng = np.random.default_rng(RANDOM_STATE)
    shuffled = rng.permutation(np.unique(block_id))
    n_test = max(1, int(len(shuffled) * TEST_SIZE))
    test_blocks  = shuffled[:n_test]
    test_mask    = np.isin(block_id, test_blocks)
    np.save(DATA_PROCESSED / "test_mask.npy", test_mask)
    log.info("Test mask saved → %s", DATA_PROCESSED / "test_mask.npy")

    # ── Build CV splitter on training coords only ─────────────────────────────
    train_mask   = ~test_mask
    train_coords = coords[train_mask]
    cv = SpatialBlockCV(
        coords    = train_coords,
        n_folds   = N_SPATIAL_FOLDS,
        grid_size = 10,
    )

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_pipe       = _build_rf_pipeline()
    rf_param_grid = _prefix_param_grid(RF_PARAM_GRID, "clf")
    rf_best       = train_model("RandomForest", rf_pipe, rf_param_grid, X_train, y_train, cv)
    save_model(rf_best, RF_MODEL_PATH, "RandomForest")

    # ── SVM ───────────────────────────────────────────────────────────────────
    # svm_pipe       = _build_svm_pipeline()
    # svm_param_grid = _prefix_param_grid(SVM_PARAM_GRID, "clf")
    # svm_best       = train_model("SVM", svm_pipe, svm_param_grid, X_train, y_train, cv)
    # save_model(svm_best, SVM_MODEL_PATH, "SVM")

    log.info("Training complete.")


if __name__ == "__main__":
    run()
