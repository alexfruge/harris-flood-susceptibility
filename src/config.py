"""
config.py — Central configuration for Harris County flood susceptibility project.
Edit this file to adapt the pipeline to a new study area.
"""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

# ── Directories ───────────────────────────────────────────────────────────────
DATA_RAW        = ROOT / "data" / "raw"
DATA_PROCESSED  = ROOT / "data" / "processed"
DATA_LABELS     = ROOT / "data" / "labels"
OUTPUTS         = ROOT / "outputs"

for _d in (DATA_RAW, DATA_PROCESSED, DATA_LABELS, OUTPUTS):
    _d.mkdir(parents=True, exist_ok=True)

# ── Coordinate reference system ───────────────────────────────────────────────
CRS = "EPSG:32615"          # UTM Zone 15N — appropriate for Harris County, TX

# ── Study area ────────────────────────────────────────────────────────────────
# Harris County bounding box in WGS84 (lon_min, lat_min, lon_max, lat_max)
BBOX_WGS84 = (-95.9100, 29.4900, -94.9000, 30.1700)
WEST, SOUTH, EAST, NORTH = BBOX_WGS84

# ── Raster grid ───────────────────────────────────────────────────────────────
RESOLUTION_M = 30           # target cell size in metres

# ── DEM — Copernicus GLO-30 ───────────────────────────────────────────────────
# Public S3 bucket (no auth required for 30m tiles)
COP_DEM_S3_BASE = (
    "https://copernicus-dem-30m.s3.amazonaws.com"
    "/Copernicus_DSM_COG_10_{lat}_{lon}_DEM"
    "/Copernicus_DSM_COG_10_{lat}_{lon}_DEM.tif"
)
COP_DEM_TILES = [
    {"lat": "N29", "lon": "W096"},
    {"lat": "N29", "lon": "W095"},
    {"lat": "N30", "lon": "W096"},
    {"lat": "N30", "lon": "W095"},
]
DEM_RAW_DIR     = DATA_RAW / "dem"
DEM_MERGED      = DATA_PROCESSED / "dem_harris.tif"
SLOPE_RASTER    = DATA_PROCESSED / "slope_harris.tif"
TWI_RASTER      = DATA_PROCESSED / "twi_harris.tif"

# ── Land cover — NLCD 2021 ────────────────────────────────────────────────────
NLCD_WCS_URL = (
    "https://www.mrlc.gov/geoserver/wcs"
    "?SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
    "&COVERAGEID=mrlc_download__NLCD_2021_Land_Cover_L48_20230630"
    "&FORMAT=image/tiff"
    "&SUBSET=Long({west},{east})"
    "&SUBSET=Lat({south},{north})"
)
NLCD_RAW        = DATA_RAW / "nlcd_2021_harris_raw.tif"
NLCD_PROCESSED  = DATA_PROCESSED / "nlcd_harris.tif"

# ── Soil — USDA SSURGO ────────────────────────────────────────────────────────
SSURGO_WFS_URL = (
    "https://SDMDataAccess.sc.egov.usda.gov/Spatial/SDM.wfs"
    "?SERVICE=WFS&VERSION=1.1.0&REQUEST=GetFeature"
    "&TYPENAME=MapunitPolyExtended"
    "&FILTER=<Filter><PropertyIsEqualTo>"
    "<PropertyName>areasymbol</PropertyName><Literal>TX201</Literal>"
    "</PropertyIsEqualTo></Filter>"
)
SOIL_RAW_DIR    = DATA_RAW / "soil"
SOIL_SHP        = DATA_RAW / "soil" / "harris_soil.shp"
SOIL_RASTER     = DATA_PROCESSED / "soil_ksat_harris.tif"

# ── Streams — USGS NHD ────────────────────────────────────────────────────────
NHD_DOWNLOAD_URL = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography"
    "/NHDPlusHR/Beta/GDB/NHDPLUS_H_1207_HU4_GDB.zip"
)
NHD_RAW_DIR         = DATA_RAW / "nhd"
NHD_GDB             = DATA_RAW / "nhd" / "NHDPLUS_H_1207_HU4_GDB.gdb"
STREAMS_SHP         = DATA_RAW / "nhd" / "nhd_flowlines_harris.shp"
DIST_STREAM_RASTER  = DATA_PROCESSED / "dist_stream_harris.tif"

# ── Rainfall — CHIRPS v3 ──────────────────────────────────────────────────────
CHIRPS_BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-3.0/global_annual/tifs"
CHIRPS_YEARS    = list(range(1981, 2024))
CHIRPS_RAW_DIR  = DATA_RAW / "chirps"
CHIRPS_MEAN     = DATA_PROCESSED / "rainfall_mean_harris.tif"

# ── Flood labels — FEMA NFHL ──────────────────────────────────────────────────
FEMA_FIPS = "48201"             # Harris County, TX

# Bulk GDB zip download from FEMA MSC (may require portal auth — REST is fallback)
FEMA_NFHL_URL = (
    f"https://msc.fema.gov/portal/downloadProduct?productID=NFHL_{FEMA_FIPS}_20240101"
)

# REST MapServer endpoint — used for paginated bbox queries (no auth needed)
FEMA_NFHL_DIRECT = (
    "https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer/28/query"
)

# Bounding box as (minx, miny, maxx, maxy) in WGS84 — used by REST query
BBOX_GEO = BBOX_WGS84   # alias so flood_labels.py can import BBOX_GEO

# Raw download directory for flood label source files
RAW_LABELS_DIR      = DATA_RAW / "fema"

# Final processed labels directory (where rasterised output goes)
LABELS_DIR          = DATA_LABELS

DFO_URL             = "https://floodobservatory.colorado.edu/temp/FloodArchive.zip"
FEMA_RAW_DIR        = DATA_RAW / "fema"
FEMA_SHP            = DATA_RAW / "fema" / "harris_fema_flood.geojson"
DFO_SHP             = DATA_RAW / "fema" / "FloodArchive.shp"
FLOOD_LABELS_RASTER = DATA_LABELS / "flood_labels_harris.tif"

# ── Feature stack ─────────────────────────────────────────────────────────────
FEATURE_STACK   = DATA_PROCESSED / "feature_stack.tif"
FEATURE_NAMES   = [
    "elevation",
    "slope",
    "twi",
    "land_cover",
    "soil_ksat",
    "dist_stream",
    "rainfall_mean",
]

# ── Model outputs ─────────────────────────────────────────────────────────────
MODEL_DIR           = OUTPUTS / "models"
RF_MODEL_PATH       = MODEL_DIR / "rf_model.joblib"
SVM_MODEL_PATH      = MODEL_DIR / "svm_model.joblib"
SUSCEPTIBILITY_MAP  = OUTPUTS / "susceptibility_map.tif"
REPORT_DIR          = OUTPUTS / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Class labels ──────────────────────────────────────────────────────────────
CLASS_NAMES         = {1: "Low", 2: "Moderate", 3: "High"}
FEMA_HIGH_ZONES     = {"A", "AE", "AH", "AO", "AR", "A99", "VE", "V"}
FEMA_MODERATE_ZONES = {"B", "X500", "0.2 PCT ANNUAL CHANCE FLOOD HAZARD"}

# ── ML settings ───────────────────────────────────────────────────────────────
RANDOM_STATE     = 42
TEST_SIZE        = 0.20
N_SPATIAL_FOLDS  = 5
N_JOBS           = -1

RF_PARAM_GRID = {
    "n_estimators":      [200, 400],
    "max_depth":         [None, 20, 40],
    "min_samples_leaf":  [1, 5],
    "class_weight":      ["balanced"],
}

SVM_PARAM_GRID = {
    "C":            [0.1, 1, 10],
    "kernel":       ["rbf"],
    "gamma":        ["scale"],
    "class_weight": ["balanced"],
}