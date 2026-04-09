"""
maps.py — Visualise the flood susceptibility map.

Outputs
-------
outputs/reports/susceptibility_map_static.png   matplotlib figure (UTM projection)
outputs/reports/susceptibility_map_folium.html  interactive folium map (WGS84)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    SUSCEPTIBILITY_MAP, REPORT_DIR, CLASS_NAMES, CRS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Colour scheme: 0=NoData(transparent), 1=Low, 2=Moderate, 3=High
PALETTE = {
    0: (0,   0,   0,   0),        # transparent
    1: (244, 231,  61, 255),      # yellow — Low
    2: (230, 126,  34, 255),      # orange — Moderate
    3: (192,  57,  43, 255),      # red    — High
}

MPL_COLORS = ["#f4e73d", "#e67e22", "#c0392b"]   # Low, Moderate, High


# ── Reproject utility ─────────────────────────────────────────────────────────

def _reproject_to_wgs84(src_path: Path) -> tuple[np.ndarray, dict]:
    """
    Reproject susceptibility_map.tif to WGS84 in memory.
    Returns (data_array, profile_dict) where data_array is (H, W) uint8.
    """
    with rasterio.open(src_path) as src:
        dst_crs = "EPSG:4326"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(
            crs       = dst_crs,
            transform = transform,
            width     = width,
            height    = height,
        )

        with MemoryFile() as memfile:
            with memfile.open(**profile) as mem:
                reproject(
                    source      = rasterio.band(src, 1),
                    destination = rasterio.band(mem, 1),
                    src_transform = src.transform,
                    src_crs       = src.crs,
                    dst_transform = transform,
                    dst_crs       = dst_crs,
                    resampling    = Resampling.nearest,
                )
                data = mem.read(1)
    return data, profile


# ── Static matplotlib figure ──────────────────────────────────────────────────

def make_static_map(
    susc_path:  Path = SUSCEPTIBILITY_MAP,
    output_dir: Path = REPORT_DIR,
) -> Path:
    """
    Render a classified susceptibility map using matplotlib.
    Downsamples large rasters for display efficiency.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(susc_path) as src:
        data = src.read(1)                 # (H, W) uint8
        bounds = src.bounds

    # Downsample if very large
    MAX_DIM = 2000
    h, w = data.shape
    if max(h, w) > MAX_DIM:
        step = max(h, w) // MAX_DIM + 1
        data = data[::step, ::step]
        log.info("Downsampled for display: %d×%d → %d×%d", h, w, *data.shape)

    # Build discrete colormap  (classes 1, 2, 3; nodata=0 → transparent)
    cmap = ListedColormap(MPL_COLORS)
    norm = BoundaryNorm(boundaries=[0.5, 1.5, 2.5, 3.5], ncolors=3)

    display = np.ma.masked_equal(data.astype(float), 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        display, cmap=cmap, norm=norm,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
    )

    # Legend
    patches = [
        mpatches.Patch(color=MPL_COLORS[i], label=CLASS_NAMES[i + 1])
        for i in range(3)
    ]
    ax.legend(handles=patches, loc="lower right", title="Susceptibility", fontsize=10)

    ax.set_title("Harris County — Flood Susceptibility", fontsize=14, fontweight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    fig.tight_layout()

    out_path = output_dir / "susceptibility_map_static.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Static map → %s", out_path)
    return out_path


# ── Folium interactive map ────────────────────────────────────────────────────

def make_folium_map(
    susc_path:  Path = SUSCEPTIBILITY_MAP,
    output_dir: Path = REPORT_DIR,
) -> Path:
    """
    Build an interactive folium map with an image overlay of the susceptibility raster.
    The raster is reprojected to WGS84 and converted to a PNG image overlay.
    """
    try:
        import folium
    except ImportError:
        log.error("folium is not installed — skipping interactive map.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Reprojecting to WGS84 for folium …")
    data_wgs84, profile = _reproject_to_wgs84(susc_path)

    # Convert raster to RGBA image for folium ImageOverlay
    h, w = data_wgs84.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cls_val, color in PALETTE.items():
        mask = data_wgs84 == cls_val
        rgba[mask] = color

    # Derive bounds from the reprojected raster's affine transform.
    # profile["transform"] is the WGS84 affine for the output grid, so:
    #   top-left  = (transform.c, transform.f)
    #   bottom-right = top-left + (width * pixel_w, height * pixel_h)
    # This is the correct extent of the warped pixels, not the original config BBOX.
    t = profile["transform"]
    west  = t.c
    north = t.f
    east  = west  + t.a * profile["width"]
    south = north + t.e * profile["height"]   # t.e is negative (north-up)
    center_lat = (south + north) / 2
    center_lon = (west  + east)  / 2

    # Save RGBA array as PNG in a temp location then embed
    from PIL import Image
    import io, base64

    img_pil = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    img_url = f"data:image/png;base64,{img_b64}"

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10,
                   tiles="CartoDB positron")

    folium.raster_layers.ImageOverlay(
        image   = img_url,
        bounds  = [[south, west], [north, east]],
        opacity = 0.70,
        name    = "Flood Susceptibility",
    ).add_to(m)

    # Legend as HTML
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:10px 14px; border-radius:6px;
                border:1px solid #ccc; font-size:13px; font-family:sans-serif;">
        <b>Flood Susceptibility</b><br>
        <span style="background:#f4e73d;display:inline-block;width:14px;height:14px;
              margin-right:6px;border:1px solid #999;"></span>Low<br>
        <span style="background:#e67e22;display:inline-block;width:14px;height:14px;
              margin-right:6px;border:1px solid #999;"></span>Moderate<br>
        <span style="background:#c0392b;display:inline-block;width:14px;height:14px;
              margin-right:6px;border:1px solid #999;"></span>High
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)

    out_path = output_dir / "susceptibility_map_folium.html"
    m.save(str(out_path))
    log.info("Folium map → %s", out_path)
    return out_path


def run() -> None:
    make_static_map()
    make_folium_map()
    log.info("Visualisation complete.")


if __name__ == "__main__":
    run()