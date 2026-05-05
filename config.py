"""
GeoFidelity-Bench v2: Global configuration.

Major changes from v1:
  * H3 resolution 7 -> 8 (5 km^2 -> 0.7 km^2 per tile, neighborhood-scale)
  * 6 -> 12 tiles/city to keep city coverage roughly constant
  * 120 candidates fetched per tile, then 6-tier curation keeps ~10 per tile
  * Segmentation backbone swapped to Mask2Former Mapillary-Vistas (Cityscapes
    domain-shift was a v1 reviewer hazard for non-European cities)
"""
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
CACHE_DIR = ROOT / "data" / "cache"
REVIEW_DIR = ROOT / "data" / "review"
MAPILLARY_V2_DIR = DATA_DIR / "mapillary_v2"
GEN_DIR = ROOT / "generations"
for d in [DATA_DIR, PROCESSED_DIR, OUTPUT_DIR, CACHE_DIR, REVIEW_DIR,
          MAPILLARY_V2_DIR, GEN_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Benchmark scales
H3_RESOLUTION = 8                # ~0.74 km^2 per tile (neighborhood scale)
TILES_PER_CITY = 12              # finer tiles, more per city
CANDIDATES_PER_TILE = 120        # fetched from Mapillary, pre-filter
TARGET_IMAGES_PER_TILE = 10      # after full curation
MIN_IMAGES_PER_TILE = 6          # drop tile if survivors below this
GEN_IMAGES_PER_TILE = 4          # k=4 generations per method per tile
IMG_SIZE = 512

# Tier 1: Mapillary API query-time filters
TIER1_CAMERA_TYPE = "perspective"           # exclude fisheye/spherical
TIER1_IS_PANO = False
TIER1_MIN_SUN_ELEVATION_DEG = 20.0          # daylight cutoff
TIER1_SEARCH_RADII_M = [800, 1500, 2500, 4000]  # concentric fallback

# Tier 2: OSM road-type filter
TIER2_EXCLUDE_HIGHWAY_TAGS = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "service", "track", "raceway", "bus_guideway",
}
TIER2_INCLUDE_HIGHWAY_TAGS = {
    "residential", "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "living_street",
    "pedestrian", "footway", "cycleway",
}
TIER2_SNAP_RADIUS_M = 30                    # max snap distance

# Tier 3: SigLIP zero-shot scene classification
TIER3_SIGLIP_MODEL = "google/siglip-so400m-patch14-384"
TIER3_URBAN_PROMPT = "a daytime urban street scene with buildings and sidewalks"
TIER3_DISTRACTOR_PROMPTS = [
    "a highway or freeway with only cars",
    "a nighttime street scene",
    "inside a tunnel or parking garage",
    "a rural road with fields or forest",
    "an image dominated by a large truck or bus blocking the view",
    "a blurry or motion-blurred image",
    "an indoor scene",
    "a close-up of a sign or billboard",
]
TIER3_URBAN_MIN_SCORE = 0.35                # urban class prob threshold

# Tier 4: Semantic segmentation (Mapillary Vistas)
TIER4_SEG_MODEL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
TIER4_RATIOS = {                            # (min, max) pixel ratio bounds
    "sky":        (0.05, 0.45),
    "building":   (0.06, 0.60),
    "road":       (0.04, 0.50),
    "vehicle":    (0.00, 0.25),
    "vegetation": (0.00, 0.70),
}
TIER4_URBANNESS_MIN = 0.10                  # composite urbanness floor

# Tier 5: Low-level image quality
TIER5_MIN_LAPLACIAN_VAR = 80.0              # blur gate
TIER5_LUMINANCE_RANGE = (55, 215)           # mean luma on 0-255
TIER5_MIN_COLORFULNESS = 14.0               # Hasler-Susstrunk
TIER5_ASPECT_RATIO_RANGE = (0.5, 2.2)

# Tier 6: Manual review
TIER6_SAMPLE_PER_CITY = 30                  # stratified per-city manual sample

# City selection (25 cities, 6 continents)
CITIES = {
    # Europe
    "london":        {"lat": 51.5074, "lon":  -0.1278, "country": "GB", "driving": "left",  "tz": "Europe/London"},
    "paris":         {"lat": 48.8566, "lon":   2.3522, "country": "FR", "driving": "right", "tz": "Europe/Paris"},
    "berlin":        {"lat": 52.5200, "lon":  13.4050, "country": "DE", "driving": "right", "tz": "Europe/Berlin"},
    "rome":          {"lat": 41.9028, "lon":  12.4964, "country": "IT", "driving": "right", "tz": "Europe/Rome"},
    "amsterdam":     {"lat": 52.3676, "lon":   4.9041, "country": "NL", "driving": "right", "tz": "Europe/Amsterdam"},
    # North America
    "new_york":      {"lat": 40.7128, "lon": -74.0060, "country": "US", "driving": "right", "tz": "America/New_York"},
    "san_francisco": {"lat": 37.7749, "lon": -122.4194,"country": "US", "driving": "right", "tz": "America/Los_Angeles"},
    "toronto":       {"lat": 43.6532, "lon": -79.3832, "country": "CA", "driving": "right", "tz": "America/Toronto"},
    "mexico_city":   {"lat": 19.4326, "lon": -99.1332, "country": "MX", "driving": "right", "tz": "America/Mexico_City"},
    # South America
    "buenos_aires":  {"lat":-34.6037, "lon": -58.3816, "country": "AR", "driving": "right", "tz": "America/Argentina/Buenos_Aires"},
    "sao_paulo":     {"lat":-23.5505, "lon": -46.6333, "country": "BR", "driving": "right", "tz": "America/Sao_Paulo"},
    # Asia
    "tokyo":         {"lat": 35.6762, "lon": 139.6503, "country": "JP", "driving": "left",  "tz": "Asia/Tokyo"},
    "singapore":     {"lat":  1.3521, "lon": 103.8198, "country": "SG", "driving": "left",  "tz": "Asia/Singapore"},
    "mumbai":        {"lat": 19.0760, "lon":  72.8777, "country": "IN", "driving": "left",  "tz": "Asia/Kolkata"},
    "bangkok":       {"lat": 13.7563, "lon": 100.5018, "country": "TH", "driving": "left",  "tz": "Asia/Bangkok"},
    "seoul":         {"lat": 37.5665, "lon": 126.9780, "country": "KR", "driving": "right", "tz": "Asia/Seoul"},
    "shanghai":      {"lat": 31.2304, "lon": 121.4737, "country": "CN", "driving": "right", "tz": "Asia/Shanghai"},
    # Middle East / Africa
    "dubai":         {"lat": 25.2048, "lon":  55.2708, "country": "AE", "driving": "right", "tz": "Asia/Dubai"},
    "cape_town":     {"lat":-33.9249, "lon":  18.4241, "country": "ZA", "driving": "left",  "tz": "Africa/Johannesburg"},
    "nairobi":       {"lat": -1.2921, "lon":  36.8219, "country": "KE", "driving": "left",  "tz": "Africa/Nairobi"},
    "cairo":         {"lat": 30.0444, "lon":  31.2357, "country": "EG", "driving": "right", "tz": "Africa/Cairo"},
    "istanbul":      {"lat": 41.0082, "lon":  28.9784, "country": "TR", "driving": "right", "tz": "Europe/Istanbul"},
    # Oceania
    "sydney":        {"lat":-33.8688, "lon": 151.2093, "country": "AU", "driving": "left",  "tz": "Australia/Sydney"},
    "melbourne":     {"lat":-37.8136, "lon": 144.9631, "country": "AU", "driving": "left",  "tz": "Australia/Melbourne"},
    # Additional diversity
    "bogota":        {"lat":  4.7110, "lon": -74.0721, "country": "CO", "driving": "right", "tz": "America/Bogota"},
}

# Hard negative config
HARD_NEG_SAME_CITY_MIN_KM = 0.7      # raised because tiles are now 0.7 km^2
HARD_NEG_SAME_CITY = 1
HARD_NEG_SAME_CLIMATE = 1
HARD_NEG_RANDOM = 1

# Compute
DEVICE = "cuda"
BATCH_SIZE = 16
NUM_WORKERS = 4

# Generation
OPEN_SOURCE_MODELS = [
    "sdxl_base",
    "sd35_large",
    "flux_dev",
    "flux_schnell",
    "pixart_sigma",
    "hunyuan_dit",
]

PROMPT_TEMPLATE = (
    "A street-level photograph taken in {city}, {country}. "
    "The image shows a typical street scene with buildings, roads, and "
    "urban environment characteristic of this location. "
    "Photorealistic, daytime, clear weather."
)

# Human evaluation
HUMAN_EVAL_PAIRS = 200
HUMAN_RATERS_PER_PAIR = 3

# ---------------------------------------------------------------------------
# v3: Block-level benchmark (coexists with v2; uses V3_* prefix everywhere)
# ---------------------------------------------------------------------------
# Design: every city is carved into V3_BLOCKS_PER_CITY named OSM-way blocks,
# stratified by highway class. Each block is sampled at ~V3_SEARCH_STEP_M
# intervals along its polyline to avoid Mapillary's "bbox too large" 500s.
# Targets ~100 curated images per block, ~1000+ per city.

V3_DATA_DIR          = DATA_DIR / "mapillary_v3"
V3_PROCESSED_DIR     = PROCESSED_DIR / "v3"
V3_CACHE_DIR         = CACHE_DIR / "v3"
V3_GEN_DIR           = ROOT / "generations_v3"
V3_OVERPASS_CACHE    = CACHE_DIR / "overpass_blocks"
V3_BLOCKS_JSON       = V3_PROCESSED_DIR / "blocks_v3.json"
V3_BENCHMARK_JSON    = V3_PROCESSED_DIR / "benchmark_v3.json"
V3_TIER1_CSV         = V3_PROCESSED_DIR / "tier1_candidates.csv"
for _d in [V3_DATA_DIR, V3_PROCESSED_DIR, V3_CACHE_DIR, V3_GEN_DIR,
           V3_OVERPASS_CACHE]:
    _d.mkdir(parents=True, exist_ok=True)

# Block carving (data/carve_blocks.py)
V3_BLOCKS_PER_CITY   = 10
V3_BLOCK_STRATA = [
    # (label, osm_highway_tags, count)
    ("major",       ("primary", "primary_link",
                     "secondary", "secondary_link"), 3),
    ("residential", ("residential", "living_street",
                     "unclassified"),                3),
    ("pedestrian",  ("pedestrian", "footway",
                     "cycleway"),                    2),
    ("tertiary",    ("tertiary", "tertiary_link"),   2),
]
V3_MIN_WAY_LENGTH_M  = 500
V3_MAX_WAY_LENGTH_M  = 2000    # longer ways are truncated to first 2 km
V3_WAY_BUFFER_M      = 30      # lateral buffer around the polyline
V3_BLOCK_H3_RES      = 9       # res-9 ~0.1 km虏, edge ~170 m
V3_CITY_SEARCH_RADIUS_M = 6000  # max distance from city centre for block selection
V3_MIN_SEPARATION_M  = 400     # picked blocks must be 鈮?this apart
V3_OVERPASS_TIMEOUT_S = 180

# Block-level Mapillary fetch (data/download_mapillary_v3.py)
V3_SEARCH_BBOX_DEG   = 0.0045  # ~500 m; Mapillary safe bbox
V3_SEARCH_STEP_M     = 200     # polyline step: 200m step + 500m bbox => 1.25x overlap
V3_SEARCH_WORKERS    = 8       # concurrent Mapillary searches
V3_DOWNLOAD_WORKERS  = 12      # concurrent thumb downloads
V3_CANDIDATES_PER_BLOCK = 500  # over-sample before curation
V3_TARGET_IMAGES_PER_BLOCK = 100   # after all six tiers
V3_MIN_IMAGES_PER_BLOCK = 25       # drop block if survivors below this
                                    # (v2 used 6; we stay 4x stricter but low
                                    #  enough to keep Berlin / Nairobi / SF)

# 5-level hard negatives (block-aware)
V3_NEG_LEVELS = [
    "same_block_diff_images",        # random split within block
    "same_neighborhood_diff_block",  # different named way, same 鈮? km radius
    "same_city_diff_neighborhood",   # different part of the same city
    "same_driving_side_diff_city",   # driving-side matched, different city
    "random_city",                   # unrelated city
]
V3_NEIGHBORHOOD_RADIUS_M = 1000    # threshold for "same neighborhood"

# v3-relaxed T4 ratio gates. v2's gates (TIER4_RATIOS) were calibrated for
# place-unit-scale bbox sampling that could include highway/tunnel/rural
# frames; v3 already filters by OSM-way membership in `carve_blocks.py`,
# so the sky/road lower bounds were dropping legitimate pedestrian streets
# and building-canyon shots that have little visible sky or road.
V3_TIER4_RATIOS = {
    "sky":        (0.01, 0.55),
    "building":   (0.03, 0.70),
    "road":       (0.01, 0.55),
    "vehicle":    (0.00, 0.40),
    "vegetation": (0.00, 0.80),
}
V3_TIER4_URBANNESS_MIN = 0.05

# Prompt conditioning levels (generation/run_generation_v3.py)
V3_PROMPT_TEMPLATES = {
    # L0: city-only; reproduces v2 semantics and acts as a lower bound.
    "L0": (
        "A street-level photograph taken in {city}, {country}. "
        "The image shows a typical street scene with buildings, roads, and "
        "urban environment characteristic of this location. "
        "Photorealistic, daytime, clear weather."
    ),
    # L1: street + neighborhood + city.
    "L1": (
        "A street-level photograph taken on {street_name} in the "
        "{neighborhood} district of {city}, {country}. "
        "The image shows a typical street scene with buildings, roads, "
        "and urban environment characteristic of this block. "
        "Photorealistic, daytime, clear weather."
    ),
    # L2: L1 + explicit GPS coordinates.
    "L2": (
        "A street-level photograph taken on {street_name} in the "
        "{neighborhood} district of {city}, {country}, "
        "near GPS coordinates ({lat:.4f}, {lon:.4f}). "
        "The image shows a typical street scene with buildings, roads, "
        "and urban environment characteristic of this block. "
        "Photorealistic, daytime, clear weather."
    ),
    # GPS_ONLY: appendix ablation, expected to be weakest
    "GPS_ONLY": (
        "A street-level photograph taken near GPS coordinates "
        "({lat:.4f}, {lon:.4f}). "
        "The image shows a typical street scene with buildings, roads, "
        "and urban environment. "
        "Photorealistic, daytime, clear weather."
    ),
}
V3_PROMPT_LEVELS_MAIN = ["L0", "L1", "L2"]
V3_PROMPT_LEVELS_APPENDIX = ["GPS_ONLY"]
V3_GEN_IMAGES_PER_BLOCK = 4

# v3 human eval (broader: metric-human correlation)
V3_HUMAN_BLOCKS_SAMPLED = 25        # 5 cities 脳 5 blocks
V3_HUMAN_PAIRS_PER_BLOCK = 10       # 5 real + 5 gen pairs
V3_HUMAN_RATERS_PER_PAIR = 3
