"""
School Zone Speed Limit Detection Pipeline
============================================
HERE Berlin Hackathon 2026

3-Stage Pipeline:
  Stage 1: Sign Validation   — Vision AI confirms school zone + speed limit signs in images
  Stage 2: Classification    — Permanent vs Conditional (time modifiers)
  Stage 3: Zone Extent       — Calculate how far the speed limit zone extends

Supports: Anthropic Claude Sonnet OR Google Gemini (set via env var or CLI)
"""

import os
import sys
import json
import time
import base64
import re
import math
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, mapping
from shapely.ops import nearest_points
import folium
from folium.plugins import MarkerCluster
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─── Configuration ──────────────────────────────────────────────────────────
BASE_DIR = Path(r"c:\Users\Lenovo\Documents\Hackathon\berlin_hackathon_data")
OUTPUT_DIR = Path(r"c:\Users\Lenovo\Documents\Hackathon\output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Data files
SCHOOL_ZONE_SIGNS_CSV = BASE_DIR / "PotsdamBerlin_schoolzone_speedlimit_signs.csv"
REMAINING_SIGNS_CSV = BASE_DIR / "PotsdamBerlin_remaining_speedlimit_signs.csv"
SCHOOL_POI_CSV = BASE_DIR / "PotsdamBerlin_school_pointsofinterest.csv"
ROAD_GEOMETRY_CSV = BASE_DIR / "PotsdamBerlin_road_geometry.csv"
DOWNLOAD_SUMMARY_CSV = BASE_DIR / "download_summary.csv"
MAPILLARY_DIR = BASE_DIR / "mapillary_images"

# Vision API config
VISION_API = os.environ.get("VISION_API", "gemini").lower()  # "anthropic" or "gemini"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Processing limits
MAX_IMAGES_PER_LOCATION = 3  # Send up to 3 images per detection point
RATE_LIMIT_DELAY = 4.5       # Seconds between API calls (Gemini free tier is 15 RPM)
MAX_ZONE_EXTENT_M = 500      # Maximum zone extent in meters (cap)
DEFAULT_ZONE_EXTENT_M = 150  # Default when no termination sign found

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "pipeline.log")
    ]
)
log = logging.getLogger(__name__)


# ─── Vision API Clients ─────────────────────────────────────────────────────

VISION_PROMPT = """You are analyzing a German street-level image near a school zone. 
Look carefully at ALL road signs visible in the image.

Answer these questions in valid JSON format:

{
  "school_zone_sign_visible": true/false,
  "speed_limit_sign_visible": true/false,
  "speed_limit_value": <integer or null>,
  "is_conditional": true/false,
  "time_window": "<string or null, e.g. 'Mo-Fr 07:00-17:00'>",
  "day_modifier": "<string or null, e.g. 'Mo-Fr' or 'Schultage'>",
  "distance_text": "<string or null, e.g. '300m' or 'Zone'>",
  "end_of_zone_sign_visible": true/false,
  "is_30_zone": true/false,
  "confidence": <float 0-1>,
  "notes": "<brief description of what you see>"
}

Key rules:
- "school_zone_sign_visible": Look for the triangular warning sign with children symbol (StVO sign 136)
- "speed_limit_sign_visible": Look for circular speed limit signs (typically 30 km/h)
- "is_conditional": true if there's a supplemental plate with time/day information
- "is_30_zone": true if this is a "Zone 30" sign (rectangular) rather than a point speed limit
- "end_of_zone_sign_visible": true if you see end-of-speed-limit or end-of-restrictions signs
- Only report what you actually see. If image is unclear, set confidence low.

IMPORTANT: Return ONLY valid JSON, no markdown, no extra text."""


def encode_image(image_path: str) -> str:
    """Read and base64-encode an image file."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def analyze_with_anthropic(image_paths: list[str]) -> dict:
    """Send images to Anthropic Claude for analysis."""
    import anthropic
    
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    content = []
    for img_path in image_paths:
        img_data = encode_image(img_path)
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": img_data
            }
        })
    content.append({"type": "text", "text": VISION_PROMPT})
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": content}]
    )
   
    text = response.content[0].text
    # Extract JSON from the response
    return parse_vision_response(text)


def analyze_with_gemini(image_paths: list[str]) -> dict:
    """Send images to Google Gemini for analysis."""
    import google.generativeai as genai
    from PIL import Image
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
   
    parts = []
    for img_path in image_paths:
        img = Image.open(img_path)
        # Convert to RGB to strip MPO/exotic formats which Gemini rejects
        img = img.convert("RGB")
        img.format = "JPEG" # Force format for the Gemini SDK
        parts.append(img)
    parts.append(VISION_PROMPT)
    
    response = model.generate_content(parts)
    text = response.text
    return parse_vision_response(text)


def parse_vision_response(text: str) -> dict:
    """Parse JSON from vision API response."""
    default = {
        "school_zone_sign_visible": False,
        "speed_limit_sign_visible": False,
        "speed_limit_value": None,
        "is_conditional": False,
        "time_window": None,
        "day_modifier": None,
        "distance_text": None,
        "end_of_zone_sign_visible": False,
        "is_30_zone": False,
        "confidence": 0.0,
        "notes": "Failed to parse response"
    }
    
    try:
        # Try to find JSON in the response
        # Remove markdown code blocks if present
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()
        
        result = json.loads(text)
        # Merge with defaults
        for k, v in default.items():
            if k not in result:
                result[k] = v
        return result
    except json.JSONDecodeError:
        # Try to extract JSON from text
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                for k, v in default.items():
                    if k not in result:
                        result[k] = v
                return result
            except json.JSONDecodeError:
                pass
        default["notes"] = f"Parse failed: {text[:200]}"
        return default

def analyze_images(image_paths: list[str]) -> dict:
    """Route to the configured vision API."""
    import time
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            if VISION_API == "anthropic":
                return analyze_with_anthropic(image_paths)
            else:
                return analyze_with_gemini(image_paths)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                wait_time = 60
                import re
                match = re.search(r'retry in (\d+(?:\.\d+)?)s', error_msg)
                if match:
                    wait_time = max(10, int(float(match.group(1))) + 2)
                log.warning(f"  Rate limited, waiting {wait_time}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                log.warning(f"  Vision API error: {e}")
                time.sleep(RATE_LIMIT_DELAY)
                if attempt == max_retries - 1:
                    raise
    return parse_vision_response("{}")


# ─── Data Loading ───────────────────────────────────────────────────────────

def load_data():
    """Load all datasets and return as a dict."""
    log.info("Loading datasets...")
    
    # School zone sign detections
    signs = pd.read_csv(SCHOOL_ZONE_SIGNS_CSV)
    log.info(f"  School zone signs: {len(signs)} rows")
    
    # Remaining speed limit signs (for zone termination)
    remaining = pd.read_csv(REMAINING_SIGNS_CSV)
    log.info(f"  Remaining signs: {len(remaining)} rows")
    
    # School POI locations
    schools = pd.read_csv(SCHOOL_POI_CSV)
    log.info(f"  School POIs: {len(schools)} rows")
    
    # Road geometry
    roads = pd.read_csv(ROAD_GEOMETRY_CSV)
    log.info(f"  Road segments: {len(roads)} rows")
    
    # Download summary (image mapping)
    downloads = pd.read_csv(DOWNLOAD_SUMMARY_CSV)
    log.info(f"  Download summary: {len(downloads)} rows")
    
    return {
        "signs": signs,
        "remaining": remaining,
        "schools": schools,
        "roads": roads,
        "downloads": downloads
    }


def resolve_image_paths(output_path_str: str) -> list[str]:
    """
    Convert image paths from download_summary (which use original machine path)
    to local paths in the mapillary_images directory.
    """
    if pd.isna(output_path_str) or not output_path_str:
        return []
    
    paths = []
    for p in str(output_path_str).split(";"):
        p = p.strip()
        if not p:
            continue
        # Extract just the filename
        filename = Path(p).name
        local_path = MAPILLARY_DIR / filename
        if local_path.exists():
            paths.append(str(local_path))
    
    return paths


def deduplicate_signs(signs: pd.DataFrame) -> pd.DataFrame:
    """
    Group signs by location (pole_latitude, pole_longitude) and aggregate.
    Multiple rows at the same pole may represent different sign components
    (speed limit + school zone + supplemental modifier).
    """
    # Group by pole location
    grouped = signs.groupby(
        ["pole_latitude", "pole_longitude", "pole_bearing"], 
        as_index=False
    ).agg({
        "iso_code": "first",
        "weekof": "first",
        "ndetections": "sum",
        "speedlimit_gfrgroup": lambda x: "|".join(sorted(set(x.dropna()))),
        "schoolzone_gfrgroup": lambda x: "|".join(sorted(set(x.dropna()))),
        "supplemental_gfrgroup": lambda x: "|".join(sorted(set(str(v) for v in x.dropna() if str(v) != "nan")))
    })
    
    log.info(f"  Deduplicated: {len(signs)} rows -> {len(grouped)} unique locations")
    return grouped


def extract_speed_limit(gfrgroup: str) -> int:
    """Extract speed limit value from gfrgroup string like 'SpeedLimit2V30'."""
    match = re.search(r"V(\d+)", gfrgroup)
    if match:
        return int(match.group(1))
    return 0


# ─── Stage 1: Sign Validation ──────────────────────────────────────────────

def stage1_validate(data: dict) -> pd.DataFrame:
    """
    Stage 1: Validate detected school zone signs using vision AI.
    
    For each detection, find the corresponding Mapillary images and
    send them to the vision API to confirm:
    - Is there really a school zone sign?
    - Is there a speed limit sign?
    """
    log.info("=" * 60)
    log.info("STAGE 1: Sign Validation")
    log.info("=" * 60)
    
    signs = deduplicate_signs(data["signs"])
    downloads = data["downloads"]
    
    # Build a lookup from (lat, lon) → image paths
    # The download_summary has input_lat, input_lon matching sign locations
    image_lookup = {}
    for _, row in downloads.iterrows():
        key = (round(row["input_lat"], 6), round(row["input_lon"], 6))
        if row["status"] == "ok" and pd.notna(row.get("output_path")):
            paths = resolve_image_paths(row["output_path"])
            if paths:
                if key not in image_lookup:
                    image_lookup[key] = []
                image_lookup[key].extend(paths)
    
    log.info(f"  Image lookup: {len(image_lookup)} locations with images")
    
    results = []
    api_available = bool(ANTHROPIC_API_KEY or GEMINI_API_KEY)
    
    if not api_available:
        log.warning("  No API key set! Using sensor data only (no vision validation).")
    
    for idx, row in tqdm(signs.iterrows(), total=len(signs), desc="Stage 1: Validating"):
        key = (round(row["pole_latitude"], 6), round(row["pole_longitude"], 6))
        
        # Find matching images
        images = image_lookup.get(key, [])
        
        # Also try nearby keys (within small tolerance)
        if not images:
            for dk, dv in image_lookup.items():
                if abs(dk[0] - key[0]) < 0.0001 and abs(dk[1] - key[1]) < 0.0001:
                    images = dv
                    break
        
        # Limit images per location
        images = images[:MAX_IMAGES_PER_LOCATION]
        
        # Extract sensor-based info
        speed = extract_speed_limit(row["speedlimit_gfrgroup"])
        has_school_zone = "SchoolZone" in str(row["schoolzone_gfrgroup"])
        has_time_mod = "TimeModifier" in str(row["supplemental_gfrgroup"])
        has_day_mod = "DayModifier" in str(row["supplemental_gfrgroup"])
        
        # Vision analysis
        vision_result = None
        if images and api_available:
            try:
                vision_result = analyze_images(images)
                time.sleep(RATE_LIMIT_DELAY)
            except Exception as e:
                log.warning(f"  Vision API error at ({key}): {e}")
                vision_result = None
        
        # Combine sensor + vision data
        result = {
            "pole_latitude": row["pole_latitude"],
            "pole_longitude": row["pole_longitude"],
            "pole_bearing": row["pole_bearing"],
            "ndetections": row["ndetections"],
            "speedlimit_gfrgroup": row["speedlimit_gfrgroup"],
            "schoolzone_gfrgroup": row["schoolzone_gfrgroup"],
            "supplemental_gfrgroup": row["supplemental_gfrgroup"],
            "speed_limit_km": speed,
            "sensor_school_zone": has_school_zone,
            "sensor_time_modifier": has_time_mod,
            "sensor_day_modifier": has_day_mod,
            "has_images": len(images) > 0,
            "num_images": len(images),
            "image_paths": ";".join(images) if images else "",
        }
        
        if vision_result:
            result.update({
                "vision_school_zone": vision_result.get("school_zone_sign_visible", False),
                "vision_speed_limit": vision_result.get("speed_limit_sign_visible", False),
                "vision_speed_value": vision_result.get("speed_limit_value"),
                "vision_conditional": vision_result.get("is_conditional", False),
                "vision_time_window": vision_result.get("time_window"),
                "vision_day_modifier": vision_result.get("day_modifier"),
                "vision_distance": vision_result.get("distance_text"),
                "vision_end_zone": vision_result.get("end_of_zone_sign_visible", False),
                "vision_is_30_zone": vision_result.get("is_30_zone", False),
                "vision_confidence": vision_result.get("confidence", 0),
                "vision_notes": vision_result.get("notes", ""),
            })
        else:
            result.update({
                "vision_school_zone": None,
                "vision_speed_limit": None,
                "vision_speed_value": None,
                "vision_conditional": None,
                "vision_time_window": None,
                "vision_day_modifier": None,
                "vision_distance": None,
                "vision_end_zone": None,
                "vision_is_30_zone": None,
                "vision_confidence": None,
                "vision_notes": "No vision analysis (no API key or no images)",
            })
        
        results.append(result)
    
    df = pd.DataFrame(results)
    log.info(f"  Stage 1 complete: {len(df)} locations processed")
    return df


# ─── Stage 2: Classification ───────────────────────────────────────────────

def stage2_classify(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 2: Classify each detection as:
    - Valid / Invalid school zone
    - Permanent / Conditional
    - Extract time windows
    """
    log.info("=" * 60)
    log.info("STAGE 2: Classification")
    log.info("=" * 60)
    
    classifications = []
    
    for _, row in df.iterrows():
        # Determine validity
        # A valid school zone needs BOTH school zone sign + speed limit
        sensor_valid = row["sensor_school_zone"] and row["speed_limit_km"] > 0
        
        # If we have vision data, use it to confirm/override
        if row["vision_school_zone"] is not None:
            vision_valid = row["vision_school_zone"] and row["vision_speed_limit"]
            # Consensus: valid if either sensor OR vision confirms (with high confidence)
            confidence = row.get("vision_confidence", 0) or 0
            if confidence > 0.6:
                is_valid = vision_valid
            else:
                is_valid = sensor_valid
        else:
            is_valid = sensor_valid
        
        # Filter out "30 Zone" (not in scope per PPT slide 31)
        is_30_zone = False
        if row.get("vision_is_30_zone"):
            is_30_zone = True
        # Also check sensor data for CalmingZone indicators
        if "CalmingZone" in str(row.get("speedlimit_gfrgroup", "")):
            is_30_zone = True
        
        # Determine permanent vs conditional
        sensor_conditional = row["sensor_time_modifier"] or row["sensor_day_modifier"]
        vision_conditional = row.get("vision_conditional", False) or False
        
        is_conditional = sensor_conditional or vision_conditional
        
        # Extract time window
        time_window = None
        if row.get("vision_time_window"):
            time_window = row["vision_time_window"]
        elif sensor_conditional:
            # Sensor detected a modifier but we don't know the exact time
            time_window = "Unknown (sensor detected modifier)"
        
        # Day modifier
        day_modifier = None
        if row.get("vision_day_modifier"):
            day_modifier = row["vision_day_modifier"]
        elif row["sensor_day_modifier"]:
            day_modifier = "Unknown (sensor detected day modifier)"
        
        # Speed limit value (prefer vision, fallback to sensor)
        speed_limit = row.get("vision_speed_value") or row["speed_limit_km"]
        
        # Distance from sign (if visible)
        distance_text = row.get("vision_distance")
        
        classifications.append({
            "is_valid_school_zone": is_valid and not is_30_zone,
            "is_30_zone_excluded": is_30_zone,
            "classification": "conditional" if is_conditional else "permanent",
            "final_speed_limit_km": speed_limit,
            "time_window": time_window,
            "day_modifier": day_modifier,
            "zone_distance_text": distance_text,
            "validation_source": "vision+sensor" if row["vision_school_zone"] is not None else "sensor_only",
        })
    
    class_df = pd.DataFrame(classifications)
    result = pd.concat([df, class_df], axis=1)
    
    valid = result["is_valid_school_zone"].sum()
    conditional = result[result["is_valid_school_zone"]]["classification"].value_counts().get("conditional", 0)
    permanent = result[result["is_valid_school_zone"]]["classification"].value_counts().get("permanent", 0)
    excluded = result["is_30_zone_excluded"].sum()
    
    log.info(f"  Valid school zones: {valid}")
    log.info(f"    Permanent: {permanent}")
    log.info(f"    Conditional: {conditional}")
    log.info(f"  Excluded (30 Zone): {excluded}")
    log.info(f"  Invalid: {len(result) - valid - excluded}")
    
    return result


# ─── Stage 3: Zone Extent ──────────────────────────────────────────────────

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates in meters."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))


def parse_road_geometry(geom_str: str) -> LineString:
    """Parse LINESTRING WKT from road geometry CSV."""
    try:
        from shapely import wkt
        return wkt.loads(geom_str)
    except Exception:
        return None


def _vectorized_haversine(lat1, lon1, lat2_arr, lon2_arr):
    """Vectorized haversine: distance from one point to arrays of points (meters)."""
    import numpy as np
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2_arr)
    dphi = np.radians(lat2_arr - lat1)
    dlam = np.radians(lon2_arr - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def _vectorized_bearing_to(lat1, lon1, lat2_arr, lon2_arr):
    """Vectorized bearing from (lat1,lon1) to arrays of points (degrees 0-360)."""
    import numpy as np
    return np.degrees(np.arctan2(lon2_arr - lon1, lat2_arr - lat1)) % 360


def _find_nearest_sign_in_direction(lat, lon, bearing, candidates_lat, candidates_lon, 
                                      candidates_labels, max_dist=500, angle_tol=90):
    """Find the nearest candidate sign within max_dist and within ±angle_tol of bearing."""
    import numpy as np
    if len(candidates_lat) == 0:
        return None, None
    
    dists = _vectorized_haversine(lat, lon, candidates_lat, candidates_lon)
    mask = (dists > 10) & (dists < max_dist)
    
    if not mask.any():
        return None, None
    
    bearings_to = _vectorized_bearing_to(lat, lon, candidates_lat, candidates_lon)
    bearing_diff = np.abs(bearings_to - bearing) % 360
    bearing_diff = np.where(bearing_diff > 180, 360 - bearing_diff, bearing_diff)
    mask &= (bearing_diff < angle_tol)
    
    if not mask.any():
        return None, None
    
    dists_masked = np.where(mask, dists, np.inf)
    best_idx = np.argmin(dists_masked)
    return dists_masked[best_idx], candidates_labels[best_idx]


def stage3_zone_extent(df: pd.DataFrame, data: dict) -> pd.DataFrame:
    """
    Stage 3: Calculate zone extent for each valid school zone.
    
    Uses OPTIMIZED spatial methods:
    - geopandas sjoin_nearest for road matching (instead of brute-force)
    - numpy vectorized haversine for sign search (instead of row-by-row)
    
    Methods (per PPT slide 31):
    1. If distance text visible (e.g., "300m"), use that
    2. Look for termination signs (EndSpeedLimit, EndBUA) nearby
    3. Look for new speed limit signs that would override
    4. Default bounding logic: cap at DEFAULT_ZONE_EXTENT_M
    """
    import numpy as np
    
    log.info("=" * 60)
    log.info("STAGE 3: Zone Extent Calculation")
    log.info("=" * 60)
    
    roads = data["roads"]
    remaining = data["remaining"]
    
    # ── Parse road geometries ──
    log.info("  Parsing road geometries...")
    roads["geometry"] = roads["link_geometry"].apply(parse_road_geometry)
    roads_gdf = gpd.GeoDataFrame(roads, geometry="geometry", crs="EPSG:4326")
    roads_gdf = roads_gdf.dropna(subset=["geometry"])
    log.info(f"  Valid road segments: {len(roads_gdf)}")
    
    # ── Build sign GeoDataFrame for spatial join ──
    valid_mask = df["is_valid_school_zone"] == True
    valid_df = df[valid_mask].copy()
    
    if len(valid_df) > 0:
        sign_gdf = gpd.GeoDataFrame(
            valid_df,
            geometry=gpd.points_from_xy(valid_df["pole_longitude"], valid_df["pole_latitude"]),
            crs="EPSG:4326"
        )
        
        # ── Nearest road via sjoin_nearest (FAST spatial index) ──
        log.info("  Finding nearest roads via spatial index...")
        nearest = gpd.sjoin_nearest(
            sign_gdf[["geometry"]], 
            roads_gdf[["geometry", "road_name", "id"]], 
            how="left", 
            distance_col="road_dist"
        )
        # Map results back by original index
        road_name_map = nearest["road_name"].to_dict()
        road_id_map = nearest["id"].to_dict()
    else:
        road_name_map = {}
        road_id_map = {}
    
    # ── Pre-compute termination sign arrays (numpy) ──
    termination_types = [
        "EndSpeedLimit", "EndBUA", "EndCalmingZone", 
        "EndAllRestrictions", "EndSpeedLimit2V"
    ]
    term_signs = remaining[
        remaining["speedlimit_gfrgroup"].str.contains("|".join(termination_types), na=False)
    ].copy()
    term_lat = term_signs["pole_latitude"].values.astype(np.float64)
    term_lon = term_signs["pole_longitude"].values.astype(np.float64)
    term_labels = term_signs["speedlimit_gfrgroup"].values
    log.info(f"  Termination signs: {len(term_signs)} (vectorized)")
    
    # ── Pre-compute override speed limit sign arrays ──
    new_sl = remaining[
        remaining["speedlimit_gfrgroup"].str.match(r"^SpeedLimit", na=False) &
        ~remaining["speedlimit_gfrgroup"].str.contains("End", na=False)
    ].copy()
    sl_lat = new_sl["pole_latitude"].values.astype(np.float64)
    sl_lon = new_sl["pole_longitude"].values.astype(np.float64)
    sl_labels = new_sl["speedlimit_gfrgroup"].values
    log.info(f"  Override speed limit signs: {len(new_sl)} (vectorized)")
    
    # ── Process each sign ──
    zone_extents = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Stage 3: Zone extent"):
        if not row.get("is_valid_school_zone"):
            zone_extents.append({
                "zone_extent_m": 0,
                "zone_method": "n/a",
                "zone_end_lat": None,
                "zone_end_lon": None,
                "nearest_road_name": None,
                "nearest_road_id": None,
            })
            continue
        
        lat, lon = row["pole_latitude"], row["pole_longitude"]
        bearing = row["pole_bearing"]
        extent = None
        method = None
        
        # Method 1: Distance text from vision
        if row.get("zone_distance_text"):
            dist_match = re.search(r"(\d+)\s*m", str(row["zone_distance_text"]))
            if dist_match:
                extent = min(int(dist_match.group(1)), MAX_ZONE_EXTENT_M)
                method = f"distance_plate ({row['zone_distance_text']})"
        
        # Method 2: Nearest termination sign (vectorized)
        if extent is None and len(term_lat) > 0:
            d, label = _find_nearest_sign_in_direction(
                lat, lon, bearing, term_lat, term_lon, term_labels, MAX_ZONE_EXTENT_M
            )
            if d is not None:
                extent = d
                method = f"termination_sign ({label})"
        
        # Method 3: New speed limit sign override (vectorized)
        if extent is None and len(sl_lat) > 0:
            d, label = _find_nearest_sign_in_direction(
                lat, lon, bearing, sl_lat, sl_lon, sl_labels, MAX_ZONE_EXTENT_M
            )
            if d is not None:
                extent = d
                method = f"new_speed_limit ({label})"
        
        # Method 4: Default cap
        if extent is None:
            extent = DEFAULT_ZONE_EXTENT_M
            method = "default_cap"
        
        # ── Snap zone line to actual road geometry ──
        road_id = road_id_map.get(idx, "")
        road_name = road_name_map.get(idx, "Unknown")
        zone_coords = None  # Will store actual road coordinates for the zone line
        
        # Find the nearest road's geometry and interpolate along it
        matching_roads = roads_gdf[roads_gdf["id"] == road_id] if road_id else gpd.GeoDataFrame()
        
        if len(matching_roads) > 0 and matching_roads.iloc[0]["geometry"] is not None:
            road_geom = matching_roads.iloc[0]["geometry"]
            sign_point = Point(lon, lat)
            
            # Project sign onto road and get position along line
            proj_dist = road_geom.project(sign_point, normalized=True)
            
            # Road length in degrees, convert extent to approximate degrees
            road_len_m = haversine_distance(
                road_geom.coords[0][1], road_geom.coords[0][0],
                road_geom.coords[-1][1], road_geom.coords[-1][0]
            )
            
            if road_len_m > 0:
                extent_frac = min(extent / max(road_len_m, 1), 1.0)
                end_frac = min(proj_dist + extent_frac, 1.0)
                end_point = road_geom.interpolate(end_frac, normalized=True)
                start_point = road_geom.interpolate(proj_dist, normalized=True)
                
                end_lat = end_point.y
                end_lon = end_point.x
                
                # Extract road coords between start and end for a nice line
                zone_coords = [[start_point.y, start_point.x], [end_point.y, end_point.x]]
            else:
                # Fallback to bearing
                end_lat = lat + (extent / 111320) * math.cos(math.radians(bearing))
                end_lon = lon + (extent / (111320 * math.cos(math.radians(lat)))) * math.sin(math.radians(bearing))
        else:
            # Fallback: straight line along bearing
            end_lat = lat + (extent / 111320) * math.cos(math.radians(bearing))
            end_lon = lon + (extent / (111320 * math.cos(math.radians(lat)))) * math.sin(math.radians(bearing))
        
        zone_extents.append({
            "zone_extent_m": round(extent, 1),
            "zone_method": method,
            "zone_end_lat": round(end_lat, 7),
            "zone_end_lon": round(end_lon, 7),
            "nearest_road_name": road_name,
            "nearest_road_id": road_id,
        })
    
    extent_df = pd.DataFrame(zone_extents)
    result = pd.concat([df, extent_df], axis=1)
    
    valid = result[result["is_valid_school_zone"] == True]
    log.info("  Zone extent methods used:")
    if len(valid) > 0:
        for method, count in valid["zone_method"].value_counts().items():
            log.info(f"    {method}: {count}")
    
    return result


# ─── Output: GeoJSON + Folium Map ──────────────────────────────────────────

def generate_geojson(df: pd.DataFrame) -> str:
    """Generate enriched GeoJSON with all pipeline results."""
    log.info("Generating GeoJSON output...")
    
    features = []
    for _, row in df.iterrows():
        speed = row.get("final_speed_limit_km", row.get("speed_limit_km", 0))
        if isinstance(speed, pd.Series):
            speed = speed.iloc[0]
        speed = int(speed) if pd.notna(speed) else None
        
        # Create point feature for each sign
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["pole_longitude"], row["pole_latitude"]]
            },
            "properties": {
                "speed_limit_km": speed,
                "is_valid_school_zone": bool(row["is_valid_school_zone"]),
                "is_30_zone_excluded": bool(row.get("is_30_zone_excluded", False)),
                "classification": row.get("classification", "unknown"),
                "time_window": row.get("time_window"),
                "day_modifier": row.get("day_modifier"),
                "zone_extent_m": float(row.get("zone_extent_m", 0)),
                "zone_method": row.get("zone_method", ""),
                "nearest_road": row.get("nearest_road_name", ""),
                "ndetections": int(row["ndetections"]),
                "pole_bearing": float(row["pole_bearing"]),
                "validation_source": row.get("validation_source", ""),
                "sensor_speed_group": row.get("speedlimit_gfrgroup", ""),
                "sensor_school_group": row.get("schoolzone_gfrgroup", ""),
                "sensor_supplemental": row.get("supplemental_gfrgroup", ""),
            }
        }
        
        # Add zone extent line if valid
        if row.get("is_valid_school_zone") and row.get("zone_end_lat"):
            zone_line = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [row["pole_longitude"], row["pole_latitude"]],
                        [row["zone_end_lon"], row["zone_end_lat"]]
                    ]
                },
                "properties": {
                    "type": "zone_extent",
                    "speed_limit_km": speed,
                    "extent_m": float(row.get("zone_extent_m", 0)),
                    "method": row.get("zone_method", ""),
                    "classification": row.get("classification", "unknown"),
                }
            }
            features.append(zone_line)
        
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "pipeline": "HERE Berlin Hackathon 2026 - School Zone Pipeline",
            "generated_at": datetime.now().isoformat(),
            "total_detections": len(df),
            "valid_school_zones": int(df["is_valid_school_zone"].sum()),
            "api_used": VISION_API,
        }
    }
    
    output_path = OUTPUT_DIR / "school_zones_output.geojson"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)
    
    log.info(f"  GeoJSON saved: {output_path}")
    return str(output_path)


def generate_folium_map(df: pd.DataFrame, data: dict) -> str:
    """Generate an interactive Folium map with all results."""
    log.info("Generating Folium map...")
    
    # Center on Berlin
    center_lat = df["pole_latitude"].mean()
    center_lon = df["pole_longitude"].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="CartoDB positron"
    )
    
    # Add school locations as blue markers
    schools = data["schools"]
    school_group = folium.FeatureGroup(name="🏫 Schools", show=True)
    for _, school in schools.iterrows():
        folium.CircleMarker(
            location=[school["latitude"], school["longitude"]],
            radius=4,
            color="#3388ff",
            fill=True,
            fill_color="#3388ff",
            fill_opacity=0.4,
            popup=f"School: {school.get('suppliers', 'Unknown')}",
        ).add_to(school_group)
    school_group.add_to(m)
    
    # Valid school zones
    valid_permanent = folium.FeatureGroup(name="✅ Permanent School Zones", show=True)
    valid_conditional = folium.FeatureGroup(name="⏰ Conditional School Zones", show=True)
    invalid_group = folium.FeatureGroup(name="❌ Invalid / Unconfirmed", show=True)
    excluded_group = folium.FeatureGroup(name="🚫 Excluded (30 Zone)", show=False)
    
    for _, row in df.iterrows():
        lat, lon = row["pole_latitude"], row["pole_longitude"]
        speed = row.get("final_speed_limit_km", row.get("speed_limit_km", "?"))
        if isinstance(speed, pd.Series):
            speed = speed.iloc[0]
        
        popup_html = f"""
        <div style='width:300px; font-family:Arial; font-size:12px;'>
            <b>Speed Limit:</b> {speed} km/h<br>
            <b>Classification:</b> {row.get('classification', 'unknown')}<br>
            <b>Valid:</b> {row.get('is_valid_school_zone', False)}<br>
            <b>Time Window:</b> {row.get('time_window', 'N/A')}<br>
            <b>Zone Extent:</b> {row.get('zone_extent_m', 0):.0f}m ({row.get('zone_method', '')})<br>
            <b>Road:</b> {row.get('nearest_road_name', 'Unknown')}<br>
            <b>Detections:</b> {row['ndetections']}<br>
            <b>Bearing:</b> {row['pole_bearing']}°<br>
            <b>Sensor Data:</b> {row.get('speedlimit_gfrgroup', '')}
            <br><b>Validation:</b> {row.get('validation_source', '')}
        </div>
        """
        
        if row.get("is_30_zone_excluded"):
            folium.CircleMarker(
                location=[lat, lon], radius=6,
                color="#999999", fill=True, fill_color="#999999",
                fill_opacity=0.6, popup=folium.Popup(popup_html, max_width=350),
            ).add_to(excluded_group)
            
        elif row.get("is_valid_school_zone"):
            is_cond = row.get("classification") == "conditional"
            color = "#ff8800" if is_cond else "#00aa00"
            icon_color = "orange" if is_cond else "green"
            group = valid_conditional if is_cond else valid_permanent
            
            # Sign marker
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=350),
                icon=folium.Icon(color=icon_color, icon="info-sign"),
            ).add_to(group)
            
            # Zone extent line
            if row.get("zone_end_lat") and row.get("zone_end_lon"):
                folium.PolyLine(
                    [[lat, lon], [row["zone_end_lat"], row["zone_end_lon"]]],
                    color=color, weight=4, opacity=0.7,
                    popup=f"{speed} km/h zone: {row.get('zone_extent_m', 0):.0f}m",
                ).add_to(group)
        else:
            folium.CircleMarker(
                location=[lat, lon], radius=5,
                color="#cc0000", fill=True, fill_color="#cc0000",
                fill_opacity=0.5, popup=folium.Popup(popup_html, max_width=350),
            ).add_to(invalid_group)
    
    valid_permanent.add_to(m)
    valid_conditional.add_to(m)
    invalid_group.add_to(m)
    excluded_group.add_to(m)
    
    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 15px; border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3); font-family: Arial; font-size: 13px;">
        <h4 style="margin:0 0 10px 0;">🚸 School Zone Legend</h4>
        <p style="margin:3px 0;"><span style="color:#00aa00;">●</span> Permanent School Zone</p>
        <p style="margin:3px 0;"><span style="color:#ff8800;">●</span> Conditional (time-limited)</p>
        <p style="margin:3px 0;"><span style="color:#cc0000;">●</span> Invalid / Unconfirmed</p>
        <p style="margin:3px 0;"><span style="color:#999999;">●</span> Excluded (30 Zone)</p>
        <p style="margin:3px 0;"><span style="color:#3388ff;">●</span> School Location</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Title
    title_html = """
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index: 1000;
                background: white; padding: 10px 20px; border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3); font-family: Arial;">
        <h3 style="margin:0;">🚸 Berlin School Zone Speed Limits — HERE Hackathon 2026</h3>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    output_path = OUTPUT_DIR / "school_zones_map.html"
    m.save(str(output_path))
    log.info(f"  Map saved: {output_path}")
    return str(output_path)


def generate_summary_csv(df: pd.DataFrame) -> str:
    """Export a clean summary CSV."""
    summary_cols = [
        "pole_latitude", "pole_longitude", "pole_bearing", "ndetections",
        "speed_limit_km", "final_speed_limit_km", "is_valid_school_zone", "is_30_zone_excluded",
        "classification", "time_window", "day_modifier",
        "zone_extent_m", "zone_method", "nearest_road_name",
        "validation_source", "speedlimit_gfrgroup", "schoolzone_gfrgroup",
        "supplemental_gfrgroup",
    ]
    
    available_cols = [c for c in summary_cols if c in df.columns]
    summary = df[available_cols].copy()
    
    output_path = OUTPUT_DIR / "results_summary.csv"
    summary.to_csv(output_path, index=False)
    log.info(f"  Summary CSV saved: {output_path}")
    return str(output_path)


# ─── Main Pipeline ─────────────────────────────────────────────────────────

def run_pipeline():
    """Execute the full 3-stage pipeline."""
    start_time = time.time()
    
    log.info("=" * 60)
    log.info("SCHOOL ZONE DETECTION PIPELINE")
    log.info(f"  Vision API: {VISION_API}")
    log.info(f"  API Key set: {'Yes' if (ANTHROPIC_API_KEY or GEMINI_API_KEY) else 'No (sensor-only mode)'}")
    log.info(f"  Data dir: {BASE_DIR}")
    log.info(f"  Output dir: {OUTPUT_DIR}")
    log.info("=" * 60)
    
    # Load data
    data = load_data()
    
    # Stage 1: Validate
    df = stage1_validate(data)
    
    # Stage 2: Classify
    df = stage2_classify(df)
    
    # Stage 3: Zone extent
    df = stage3_zone_extent(df, data)
    
    # Generate outputs
    geojson_path = generate_geojson(df)
    csv_path = generate_summary_csv(df)
    map_path = generate_folium_map(df, data)
    
    # Final summary
    elapsed = time.time() - start_time
    valid = df["is_valid_school_zone"].sum()
    
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info(f"  Total time: {elapsed:.1f}s")
    log.info(f"  Total detections: {len(df)}")
    log.info(f"  Valid school zones: {valid}")
    log.info(f"  Permanent: {(df['classification'] == 'permanent').sum()}")
    log.info(f"  Conditional: {(df['classification'] == 'conditional').sum()}")
    log.info(f"  Excluded (30 Zone): {df['is_30_zone_excluded'].sum()}")
    log.info(f"  Outputs:")
    log.info(f"    GeoJSON: {geojson_path}")
    log.info(f"    CSV:     {csv_path}")
    log.info(f"    Map:     {map_path}")
    log.info("=" * 60)
    
    return df


if __name__ == "__main__":
    # Usage:
    #   python school_zone_pipeline.py                          # sensor-only mode
    #   GEMINI_API_KEY=xxx python school_zone_pipeline.py       # with Gemini
    #   ANTHROPIC_API_KEY=xxx VISION_API=anthropic python ...   # with Claude
    
    # Check for CLI args
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print(__doc__)
            print("\nEnvironment variables:")
            print("  VISION_API=gemini|anthropic    (default: gemini)")
            print("  GEMINI_API_KEY=xxx             Google Gemini API key")
            print("  ANTHROPIC_API_KEY=xxx          Anthropic Claude API key")
            sys.exit(0)
    
    run_pipeline()
