"""Live FAA and NOAA lookups, plus deterministic Part 107 checks on their results.

Rule checks are plain code, not model output, so limits always come out the same.
"""

from __future__ import annotations

import re

import httpx

METAR_URL = "https://aviationweather.gov/api/data/metar"
AIRSPACE_URL = (
    "https://services6.arcgis.com/ssFJjBXIUyZDrSYZ/arcgis/rest/services/"
    "Class_Airspace/FeatureServer/0/query"
)

MIN_VISIBILITY_SM = 3.0  # 14 CFR 107.51(c)
MIN_BELOW_CLOUD_FT = 500  # 14 CFR 107.51(d)(1)
MAX_ALTITUDE_AGL_FT = 400  # 14 CFR 107.51(b)
AUTHORIZATION_CLASSES = {"B", "C", "D", "E"}  # 14 CFR 107.41, when the layer reaches the surface

AIRSPACE_CAVEAT = (
    "Screening only. Confirm with LAANC or B4UFLY, and check TFRs and NOTAMs before any flight."
)


class LookupError_(RuntimeError):
    """A public data source failed or returned nothing usable."""


def parse_visibility(value: object) -> float | None:
    """Convert METAR visibility such as '10+', '1 1/2' or 6 into statute miles."""
    if value is None:
        return None
    text = str(value).strip().rstrip("+")
    match = re.fullmatch(r"(?:(\d+)\s+)?(\d+)/(\d+)", text)
    if match:
        whole = int(match.group(1) or 0)
        return whole + int(match.group(2)) / int(match.group(3))
    try:
        return float(text)
    except ValueError:
        return None


def evaluate_weather(metar: dict, planned_altitude_ft_agl: int) -> dict:
    """Check one METAR against the Part 107 visibility, cloud and altitude limits."""
    findings: list[dict] = []

    if planned_altitude_ft_agl > MAX_ALTITUDE_AGL_FT:
        findings.append(
            {
                "rule": "14 CFR 107.51(b)",
                "status": "fail",
                "detail": f"{planned_altitude_ft_agl} ft AGL is above the {MAX_ALTITUDE_AGL_FT} ft "
                "limit, unless flown within 400 ft of a structure.",
            }
        )

    visibility = parse_visibility(metar.get("visib"))
    if visibility is None:
        findings.append(
            {"rule": "14 CFR 107.51(c)", "status": "unknown", "detail": "No visibility reported."}
        )
    else:
        passed = visibility >= MIN_VISIBILITY_SM
        findings.append(
            {
                "rule": "14 CFR 107.51(c)",
                "status": "pass" if passed else "fail",
                "detail": f"Reported visibility {metar.get('visib')} SM; minimum is "
                f"{MIN_VISIBILITY_SM:g} SM.",
            }
        )

    bases = [layer["base"] for layer in metar.get("clouds") or [] if layer.get("base") is not None]
    if not bases:
        findings.append(
            {
                "rule": "14 CFR 107.51(d)(1)",
                "status": "pass",
                "detail": "No cloud layers with a reported base.",
            }
        )
    else:
        lowest = min(bases)
        clearance = lowest - planned_altitude_ft_agl
        findings.append(
            {
                "rule": "14 CFR 107.51(d)(1)",
                "status": "pass" if clearance >= MIN_BELOW_CLOUD_FT else "fail",
                "detail": f"Lowest cloud base {lowest} ft gives {clearance} ft of clearance; "
                f"minimum is {MIN_BELOW_CLOUD_FT} ft below clouds.",
            }
        )

    statuses = {finding["status"] for finding in findings}
    overall = "fail" if "fail" in statuses else "unknown" if "unknown" in statuses else "pass"
    return {
        "overall": overall,
        "findings": findings,
        "caveats": [
            "A METAR describes the reporting station, which may be miles from the flight site.",
            "The 2,000 ft horizontal cloud clearance in 14 CFR 107.51(d)(2) cannot be checked "
            "from a METAR.",
        ],
    }


def get_metar(station: str, client: httpx.Client | None = None) -> dict:
    """Fetch the latest METAR for an ICAO station such as KPHX."""
    station = station.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{3,4}", station):
        raise LookupError_(f"'{station}' is not a valid ICAO station identifier.")
    http = client or httpx.Client(timeout=15)
    response = http.get(METAR_URL, params={"ids": station, "format": "json"})
    if response.status_code == 204 or not response.content:
        raise LookupError_(f"No current METAR for {station}.")
    response.raise_for_status()
    results = response.json()
    if not results:
        raise LookupError_(f"No current METAR for {station}.")
    metar = results[0]
    return {
        "station": metar.get("icaoId", station),
        "observed_at": metar.get("reportTime"),
        "raw": metar.get("rawOb"),
        "visib": metar.get("visib"),
        "wind_direction_deg": metar.get("wdir"),
        "wind_speed_kt": metar.get("wspd"),
        "wind_gust_kt": metar.get("wgst"),
        "temperature_c": metar.get("temp"),
        "clouds": metar.get("clouds") or [],
    }


def summarize_airspace(features: list[dict]) -> dict:
    """Decide whether any layer at the point reaches the surface in an authorization class."""
    layers = []
    for feature in features:
        attributes = feature.get("attributes", {})
        layers.append(
            {
                "class": attributes.get("CLASS"),
                "name": attributes.get("NAME"),
                "lower_ft": attributes.get("LOWER_VAL"),
                "upper_ft": attributes.get("UPPER_VAL"),
            }
        )
    surface = [
        layer
        for layer in layers
        if layer["class"] in AUTHORIZATION_CLASSES and layer["lower_ft"] == 0
    ]
    return {
        "authorization_required": bool(surface),
        "rule": "14 CFR 107.41",
        "surface_controlled_layers": surface,
        "all_layers": layers,
        "caveat": AIRSPACE_CAVEAT,
    }


def lookup_airspace(latitude: float, longitude: float, client: httpx.Client | None = None) -> dict:
    """Query the FAA Class Airspace dataset for the layers above a point."""
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise LookupError_("Latitude or longitude is out of range.")
    http = client or httpx.Client(timeout=20)
    response = http.get(
        AIRSPACE_URL,
        params={
            "where": "1=1",
            "geometry": f"{longitude},{latitude}",
            "geometryType": "esriGeometryPoint",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "CLASS,NAME,LOWER_VAL,UPPER_VAL",
            "returnGeometry": "false",
            "f": "json",
        },
    )
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise LookupError_(f"FAA airspace service error: {payload['error']}")
    return summarize_airspace(payload.get("features", []))
