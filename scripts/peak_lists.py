"""Curated peak lists, matched against tracks by position.

Matching is by coordinate, not name. OSM spells these inconsistently (East
Osceola is just "East Peak", Owl's Head is "Owls Head"), and coordinates also
separate peaks that share a name — there is a Mount Adams in both New Hampshire
and Washington, and these two lists would otherwise claim each other's summits.

The NH48 coordinates below are OSM's own summit nodes, so they line up exactly
with the peaks found in tracks. The name in each entry is the canonical one
shown on the page, which is not always what OSM calls it.

These are also matched against tracks directly, not only via OSM, so a summit
OSM happens to be missing still counts.
"""

# Tight enough that a false summit doesn't count as the real one — Pikers Peak
# sits about 1 km below the true summit of Mount Adams (WA).
MATCH_RADIUS_M = 500

# The AMC's 48 four-thousand-footers of New Hampshire.
NH48 = [
    ("Mount Washington", 44.27049, -71.30330),
    ("Mount Adams", 44.32058, -71.29164),
    ("Mount Jefferson", 44.30423, -71.31675),
    ("Mount Monroe", 44.25506, -71.32150),
    ("Mount Madison", 44.32879, -71.27667),
    ("Mount Lafayette", 44.16071, -71.64438),
    ("Mount Lincoln", 44.14883, -71.64457),
    ("South Twin Mountain", 44.18756, -71.55480),
    ("Carter Dome", 44.26740, -71.17912),
    ("Mount Moosilauke", 44.02341, -71.83143),
    ("Mount Eisenhower", 44.24069, -71.35040),
    ("North Twin Mountain", 44.20255, -71.55799),
    ("Mount Carrigain", 44.09362, -71.44681),
    ("Mount Bond", 44.15292, -71.53121),
    ("Middle Carter Mountain", 44.30308, -71.16771),
    ("West Bond", 44.15476, -71.54363),
    ("Mount Garfield", 44.18724, -71.61069),
    ("Mount Liberty", 44.11582, -71.64221),
    ("South Carter Mountain", 44.28987, -71.17642),
    ("Wildcat Mountain", 44.25920, -71.20161),
    ("Mount Hancock", 44.08345, -71.49320),
    ("South Kinsman Mountain", 44.12296, -71.73663),
    ("Mount Field", 44.19619, -71.43317),
    ("Mount Osceola", 44.00159, -71.53597),
    ("Mount Flume", 44.10882, -71.62790),
    ("Mount Pierce", 44.22685, -71.36571),
    ("South Hancock", 44.07327, -71.48694),
    ("North Kinsman Mountain", 44.13335, -71.73681),
    ("Mount Willey", 44.18344, -71.42097),
    ("Bondcliff", 44.14058, -71.54092),
    ("Mount Zealand", 44.18002, -71.52158),
    ("North Tripyramid", 43.97316, -71.44278),
    ("Mount Cabot", 44.50607, -71.41454),
    ("East Osceola", 44.00610, -71.52058),
    ("Middle Tripyramid", 43.96457, -71.44007),
    ("Cannon Mountain", 44.15669, -71.69862),
    ("Mount Hale", 44.22173, -71.51202),
    ("Mount Jackson", 44.20318, -71.37546),
    ("Mount Tom", 44.21054, -71.44623),
    ("Mount Moriah", 44.34053, -71.13172),
    ("Mount Passaconaway", 43.95477, -71.38070),
    ("Owl's Head", 44.14412, -71.60497),
    ("Galehead Mountain", 44.18502, -71.57343),
    ("Mount Whiteface", 43.93653, -71.40763),
    ("Mount Waumbek", 44.43263, -71.41751),
    ("Mount Isolation", 44.21483, -71.30926),
    ("Mount Tecumseh", 43.96670, -71.55650),
    ("Wildcat D", 44.24941, -71.22359),
]

# The five glaciated Cascade volcanoes of Washington. Only Rainier's summit is
# in OSM's peak data for the tiles fetched so far, so these coordinates come
# from the published summit positions, with elevations supplied as a fourth
# field since there's no OSM node to read them from.
WA_VOLCANOES = [
    ("Mount Rainier", 46.85287, -121.76041, 14411),
    ("Mount Adams", 46.20249, -121.49060, 12281),
    ("Mount Baker", 48.77675, -121.81431, 10781),
    ("Glacier Peak", 48.11220, -121.11365, 10541),
    ("Mount St. Helens", 46.19125, -122.19588, 8363),
]

LISTS = {
    "nh48": {"label": "NH48", "peaks": NH48},
    "wa_volcanoes": {"label": "WA volcanoes", "peaks": WA_VOLCANOES},
}


# Coarse regions for filtering, tested in order — first match wins, so the
# boxes don't need to be disjoint. These are deliberately rough: they only have
# to sort summits into recognisable buckets, not survey state lines.
REGIONS = [
    # key, label, (lat_min, lat_max, lon_min, lon_max)
    ("pnw", "Pacific NW", (42.0, 60.0, -125.0, -116.0)),
    ("california", "California", (32.0, 42.0, -125.0, -114.0)),
    ("southwest", "Southwest", (31.0, 38.0, -114.0, -102.0)),
    ("rockies", "Rockies", (38.0, 60.0, -116.0, -102.0)),
    ("midwest", "Midwest", (37.0, 60.0, -102.0, -80.0)),
    ("south", "South", (24.0, 37.0, -102.0, -75.0)),
    ("northeast", "Northeast", (37.0, 60.0, -80.0, -60.0)),
]

REGION_LABELS = dict((key, label) for key, label, _ in REGIONS)
REGION_LABELS["other"] = "Elsewhere"

# How each region reads in a sentence: "13 named peaks in the Pacific NW", but
# "6 named peaks in California". Carried in the data so the page doesn't have to
# guess which labels take an article.
REGION_PHRASES = {
    "pnw": "in the Pacific NW",
    "california": "in California",
    "southwest": "in the Southwest",
    "rockies": "in the Rockies",
    "midwest": "in the Midwest",
    "south": "in the South",
    "northeast": "in the Northeast",
    "other": "elsewhere",
}


def region_for(lat, lon):
    for key, _, (lat_min, lat_max, lon_min, lon_max) in REGIONS:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return key
    # Alaska, Hawaii, and anywhere outside North America land here.
    return "other"
