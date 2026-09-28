#!/usr/bin/env python3
"""The cuts table on /weather/observations/ must state the cuts the code applies.

The table is hand-written HTML, and the thresholds live in build_undercast_record.py.
Nothing else connects them, so a re-tune that changes a constant would leave the
page describing a rule the record no longer uses -- and every "cuts:" pill beneath
it would then be read against the wrong numbers. Each cell carries data-cut naming
its constant; this reads the number out of the cell and compares.

    python3 scripts/test_observation_cuts.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_undercast_record as B  # noqa: E402

PAGE = "_pages/weather-observations.html"
NUMERIC = ("MIN_VIS_SM", "AMB_VIS_SM", "MIN_DEPTH_FT", "AMB_DEPTH_FT",
           "MIN_LOWEST_ABOVE_FT", "AMB_LOWEST_ABOVE_FT", "MIN_LID_FT", "AMB_LID_FT")


def main():
    html = open(PAGE, encoding="utf-8").read()
    cells = dict(re.findall(r'data-cut="([A-Z_]+)"[^>]*>(.*?)</td>', html, re.S))
    fails = []

    for name in NUMERIC:
        if name not in cells:
            fails.append(f"{name}: no cell on the page")
            continue
        m = re.search(r"(\d[\d,]*)", cells[name])
        shown = int(m.group(1).replace(",", "")) if m else None
        want = getattr(B, name)
        status = "ok" if shown == want else "MISMATCH"
        print(f"  {name:20s} page={shown!s:>6}  code={want:>6}  {status}")
        if shown != want:
            fails.append(f"{name}: page says {shown}, code uses {want}")

    covers = {"POS_COVER": B.POS_COVER, "AMB_COVER": B.AMB_COVER}
    for name, want in covers.items():
        shown = tuple(re.findall(r"\b(FEW|SCT|BKN|OVC)\b", cells.get(name, "")))
        status = "ok" if set(shown) == set(want) else "MISMATCH"
        print(f"  {name:20s} page={'/'.join(shown):>6}  code={'/'.join(want):>6}  {status}")
        if set(shown) != set(want):
            fails.append(f"{name}: page says {shown}, code uses {want}")

    if fails:
        print("\nFAIL: the cuts table disagrees with build_undercast_record.py:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("\nPASS: every cut on the observations page matches the code")


if __name__ == "__main__":
    main()
