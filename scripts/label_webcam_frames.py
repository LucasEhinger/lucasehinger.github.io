#!/usr/bin/env python3
"""Date-label Mount Washington webcam timelapse frames by OCRing the burned-in clock.

Frames extracted from the MWOBS timelapse videos are named frame_0000.jpg with no
date. Each image has the timestamp drawn into the bottom strip:

    tower        2024-05-25  12:00:00   Mount Washington Observatory   (1920x1080)
    observatory  Mount Washington Observatory Sat Jul  6 12:00:00 2024 (1280x720)

BOTH cameras are daily-noon images; the video holds each day for 1-6 frames, so
1700 tower frames cover ~589 days. Output is therefore ONE image per date.

OCR here is genuinely hard: the caption sits over sunlit lichen-covered rock in
summer and over featureless rime in winter, so white-on-white kills naive
thresholding. Three defences:

  1. Contrast-enhance the caption strip (upscale 4x, then CLAHE / min-max
     stretch / morphological top-hat) before any OCR sees it.
  2. Vote across engines and variants: macOS Vision (via the `visocr` helper,
     far better on low contrast) plus tesseract on a low-saturation mask -- the
     overlay text is pure white while rock is coloured.
  3. Repair against the sequence. Frames are chronological and step by 0 or 1
     day, so we keep the longest non-decreasing subsequence as anchors and fill
     gaps by rounding to WHOLE DAYS. An earlier version interpolated fractional
     timestamps and invented times like 15:47 for a frame stamped 12:00:01 --
     never interpolate a quantity the source does not vary.

Confidence is recorded per date so low-trust dates can be re-checked by eye
rather than silently trusted.

Usage:
    python3 scripts/label_webcam_frames.py --src <mp4_jpgs dir> --out files/weather/webcam
    python3 scripts/label_webcam_frames.py --src ... --limit 60 --dry-run
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import cv2
import numpy as np

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]
CAMS = ("tower", "observatory")
CROP = {"tower": (0.00, 0.945, 0.40), "observatory": (0.18, 0.945, 0.80)}
VISOCR = os.environ.get("VISOCR", "visocr")       # path to the Swift Vision helper
RX_ISO = re.compile(r'(20\d{2})\s*-\s*(\d{1,2})\s*-\s*(\d{1,2})')
RX_NAMED = re.compile(r'([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2})\b.*?\b(20\d{2})\b')
RX_NAMED2 = re.compile(r'\b(20\d{2})\b.*?([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2})\b')


def enhance(img, cam, outdir, stem):
    """Write contrast-enhanced variants of the caption strip; return their paths."""
    H, W = img.shape[:2]
    x0, y0, x1 = CROP[cam]
    strip = img[int(H * y0):, int(W * x0):int(W * x1)]
    g = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(g, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    tophat = cv2.morphologyEx(
        up, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60)))
    variants = {
        "stretch": cv2.normalize(up, None, 0, 255, cv2.NORM_MINMAX),
        "tophat": cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX),
        "clahe": cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(up),
    }
    paths = []
    for k, v in variants.items():
        p = os.path.join(outdir, f"{stem}__{k}.png")
        cv2.imwrite(p, v)
        paths.append(p)
    # low-saturation mask: overlay text is pure white, rock is coloured
    s = cv2.resize(strip, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC).astype(int)
    mx, mn = s.max(2), s.min(2)
    mask = np.where((mx >= 225) & ((mx - mn) <= 28), 255, 0).astype("uint8")
    p = os.path.join(outdir, f"{stem}__white.png")
    cv2.imwrite(p, mask)
    paths.append(p)
    return paths


def parse_date(text):
    m = RX_ISO.search(text)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
        except ValueError:
            pass
    # order = which capture group holds (month, day, year) for each pattern
    for rx, order in ((RX_NAMED, (1, 2, 3)), (RX_NAMED2, (2, 3, 1))):
        m = rx.search(text)
        if not m:
            continue
        mon_tok, day_tok, yr_tok = (m.group(i) for i in order)
        mon = next((i + 1 for i, mm in enumerate(MONTHS)
                    if mon_tok.lower()[:3] == mm), None)
        if not mon:
            continue
        try:
            return datetime(int(yr_tok), mon, int(day_tok)).date()
        except ValueError:
            continue
    return None


def run_vision(paths):
    """Batch the Swift Vision helper; returns {path: text}."""
    if not paths:
        return {}
    out = {}
    for i in range(0, len(paths), 150):
        chunk = paths[i:i + 150]
        try:
            r = subprocess.run([VISOCR, "--full"] + chunk,
                               capture_output=True, timeout=600)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
        for line in r.stdout.decode("utf-8", "replace").splitlines():
            if "\t" in line:
                p, t = line.split("\t", 1)
                out[p] = t
    return out


def run_tesseract(path, cam):
    cmd = ["tesseract", path, "stdout", "--psm", "7"]
    if cam == "tower":
        cmd += ["-c", "tessedit_char_whitelist=0123456789-"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        return ""
    return r.stdout.decode("utf-8", "replace").strip().replace("\n", " ")


def repair_dates(seq):
    """Non-decreasing dates; unread frames filled to whole days. -> (dates, filled)."""
    n = len(seq)
    idx = [i for i, v in enumerate(seq) if v is not None]
    if not idx:
        return seq, [True] * n
    best = [1] * len(idx)
    prev = [-1] * len(idx)
    for a in range(len(idx)):
        for b in range(a):
            if seq[idx[b]] <= seq[idx[a]] and best[b] + 1 > best[a]:
                best[a], prev[a] = best[b] + 1, b
    cur = max(range(len(idx)), key=lambda i: best[i])
    keep = []
    while cur != -1:
        keep.append(idx[cur])
        cur = prev[cur]
    keep.reverse()
    anchors = set(keep)
    out, filled = list(seq), [i not in anchors for i in range(n)]
    for i in range(n):
        if i in anchors:
            continue
        lo = max((k for k in keep if k < i), default=None)
        hi = min((k for k in keep if k > i), default=None)
        if lo is not None and hi is not None:
            span = (seq[hi] - seq[lo]).days
            out[i] = seq[lo] + timedelta(days=round(span * (i - lo) / (hi - lo)))
        elif lo is not None:
            out[i] = seq[lo]
        else:
            out[i] = seq[hi]
    return out, filled


def process(cam, srcdir, outroot, limit, dry):
    files = sorted(f for f in os.listdir(srcdir) if f.lower().endswith(".jpg"))
    if limit:
        files = files[:limit]
    print(f"{cam}: {len(files)} frames")
    tmp = tempfile.mkdtemp(prefix=f"wc_{cam}_")
    allpaths, owner = [], {}
    for i, f in enumerate(files):
        img = cv2.imread(os.path.join(srcdir, f))
        if img is None:
            continue
        for p in enhance(img, cam, tmp, f"{i:05d}"):
            allpaths.append(p)
            owner[p] = i
        if (i + 1) % 400 == 0:
            print(f"  prepared {i+1}/{len(files)}")
    print(f"  running Vision on {len(allpaths)} crops...")
    vis = run_vision(allpaths)

    cands = defaultdict(list)
    for p, t in vis.items():
        d = parse_date(t)
        if d:
            cands[owner[p]].append(d)
    # tesseract only where Vision gave nothing -- it is slower and weaker here
    missing = [i for i in range(len(files)) if not cands.get(i)]
    print(f"  Vision read {len(files)-len(missing)}/{len(files)}; "
          f"tesseract fallback on {len(missing)}")
    for i in missing:
        for p in [q for q in allpaths if owner[q] == i]:
            d = parse_date(run_tesseract(p, cam))
            if d:
                cands[i].append(d)

    seq, votes = [], []
    for i in range(len(files)):
        c = cands.get(i) or []
        if c:
            d, n = Counter(c).most_common(1)[0]
            seq.append(d)
            votes.append(n)
        else:
            seq.append(None)
            votes.append(0)
    read = sum(1 for v in seq if v)
    fixed, filled = repair_dates(seq)
    print(f"  OCR ok {read}/{len(seq)} ({100*read/len(seq):.1f}%), "
          f"{sum(filled)} filled from neighbours")
    steps = Counter((fixed[i+1] - fixed[i]).days for i in range(len(fixed)-1)
                    if fixed[i] and fixed[i+1])
    print(f"  day-steps between frames: {sorted(steps.items())[:6]}")

    # one output image per date: prefer the frame with the strongest vote
    #
    # NOTE (2026-09-28) -- this choice was CORRECTED BY HAND AFTERWARDS, and
    # re-running this script will undo that correction.
    #
    # The MWOBS timelapse crossfades continuously from one day to the next, so
    # most frames are a 50/50-ish mix of two days. "Strongest OCR vote" often
    # lands on one of those: the caption stays legible mid-fade while the scene
    # behind it is two days superimposed. A scan found ~half of all date images
    # were blends (tower 291/547, observatory 255/514).
    #
    # They were re-selected with a one-off script (not committed) that, for each
    # blended date:
    #   1. found the "clean" keyframes within +/-4 frames -- a frame that is NOT
    #      well explained as a*prev + (1-a)*next (mean |residual| >= 5 on a
    #      320x180 greyscale copy; mid-fade frames sit near 1);
    #   2. OCR'd each candidate's caption with BOTH tesseract and macOS Vision
    #      (a small Swift `visocr` helper, rebuilt for this -- see VISOCR above),
    #      and kept only candidates whose own caption reads the target date.
    #      Picking by visual similarity instead chose the NEIGHBOURING day on 50
    #      of 414 dates (e.g. 2025-06-29 -> a 06-30 frame), which is why the
    #      timestamp has to confirm it;
    #   3. swapped to the confirmed frame with the most OCR votes.
    # Result: 211 tower + 156 observatory dates swapped; 80 + 99 kept their
    # blended image because no clean frame could be confirmed, and are flagged.
    #
    # Both are recorded in the manifests, in two columns this script does not
    # write: `reselected_from` (the frame the date used to point at) and
    # `blended` (1 = still a crossfade; the galleries mark these with an
    # asterisk). ocr_votes/confidence still describe the ORIGINAL pick. The
    # replaced images and the untouched manifests are backed up under
    # files/weather/webcam/_replaced_blended_frames/.
    #
    # Hand labels were NOT changed: they were scored from the best image viewed
    # by hand, not from these files.
    #
    # A second pass the same day checked every date image's caption against its
    # date. Every date this script READ was right; the misdated ones were all
    # dates it INFERRED from neighbours (`filled` = 1): 26 of 34 tower and 18 of
    # 46 observatory captions named a different day, typically 2-10 days off
    # (worst -74). Every timelapse frame was then OCR'd, and for each such date:
    #   * a frame whose caption reads that date, lying between the frames of the
    #     nearest confirmed dates either side, was swapped in (22, each caption
    #     checked by eye; blended ones flagged as above);
    #   * with no such frame, the image was removed, so the galleries show no
    #     frame that day (22) -- the original is in the backup folder;
    #   * the 3 tower dates whose caption could not be read and had no
    #     confirming frame were kept.
    #   Then, where a removed image's caption named a day with no image of
    #   its own, a clean frame of that day was filed under it (tower
    #   2025-04-02, 2025-07-20, 2025-09-06; `date_check` = refiled). The
    #   2025-08-08 candidate's caption is a crossfade of 07 and 08, so it
    #   was not used.
    # Recorded in two more columns: `date_check` (swapped / removed /
    # unconfirmed) and `caption_read` (what the old image's caption said). The
    # manifests from before this pass are *_manifest.before_date_fix.csv in the
    # backup folder. Filling gaps by neighbour inference is the weak step here.
    bydate = defaultdict(list)
    for i, d in enumerate(fixed):
        if d:
            bydate[d].append(i)
    outdir = os.path.join(outroot, cam)
    if not dry:
        os.makedirs(outdir, exist_ok=True)
    rows = []
    for d in sorted(bydate):
        members = bydate[d]
        pick = max(members, key=lambda i: (votes[i], -abs(i - members[len(members)//2])))
        name = f"{d:%Y-%m-%d}.jpg"
        if not dry:
            shutil.copy2(os.path.join(srcdir, files[pick]),
                         os.path.join(outdir, name))
        rows.append({"date": f"{d:%Y-%m-%d}", "output_name": name,
                     "source_frame": files[pick], "n_frames_for_date": len(members),
                     "ocr_votes": votes[pick], "filled": int(filled[pick]),
                     "confidence": "high" if votes[pick] >= 2 else
                                   ("low" if votes[pick] == 1 else "inferred")})
    shutil.rmtree(tmp, ignore_errors=True)
    if not rows:
        print(f"  ERROR: no dates parsed for {cam} -- refusing to write an empty "
              f"manifest. Check the crop region and date regexes.")
        return
    if dry:
        for r in rows[:12]:
            print(f"    {r['date']}  <- {r['source_frame']}  "
                  f"votes={r['ocr_votes']} conf={r['confidence']}")
        return
    man = os.path.join(outroot, f"{cam}_manifest.csv")
    os.makedirs(outroot, exist_ok=True)
    with open(man, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    conf = Counter(r["confidence"] for r in rows)
    span = (max(bydate) - min(bydate)).days + 1
    print(f"  wrote {len(rows)} dated images ({min(bydate)} .. {max(bydate)}, "
          f"{span - len(rows)} calendar days missing)")
    print(f"  confidence: {dict(conf)}")
    print(f"  manifest {man}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--out", default="files/weather/webcam")
    p.add_argument("--cams", nargs="+", default=list(CAMS), choices=CAMS)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    for cam in a.cams:
        d = os.path.join(a.src, cam)
        if os.path.isdir(d):
            process(cam, d, a.out, a.limit, a.dry_run)
        else:
            print(f"{cam}: missing {d}")


if __name__ == "__main__":
    main()
