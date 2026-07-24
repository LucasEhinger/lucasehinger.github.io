#!/usr/bin/env python3
"""
Organize files in files/weather/examples/ into per-date subfolders.
Moves matching JSONs and their corresponding _Obs/_Tower images into YYYY-MM-DD folders,
then rebuilds index.json with updated paths.
"""
import os
import re
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / 'files' / 'weather' / 'examples'
INDEX_FILE = EXAMPLES_DIR / 'index.json'

PAT = re.compile(r'weather_data_[^_]*_(\d{4}-\d{1,2}-\d{1,2})\.json$', re.IGNORECASE)

def pad_date(date_str):
    y, m, d = date_str.split('-')
    return f"{y}-{int(m):02d}-{int(d):02d}"

def main():
    if not EXAMPLES_DIR.exists():
        print('Examples dir not found:', EXAMPLES_DIR)
        return

    json_files = sorted([p for p in EXAMPLES_DIR.iterdir() if p.is_file() and p.name.lower().endswith('.json')])
    entries = []

    for jf in json_files:
        m = PAT.search(jf.name)
        if not m:
            # try match even if filename uses -2 (unpadded) pattern
            mm = re.search(r'(\d{4}-\d{1,2}-\d{1,2})', jf.name)
            if mm:
                date_raw = mm.group(1)
            else:
                print('Skipping non-matching json:', jf.name)
                continue
        else:
            date_raw = m.group(1)

        date_padded = pad_date(date_raw)
        folder = EXAMPLES_DIR / date_padded
        folder.mkdir(exist_ok=True)

        # move json into folder
        dest_json = folder / jf.name
        if jf.resolve() != dest_json.resolve():
            print('Moving', jf.name, '->', dest_json)
            jf.rename(dest_json)
        else:
            print('Already in place:', dest_json)

        # move images if present
        for suf in ['_Obs.jpg', '_Tower.jpg']:
            img_name = f"{date_padded}{suf}"
            # possible image files may live at root either padded or unpadded
            src1 = EXAMPLES_DIR / img_name
            src2 = EXAMPLES_DIR / f"{date_raw}{suf}"
            dest_img = folder / img_name
            if src1.exists():
                print('Moving image', src1.name, '->', dest_img)
                src1.rename(dest_img)
            elif src2.exists():
                # if unpadded exists, move and rename to padded name
                print('Moving image', src2.name, '->', dest_img)
                src2.rename(dest_img)
            else:
                # nothing found; skip
                pass

        entries.append({'date': date_padded, 'json': str(dest_json.relative_to(ROOT).as_posix())})

    # rebuild index.json sorted by date descending
    entries.sort(key=lambda e: e['date'], reverse=True)
    index_list = []
    for e in entries:
        filename = Path(e['json']).name
        label = f"Mt Washington — {e['date']}"
        path = '/' + e['json']
        index_list.append({'filename': filename, 'label': label, 'path': path})

    # write index.json
    with INDEX_FILE.open('w') as f:
        json.dump(index_list, f, indent=2)
    print('Wrote index.json with', len(index_list), 'entries')

if __name__ == '__main__':
    main()
