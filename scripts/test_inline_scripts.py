#!/usr/bin/env python3
"""Every inline <script> must still parse after the production build squashes it.

The deployed site pipes each page through _layouts/compress.html, which splits
the whole page on whitespace and rejoins it with single spaces -- except
inside <pre>. Locally `jekyll serve` runs in development, where compression is
off, so a script that works on localhost can be dead on the live site. The
classic way to break one is a `//` comment: once the newline after it is gone,
it comments out everything that follows, and the page throws "Unexpected end
of input". Use /* */ comments in inline scripts, or move the code to a .js file.

This does what compress.html does to each inline script and asks node to parse
the result. Scripts carrying Liquid tags are skipped (they are not JavaScript
until Jekyll renders them). Needs node on PATH.

    python3 scripts/test_inline_scripts.py
"""
import glob
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = ("_pages/*.html", "_includes/**/*.html", "_layouts/*.html")
SCRIPT = re.compile(
    r"<script(?![^>]*\bsrc=)(?![^>]*application/(?:ld\+)?json)[^>]*>(.*?)</script>", re.S)


def html_comments_removed(html):
    # A script inside an HTML comment is not rendered, so it cannot break.
    return re.sub(r"<!--.*?-->", "", html, flags=re.S)


def main():
    files = sorted({p for pat in SOURCES for p in glob.glob(os.path.join(ROOT, pat), recursive=True)})
    checked = failed = 0
    for path in files:
        html = html_comments_removed(open(path, encoding="utf-8").read())
        for k, body in enumerate(SCRIPT.findall(html)):
            if not body.strip() or "{{" in body or "{%" in body:
                continue
            squashed = " ".join(body.split())   # what compress.html does
            with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
                fh.write(squashed)
                tmp = fh.name
            r = subprocess.run(["node", "--check", tmp], capture_output=True, text=True)
            os.unlink(tmp)
            checked += 1
            if r.returncode:
                failed += 1
                err = next((l for l in r.stderr.splitlines() if "Error" in l), r.stderr.strip()[:120])
                print(f"FAIL  {os.path.relpath(path, ROOT)} script #{k}: {err}")
                print(f"      starts: {squashed[:90]}")
    print(f"\n{checked} inline scripts checked, {failed} broken after compression")
    if failed:
        print("Fix: use /* */ instead of // inside inline scripts.")
        return 1
    print("PASS: every inline script survives the production build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
