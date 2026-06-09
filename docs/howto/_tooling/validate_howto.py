"""Validate a rendered how-to folder. Pure stdlib — safe to run in CI.

Checks, for docs/howto/<version>/howto.md:
  1. howto.md exists and is stamped with the expected version (e.g. 'v2.2.0');
  2. every screenshots/* image it references exists and is non-empty;
  3. every internal #anchor link resolves to a heading (GitHub slug rules).

Exits non-zero (with a clear message) on any failure so a release workflow can
use it as a gate. Warns about screenshots present but never referenced.
"""
import argparse
import os
import re
import sys

from _version import REPO_ROOT, resolve

_PUNCT = re.compile(r"[^\w\s-]", re.UNICODE)


def slugify(heading, seen):
    """Approximate GitHub's heading-anchor algorithm (github-slugger)."""
    s = heading.strip().lower()
    s = _PUNCT.sub("", s)          # drop punctuation/symbols (keep word chars, space, hyphen)
    s = s.replace(" ", "-")
    base = s
    n = seen.get(base, 0)
    seen[base] = n + 1
    return base if n == 0 else "%s-%d" % (base, n)


def validate(version=None, repo_root=REPO_ROOT):
    display, folder = resolve(version, repo_root)
    base = os.path.join(str(repo_root), "docs", "howto", folder)
    md_path = os.path.join(base, "howto.md")
    errors, warnings = [], []

    if not os.path.isfile(md_path):
        return [f"Missing {md_path}. Run make_howto.py / render_howto.py for {display}."], []

    with open(md_path, "r", encoding="utf-8") as fh:
        md = fh.read()

    # 1. version stamp
    if display not in md:
        errors.append(f"howto.md is not stamped with {display} (stale version?).")

    # 2. referenced images exist and are non-empty
    srcs = re.findall(r"!\[[^\]]*\]\((screenshots/[^)]+)\)", md)
    if not srcs:
        errors.append("No screenshots are referenced by howto.md.")
    for src in srcs:
        p = os.path.join(base, src)
        if not os.path.isfile(p):
            errors.append(f"Referenced screenshot missing: {src} "
                          f"(capture and commit docs/howto/{folder}/screenshots/).")
        elif os.path.getsize(p) == 0:
            errors.append(f"Referenced screenshot is empty: {src}")

    # 3. internal anchors resolve to heading slugs
    seen = {}
    slugs = set()
    for line in md.splitlines():
        m = re.match(r"^#{1,6}\s+(.*?)\s*#*$", line)
        if m:
            slugs.add(slugify(m.group(1), seen))
    for href in re.findall(r"\]\(#([^)]+)\)", md):
        if href not in slugs:
            errors.append(f"Broken in-page link: #{href} has no matching heading.")

    # warn: screenshots on disk but never referenced
    shots_dir = os.path.join(base, "screenshots")
    if os.path.isdir(shots_dir):
        referenced = {os.path.basename(s) for s in srcs}
        for f in sorted(os.listdir(shots_dir)):
            if f.lower().endswith(".png") and f not in referenced:
                warnings.append(f"Screenshot present but not referenced: screenshots/{f}")

    return errors, warnings


def main():
    ap = argparse.ArgumentParser(description="Validate a rendered how-to folder.")
    ap.add_argument("--version", default=None,
                    help="Version/tag (e.g. v2.2.0). Defaults to APP_VERSION in config.py.")
    args = ap.parse_args()
    errors, warnings = validate(args.version)
    for w in warnings:
        print("WARNING:", w)
    if errors:
        for e in errors:
            print("ERROR:", e)
        print(f"\nValidation FAILED with {len(errors)} error(s).")
        sys.exit(1)
    print("Validation passed.")


if __name__ == "__main__":
    main()
