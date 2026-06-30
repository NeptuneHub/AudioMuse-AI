#!/usr/bin/env python3
"""Fail if emoji / decorative icon glyphs appear in Python source.

Emoji in shipped code break the Windows console (cp1252) when logged; use plain
ASCII instead (OK / X / ->). Detection uses the maintained `emoji` library for
true emoji, plus a short codepoint list for decorative text-symbols it does not
classify as emoji (arrows, check/ballot marks, bullet). query/ and
lyrics/lyrics_transcriber.py (which uses symbol-range bounds to strip emoji from
lyrics) are excluded; the rating-intent regex star in app_chat.py is allowed.
"""

import subprocess
import sys

import emoji

# Decorative text-symbols the emoji library does not flag, by codepoint so this
# file itself stays pure ASCII (and never flags itself).
EXTRA = {chr(c) for c in (0x2192, 0x2194, 0x2713, 0x2717, 0x2022)}
# Symbols used functionally as code, not decoration.
ALLOW = {chr(0x2B50)}  # star: rating-intent regex in app_chat.py
SKIP_FILES = {"lyrics/lyrics_transcriber.py"}  # builds emoji-stripping regex ranges


def _is_banned(ch):
    return (emoji.is_emoji(ch) or ch in EXTRA) and ch not in ALLOW


def main():
    files = subprocess.check_output(["git", "ls-files", "*.py"]).decode().split()
    files = [f for f in files if not f.startswith("query/") and f not in SKIP_FILES]
    bad = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for num, line in enumerate(fh, 1):
                hits = sorted({c for c in line if _is_banned(c)})
                if hits:
                    cps = " ".join("U+%04X" % ord(c) for c in hits)
                    bad.append("%s:%d: %s" % (path, num, cps))
    if bad:
        print("Emoji/icons found in Python source (use ASCII: OK / X / ->):")
        for entry in bad:
            print("  " + entry)
        print("See CONTRIBUTING.md (PR Requirements).")
        return 1
    print("OK: no emoji in Python source (checked with the emoji library).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
