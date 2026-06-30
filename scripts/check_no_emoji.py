#!/usr/bin/env python3

import subprocess
import sys

import emoji

EXTRA = {chr(c) for c in (0x2192, 0x2194, 0x2713, 0x2717, 0x2022)}
ALLOW = {chr(0x2B50)}
SKIP_FILES = {"lyrics/lyrics_transcriber.py"}


def _is_banned(ch):
    return (emoji.is_emoji(ch) or ch in EXTRA) and ch not in ALLOW


def main():
    out = subprocess.check_output(["git", "ls-files", "-z", "*.py"]).decode()
    files = [
        f
        for f in out.split("\0")
        if f
        and not f.startswith("query/")
        and f not in SKIP_FILES
        and not (f.startswith("native-build/") and "/_vendor/" in f)
    ]
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
