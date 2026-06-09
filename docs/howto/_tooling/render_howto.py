"""Render the version-stamped how-to page from the Markdown prose template.

Substitutes the {{VERSION}} placeholder in howto.template.md and writes
docs/howto/<version>/howto.md. Pure stdlib; always writes LF newlines so the
output stays CRLF-free regardless of the OS it runs on.
"""
import argparse
import os

from _version import REPO_ROOT, resolve

TOOLING_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(TOOLING_DIR, "howto.template.md")


def render(version=None, repo_root=REPO_ROOT, template=TEMPLATE):
    display, folder = resolve(version, repo_root)
    with open(template, "r", encoding="utf-8") as fh:
        md = fh.read()
    md = md.replace("{{VERSION}}", display)
    out_dir = os.path.join(str(repo_root), "docs", "howto", folder)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "howto.md")
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(md)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Render the version-stamped how-to Markdown.")
    ap.add_argument("--version", default=None,
                    help="Version/tag (e.g. v2.2.0). Defaults to APP_VERSION in config.py.")
    args = ap.parse_args()
    print("Rendered", render(args.version))


if __name__ == "__main__":
    main()
