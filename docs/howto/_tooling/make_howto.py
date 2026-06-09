"""One-shot local update: capture screenshots + render the how-to HTML.

Run this against your own analysed AudioMuse-AI instance when cutting a release,
then commit the resulting docs/howto/<version>/ folder.

    python make_howto.py --base-url http://192.168.3.204:8000 --user root --password root

Version defaults to APP_VERSION in config.py; pass --version to override
(e.g. when preparing docs ahead of a tag).
"""
import argparse
import os

from _version import REPO_ROOT, resolve
import howto_capture
import render_howto


def main():
    ap = argparse.ArgumentParser(description="Capture + render the how-to for a release.")
    ap.add_argument("--base-url", default=os.environ.get("HOWTO_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--user", default=os.environ.get("HOWTO_USER", "root"))
    ap.add_argument("--password", default=os.environ.get("HOWTO_PASSWORD", "root"))
    ap.add_argument("--version", default=None)
    ap.add_argument("--browser-channel", default="chrome")
    ap.add_argument("--mock-all", action="store_true",
                    help="Fulfill every data /api/** with placeholder JSON (empty CI instance).")
    ap.add_argument("--skip-capture", action="store_true",
                    help="Only re-render the Markdown from the template (no browser).")
    args = ap.parse_args()

    display, folder = resolve(args.version)
    out_dir = os.path.join(str(REPO_ROOT), "docs", "howto", folder, "screenshots")

    if not args.skip_capture:
        print("== Capturing screenshots for %s ==" % display)
        howto_capture.capture(args.base_url, args.user, args.password, out_dir,
                              channel=args.browser_channel, mock_all=args.mock_all)

    print("== Rendering howto.md for %s ==" % display)
    out_html = render_howto.render(args.version)
    print("Done. Review and commit docs/howto/%s/  (HTML: %s)" % (folder, out_html))


if __name__ == "__main__":
    main()
