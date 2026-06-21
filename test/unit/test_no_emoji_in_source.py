"""Guard test: no emoji / pictographic icon codepoints in CODE files.

Emoji embedded in source (especially in log strings) crash the Windows
standalone build when written to the Windows console (cp1252). They are only
allowed in rendered HTML pages, never in Python/shell/CI/config code.

The candidate file list is built from ``git ls-files`` so it matches CI exactly.
We flag only the unambiguous pictographic emoji codepoints to avoid false
positives on legitimate accented Latin, CJK lyrics fixtures, typographic
quotes/dashes, or mathematical symbols.
"""
import os
import subprocess

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)

# Code file extensions to scan. HTML is intentionally absent (emoji allowed
# in rendered pages). JSON is absent (test fixtures carry intentional unicode).
CODE_EXTENSIONS = (
    '.py', '.js', '.sh', '.bat', '.ps1',
    '.yml', '.yaml', '.toml', '.cfg', '.ini',
)

# Files / trees excluded from the guard, with the reason:
#   - plotly-2.29.1.min.js: vendored minified library (treat as binary blob).
#   - lyrics_transcriber.py: contains a deliberate emoji-STRIPPING regex whose
#     character class is literally built from emoji range boundaries.
EXACT_EXCLUDES = {
    'static/plotly-2.29.1.min.js',
    'lyrics/lyrics_transcriber.py',
}

# Path fragments excluded outright (vendored / generated / non-console UI).
PATH_FRAGMENT_EXCLUDES = (
    '.venv',
    'node_modules',
    'screenshot/',
)


def _is_pictographic_emoji(codepoint):
    """True for unambiguous pictographic emoji / icon codepoints.

    Deliberately narrow: the astral Supplemental-Symbols/Emoji plane plus the
    emoji presentation selector. This is the surface that crashes the Windows
    console build and has no legitimate use in code. Accented Latin, CJK,
    typographic punctuation and math symbols are all left untouched.
    """
    return (0x1F000 <= codepoint <= 0x1FAFF) or codepoint == 0xFE0F


def _git_ls_files():
    out = subprocess.check_output(
        ['git', 'ls-files'], cwd=REPO_ROOT
    ).decode('utf-8')
    return [line for line in out.splitlines() if line]


def _is_candidate(rel_path):
    posix = rel_path.replace('\\', '/')
    if not posix.endswith(CODE_EXTENSIONS):
        return False
    if posix.endswith('.html') or posix.startswith('templates/'):
        return False
    if posix.endswith('.json'):
        return False
    if posix in EXACT_EXCLUDES:
        return False
    # Browser-delivered UI scripts render to the DOM, not the Windows console,
    # so emoji button labels there do not crash the build.
    if posix.startswith('static/') and posix.endswith('.js'):
        return False
    for frag in PATH_FRAGMENT_EXCLUDES:
        if frag in posix:
            return False
    return True


def _candidate_files():
    return [f for f in _git_ls_files() if _is_candidate(f)]


def _scan_for_emoji(rel_path):
    abs_path = os.path.join(REPO_ROOT, rel_path)
    offenders = []
    try:
        with open(abs_path, encoding='utf-8') as handle:
            lines = handle.read().splitlines()
    except (OSError, UnicodeDecodeError):
        return offenders
    for line_no, line in enumerate(lines, start=1):
        bad = sorted({ch for ch in line if _is_pictographic_emoji(ord(ch))})
        if bad:
            codes = ' '.join('U+%04X' % ord(c) for c in bad)
            offenders.append((line_no, codes))
    return offenders


def test_candidate_file_list_is_non_empty():
    # Sanity check: git ls-files resolved and produced code files to scan.
    assert _candidate_files(), 'no candidate code files found via git ls-files'


def test_no_emoji_in_code_files():
    failures = []
    for rel_path in _candidate_files():
        for line_no, codes in _scan_for_emoji(rel_path):
            failures.append('{0}:{1} ({2})'.format(rel_path, line_no, codes))
    assert not failures, (
        'Emoji / pictographic codepoints found in code files '
        '(emoji are only allowed in HTML pages; they crash the Windows '
        'console build):\n  ' + '\n  '.join(failures)
    )
