import os
import subprocess

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

EM_DASH = chr(0x2014)

PATH_FRAGMENT_EXCLUDES = (
    '.venv',
    'node_modules',
    '/vendor/',
    '.min.js',
    'plotly-',
)


def _git_ls_files():
    out = subprocess.check_output(['git', 'ls-files'], cwd=REPO_ROOT).decode('utf-8')
    return [line for line in out.splitlines() if line]


def _is_candidate(rel_path):
    posix = rel_path.replace('\\', '/')
    for frag in PATH_FRAGMENT_EXCLUDES:
        if frag in posix:
            return False
    return True


def _read_text(rel_path):
    abs_path = os.path.join(REPO_ROOT, rel_path)
    try:
        with open(abs_path, encoding='utf-8') as handle:
            return handle.read()
    except (OSError, UnicodeDecodeError):
        return None


def _candidate_files():
    return [f for f in _git_ls_files() if _is_candidate(f)]


def test_candidate_file_list_is_non_empty():
    assert _candidate_files(), 'no candidate files found via git ls-files'


def test_no_em_dash_in_repo():
    failures = []
    for rel_path in _candidate_files():
        text = _read_text(rel_path)
        if text is None:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if EM_DASH in line:
                failures.append('{0}:{1}'.format(rel_path, line_no))
    assert not failures, (
        'Em-dash (U+2014) is banned - use a plain hyphen "-" instead:\n  '
        + '\n  '.join(failures)
    )
