"""Shared media server helper utilities."""

import re


def detect_path_format(tracks):
    """Classify track path samples as absolute, relative, none, or mixed."""
    def _is_absolute_path(path):
        if not path:
            return False
        path_str = str(path)
        lower = path_str.lower()
        return (
            path_str.startswith('/')
            or path_str.startswith('\\')
            or lower.startswith('file://')
            or re.match(r'^[A-Za-z]:[\\/]', path_str)
        )

    paths = []
    for track in tracks or []:
        if not isinstance(track, dict):
            continue
        # Support lowercase/uppercase path keys and legacy URL fields.
        path = (
            track.get('path')
            or track.get('Path')
            or track.get('url')
            or track.get('Url')
        )
        if path:
            paths.append(path)

    if not paths:
        return 'none'

    ratio = sum(1 for p in paths if _is_absolute_path(p)) / len(paths)
    if ratio >= 0.8:
        return 'absolute'
    if ratio <= 0.2:
        return 'relative'
    return 'mixed'
