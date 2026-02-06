#!/usr/bin/env python3
"""
AudioMuse-AI - AI Playlist Naming Performance Test

Compares how different AI models perform on the same playlist naming prompt.
Sends identical song lists to multiple Ollama + OpenRouter models, runs N times
each, and produces a comparison report (console, TXT, HTML, JSON).

Usage:
  python testing_suite/test_ai_naming.py
  python testing_suite/test_ai_naming.py --config path/to/config.yaml
  python testing_suite/test_ai_naming.py --runs 10
  python testing_suite/test_ai_naming.py --dry-run
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime

import ftfy
import requests
import yaml


# ---------------------------------------------------------------------------
# Prompt template (inlined from ai.py:16-28)
# ---------------------------------------------------------------------------
CREATIVE_PROMPT_TEMPLATE = (
    "You are an expert music collector and MUST give a title to this playlist.\n"
    "The title MUST represent the mood and the activity of when you are listening to the playlist.\n"
    "The title MUST use ONLY standard ASCII (a-z, A-Z, 0-9, spaces, and - & ' ! . , ? ( ) [ ]).\n"
    "The title MUST be within the range of 5 to 40 characters long.\n"
    "No special fonts or emojis.\n"
    "* BAD EXAMPLES: 'Ambient Electronic Space - Electric Soundscapes - Emotional Waves' (Too long/descriptive)\n"
    "* BAD EXAMPLES: 'Blues Rock Fast Tracks' (Too direct/literal, not evocative enough)\n"
    "* BAD EXAMPLES: '\U0001d46f\U0001d4f0\U0001d4ea \U0001d4ea\U0001d4fb\U0001d4f8\U0001d4f7\U0001d4f2 \U0001d4ed\U0001d4ea\U0001d4fd\U0001d4fc' (Non-standard characters)\n\n"
    "CRITICAL: Your response MUST be ONLY the single playlist name. No explanations, no 'Playlist Name:', no numbering, no extra text or formatting whatsoever.\n\n"
    "This is the playlist:\n{song_list_sample}\n\n"
)

MIN_NAME_LENGTH = 5
MAX_NAME_LENGTH = 40


# ---------------------------------------------------------------------------
# Name cleaning (inlined from ai.py:30-43)
# ---------------------------------------------------------------------------
def clean_playlist_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = ftfy.fix_text(name)
    name = unicodedata.normalize('NFKC', name)
    cleaned = re.sub(r'[^a-zA-Z0-9\s\-\&\'!\.\,\?\(\)\[\]]', '', name)
    cleaned = re.sub(r'\s\(\d+\)$', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# ---------------------------------------------------------------------------
# Song data fetching
# ---------------------------------------------------------------------------
def fetch_songs_from_db(container: str, user: str, database: str, total: int) -> list[dict]:
    """Fetch random songs from the PostgreSQL database via docker exec."""
    query = (
        f"SELECT title, author FROM score "
        f"WHERE title IS NOT NULL AND author IS NOT NULL "
        f"ORDER BY RANDOM() LIMIT {total}"
    )
    cmd = [
        "docker", "exec", container,
        "psql", "-U", user, "-d", database,
        "-t", "-A", "-F", "|",
        "-c", query,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        print("ERROR: 'docker' command not found. Is Docker installed and in PATH?")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("ERROR: Database query timed out after 30 seconds.")
        sys.exit(1)

    if result.returncode != 0:
        print(f"ERROR: Database query failed.\n  Command: {' '.join(cmd)}\n  Stderr: {result.stderr.strip()}")
        sys.exit(1)

    songs = []
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = line.split('|', 1)
        if len(parts) == 2:
            songs.append({"title": parts[0].strip(), "author": parts[1].strip()})

    if len(songs) < total:
        print(f"ERROR: Not enough songs in database. Found {len(songs)}, need {total}.")
        sys.exit(1)

    return songs


def apply_defaults(config: dict) -> None:
    """Merge provider defaults (url, api_key) into each model entry."""
    defaults = config.get("defaults", {})
    for model in config.get("models", []):
        provider = model.get("provider", "")
        provider_defaults = defaults.get(provider, {})
        for key, value in provider_defaults.items():
            if key not in model:
                model[key] = value


def split_into_playlists(songs: list[dict], num_playlists: int, per_playlist: int) -> list[list[dict]]:
    """Split a flat song list into N playlists of M songs each."""
    playlists = []
    for i in range(num_playlists):
        start = i * per_playlist
        playlists.append(songs[start:start + per_playlist])
    return playlists


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(songs: list[dict], template: str | None = None) -> str:
    """Build the full prompt from a list of songs and an optional template."""
    formatted = "\n".join(f"- {s['title']} by {s['author']}" for s in songs)
    tpl = template if template else CREATIVE_PROMPT_TEMPLATE
    return tpl.format(song_list_sample=formatted)


# ---------------------------------------------------------------------------
# API calling (inlined from ai.py:47-183)
# ---------------------------------------------------------------------------
def call_model(model_cfg: dict, prompt: str, timeout: int) -> dict:
    """
    Call an AI model and return result dict with keys:
      name, raw_response, cleaned_name, valid, elapsed, error
    """
    provider = model_cfg["provider"]
    url = model_cfg["url"]
    model_id = model_cfg["model_id"]
    api_key = model_cfg.get("api_key", "")

    is_openai_format = (
        bool(api_key) or
        "openai" in url.lower() or
        "openrouter" in url.lower()
    )

    headers = {"Content-Type": "application/json"}

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if "openrouter" in url.lower():
        headers["HTTP-Referer"] = "https://github.com/NeptuneHub/AudioMuse-AI"
        headers["X-Title"] = "AudioMuse-AI"

    if is_openai_format:
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 8000,
        }
    else:
        payload = {
            "model": model_id,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": 8000,
                "temperature": 0.7,
            },
        }

    start_time = time.time()
    try:
        response = requests.post(
            url, headers=headers, data=json.dumps(payload),
            stream=True, timeout=timeout,
        )
        response.raise_for_status()

        full_raw = ""
        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode('utf-8', errors='ignore').strip()

            if line_str.startswith(':'):
                continue

            if line_str.startswith('data: '):
                line_str = line_str[6:]
                if line_str == '[DONE]':
                    break

            try:
                chunk = json.loads(line_str)
                if is_openai_format:
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        choice = chunk['choices'][0]
                        finish_reason = choice.get('finish_reason')
                        if finish_reason in ('stop', 'length'):
                            # Grab any final content before breaking
                            if 'delta' in choice:
                                c = choice['delta'].get('content')
                                if c:
                                    full_raw += c
                            break
                        if 'delta' in choice:
                            c = choice['delta'].get('content')
                            if c is not None:
                                full_raw += c
                        elif 'text' in choice:
                            t = choice.get('text')
                            if t is not None:
                                full_raw += t
                else:
                    if 'response' in chunk:
                        full_raw += chunk['response']
                    if chunk.get('done'):
                        break
            except json.JSONDecodeError:
                continue

        elapsed = time.time() - start_time

        # Strip think tags (inlined from ai.py:178-182)
        extracted = full_raw.strip()
        for tag in ["</think>", "[/INST]", "[/THOUGHT]"]:
            if tag in extracted:
                extracted = extracted.split(tag, 1)[-1].strip()

        if not extracted:
            return {
                "raw_response": full_raw,
                "cleaned_name": "",
                "valid": False,
                "elapsed": elapsed,
                "error": "Empty response after think-tag stripping",
            }

        cleaned = clean_playlist_name(extracted)
        valid = MIN_NAME_LENGTH <= len(cleaned) <= MAX_NAME_LENGTH

        return {
            "raw_response": extracted,
            "cleaned_name": cleaned,
            "valid": valid,
            "elapsed": elapsed,
            "error": None,
        }

    except requests.exceptions.ConnectionError:
        elapsed = time.time() - start_time
        return {
            "raw_response": "",
            "cleaned_name": "",
            "valid": False,
            "elapsed": elapsed,
            "error": "Connection refused",
        }
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        return {
            "raw_response": "",
            "cleaned_name": "",
            "valid": False,
            "elapsed": elapsed,
            "error": f"Timeout after {timeout}s",
        }
    except requests.exceptions.HTTPError as e:
        elapsed = time.time() - start_time
        detail = ""
        try:
            detail = e.response.text[:200]
        except Exception:
            pass
        return {
            "raw_response": "",
            "cleaned_name": "",
            "valid": False,
            "elapsed": elapsed,
            "error": f"HTTP {e.response.status_code}: {detail}",
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "raw_response": "",
            "cleaned_name": "",
            "valid": False,
            "elapsed": elapsed,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_summary_table(results: dict, timestamp: str) -> str:
    """Generate the ASCII summary table."""
    lines = []
    lines.append("=" * 70)
    lines.append(f" RESULTS - AI Playlist Naming Test ({timestamp})")
    lines.append("=" * 70)
    lines.append(f" {'Model':<22} {'Tests':>5}  {'Valid':>5}  {'Rate':>6}  {'Avg':>6}  {'Min':>6}  {'Max':>6}")
    lines.append("-" * 70)

    for model_name, model_data in results.items():
        all_runs = model_data["runs"]
        total = len(all_runs)
        valid = sum(1 for r in all_runs if r["valid"])
        rate = (valid / total * 100) if total > 0 else 0
        times = [r["elapsed"] for r in all_runs if r["error"] is None]
        avg_t = sum(times) / len(times) if times else 0
        min_t = min(times) if times else 0
        max_t = max(times) if times else 0

        lines.append(
            f" {model_name:<22} {total:>5}  {valid:>5}  {rate:>5.1f}%  {avg_t:>5.1f}s  {min_t:>5.1f}s  {max_t:>5.1f}s"
        )

    lines.append("-" * 70)
    return "\n".join(lines)


def generate_names_table(results: dict, playlists: list[list[dict]], num_runs: int) -> str:
    """Generate per-playlist names detail table."""
    lines = []
    model_names = list(results.keys())
    num_playlists = len(playlists)

    for pi in range(num_playlists):
        lines.append(f"\nPlaylist {pi + 1} - Generated Names:")
        header = f"  {'Model':<22}"
        for ri in range(num_runs):
            header += f"  {'Run ' + str(ri + 1):<24}"
        lines.append(header)
        lines.append("  " + "-" * (22 + num_runs * 26))

        for model_name in model_names:
            row = f"  {model_name:<22}"
            runs = results[model_name]["runs"]
            # Filter runs for this playlist
            playlist_runs = [r for r in runs if r["playlist_index"] == pi]
            for r in playlist_runs:
                name = r.get("cleaned_name", "")
                if r.get("error"):
                    name = f"[ERR: {r['error'][:15]}]"
                elif not r["valid"]:
                    name = f"[INVALID: {name[:12]}]"
                # Truncate to fit column
                if len(name) > 23:
                    name = name[:20] + "..."
                row += f"  {name:<24}"
            lines.append(row)

    return "\n".join(lines)


def generate_html_report(results: dict, playlists: list[list[dict]],
                         num_runs: int, timestamp: str, config: dict,
                         save_raw: bool) -> str:
    """Generate a self-contained HTML report."""
    model_names = list(results.keys())
    num_playlists = len(playlists)

    # Build summary rows
    summary_rows = ""
    for model_name, model_data in results.items():
        all_runs = model_data["runs"]
        total = len(all_runs)
        valid = sum(1 for r in all_runs if r["valid"])
        rate = (valid / total * 100) if total > 0 else 0
        errors = sum(1 for r in all_runs if r["error"])
        times = [r["elapsed"] for r in all_runs if r["error"] is None]
        avg_t = sum(times) / len(times) if times else 0
        min_t = min(times) if times else 0
        max_t = max(times) if times else 0

        rate_class = "pass" if rate >= 80 else ("warn" if rate >= 50 else "fail")
        provider = model_data.get("provider", "")

        summary_rows += f"""<tr>
            <td>{model_name}</td><td>{provider}</td>
            <td>{total}</td><td>{valid}</td><td>{errors}</td>
            <td class="{rate_class}">{rate:.1f}%</td>
            <td>{avg_t:.2f}s</td><td>{min_t:.2f}s</td><td>{max_t:.2f}s</td>
        </tr>\n"""

    # Build per-playlist detail sections
    playlist_sections = ""
    for pi in range(num_playlists):
        song_list_html = "<ul>\n"
        for s in playlists[pi]:
            song_list_html += f"  <li>{s['title']} &mdash; {s['author']}</li>\n"
        song_list_html += "</ul>"

        detail_rows = ""
        for model_name in model_names:
            runs = [r for r in results[model_name]["runs"] if r["playlist_index"] == pi]
            valid_count = sum(1 for r in runs if r["valid"])
            total_count = len(runs)
            times = [r["elapsed"] for r in runs if r["error"] is None]
            avg_t = sum(times) / len(times) if times else 0
            rate = (valid_count / total_count * 100) if total_count else 0
            rate_class = "pass" if rate >= 80 else ("warn" if rate >= 50 else "fail")

            # Build all names into a single cell
            names_html = ""
            for ri, r in enumerate(runs):
                name = r.get("cleaned_name", "")
                error = r.get("error")
                raw = r.get("raw_response", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

                if error:
                    names_html += f'<div class="name-entry error">Run {ri + 1}: <em>{error}</em> ({r["elapsed"]:.1f}s)</div>\n'
                elif r["valid"]:
                    raw_detail = f' <details class="inline-raw"><summary>raw</summary><pre>{raw}</pre></details>' if save_raw and raw else ""
                    names_html += f'<div class="name-entry pass">Run {ri + 1}: {name} ({r["elapsed"]:.1f}s){raw_detail}</div>\n'
                else:
                    raw_detail = f' <details class="inline-raw"><summary>raw</summary><pre>{raw}</pre></details>' if save_raw and raw else ""
                    names_html += f'<div class="name-entry fail">Run {ri + 1}: {name} <span class="len">({len(name)} chars)</span> ({r["elapsed"]:.1f}s){raw_detail}</div>\n'

            detail_rows += f"""<tr>
                <td><strong>{model_name}</strong><br><span class="meta">{results[model_name].get('provider', '')}</span></td>
                <td class="{rate_class}">{valid_count}/{total_count} ({rate:.0f}%)</td>
                <td>{avg_t:.2f}s</td>
                <td class="names-cell">{names_html}</td>
            </tr>\n"""

        playlist_sections += f"""
        <details open>
            <summary><h3>Playlist {pi + 1}</h3></summary>
            <div class="song-list">
                <strong>Songs used:</strong>
                {song_list_html}
            </div>
            <table>
                <thead><tr>
                    <th>Model</th><th>Valid</th><th>Avg Time</th>
                    <th>Generated Names</th>
                </tr></thead>
                <tbody>{detail_rows}</tbody>
            </table>
        </details>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Playlist Naming Test - {timestamp}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           margin: 2rem; background: #f8f9fa; color: #212529; }}
    h1 {{ color: #2563eb; }}
    h2 {{ margin-top: 2rem; border-bottom: 2px solid #dee2e6; padding-bottom: 0.5rem; }}
    h3 {{ margin: 0; display: inline; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: #fff; }}
    th, td {{ border: 1px solid #dee2e6; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #e9ecef; font-weight: 600; }}
    tr:nth-child(even) {{ background: #f8f9fa; }}
    .pass {{ background: #d4edda; color: #155724; font-weight: bold; }}
    .fail {{ background: #f8d7da; color: #721c24; font-weight: bold; }}
    .warn {{ background: #fff3cd; color: #856404; font-weight: bold; }}
    .error {{ background: #f8d7da; color: #721c24; }}
    .names-cell {{ padding: 0.25rem 0.5rem; }}
    .name-entry {{ padding: 0.3rem 0.5rem; margin: 0.2rem 0; border-radius: 3px; font-size: 0.9rem; }}
    .name-entry.pass {{ background: #d4edda; color: #155724; font-weight: normal; }}
    .name-entry.fail {{ background: #fff3cd; color: #856404; font-weight: normal; }}
    .name-entry.error {{ background: #f8d7da; color: #721c24; font-weight: normal; }}
    .name-entry .len {{ font-size: 0.8rem; opacity: 0.7; }}
    .meta {{ font-size: 0.8rem; color: #6c757d; }}
    .inline-raw {{ display: inline; margin-left: 0.5rem; }}
    .inline-raw summary {{ display: inline; background: none; padding: 0; font-size: 0.75rem;
                           color: #6c757d; text-decoration: underline; }}
    .inline-raw pre {{ margin-top: 0.25rem; }}
    details {{ margin: 1rem 0; }}
    summary {{ cursor: pointer; padding: 0.5rem; background: #e9ecef; border-radius: 4px; }}
    summary:hover {{ background: #dee2e6; }}
    .song-list {{ background: #fff; padding: 1rem; border: 1px solid #dee2e6;
                  border-radius: 4px; margin: 0.5rem 0; max-height: 300px; overflow-y: auto; }}
    .song-list ul {{ margin: 0.5rem 0; padding-left: 1.5rem; }}
    .config {{ background: #fff; padding: 1rem; border: 1px solid #dee2e6;
               border-radius: 4px; font-family: monospace; font-size: 0.85rem; }}
    .prompt-box {{ white-space: pre-wrap; max-width: 100%; background: #fff; padding: 1rem;
                   border: 1px solid #dee2e6; border-radius: 4px; font-size: 0.9rem; }}
    .prompt-box em {{ color: #2563eb; font-style: normal; font-weight: bold; }}
    pre {{ white-space: pre-wrap; word-break: break-all; max-width: 400px;
           font-size: 0.8rem; background: #f1f3f5; padding: 0.5rem; border-radius: 4px; }}
    .footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #dee2e6;
               color: #6c757d; font-size: 0.85rem; }}
</style>
</head>
<body>
<h1>AI Playlist Naming Test</h1>
<p><strong>Date:</strong> {timestamp} &nbsp;|&nbsp;
   <strong>Runs per model:</strong> {num_runs} &nbsp;|&nbsp;
   <strong>Playlists:</strong> {num_playlists} &nbsp;|&nbsp;
   <strong>Songs per playlist:</strong> {len(playlists[0]) if playlists else 0}</p>

<h2>Summary</h2>
<table>
    <thead><tr>
        <th>Model</th><th>Provider</th><th>Tests</th><th>Valid</th><th>Errors</th>
        <th>Valid Rate</th><th>Avg Time</th><th>Min Time</th><th>Max Time</th>
    </tr></thead>
    <tbody>{summary_rows}</tbody>
</table>

<h2>Prompt Used</h2>
<details open>
    <summary>Show prompt template</summary>
    <pre class="prompt-box">{(config.get('prompt') or CREATIVE_PROMPT_TEMPLATE).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('{song_list_sample}', '<em>&lbrace;song_list_sample&rbrace;</em>')}</pre>
</details>

<h2>Detailed Results</h2>
{playlist_sections}

<h2>Test Configuration</h2>
<div class="config">{json.dumps(config, indent=2, default=str)}</div>

<div class="footer">
    Generated by AudioMuse-AI Testing Suite
</div>
</body>
</html>"""
    return html


def generate_json_report(results: dict, playlists: list[list[dict]],
                         timestamp: str, config: dict) -> dict:
    """Generate the full JSON report."""
    report = {
        "timestamp": timestamp,
        "config": config,
        "playlists": [
            [{"title": s["title"], "author": s["author"]} for s in pl]
            for pl in playlists
        ],
        "models": {},
    }

    for model_name, model_data in results.items():
        all_runs = model_data["runs"]
        total = len(all_runs)
        valid = sum(1 for r in all_runs if r["valid"])
        errors = sum(1 for r in all_runs if r["error"])
        times = [r["elapsed"] for r in all_runs if r["error"] is None]

        report["models"][model_name] = {
            "provider": model_data.get("provider", ""),
            "model_id": model_data.get("model_id", ""),
            "url": model_data.get("url", ""),
            "summary": {
                "total_tests": total,
                "valid": valid,
                "invalid": total - valid - errors,
                "errors": errors,
                "valid_rate": round(valid / total * 100, 1) if total > 0 else 0,
                "avg_time": round(sum(times) / len(times), 3) if times else 0,
                "min_time": round(min(times), 3) if times else 0,
                "max_time": round(max(times), 3) if times else 0,
            },
            "runs": [
                {
                    "playlist_index": r["playlist_index"],
                    "run_index": r["run_index"],
                    "cleaned_name": r.get("cleaned_name", ""),
                    "raw_response": r.get("raw_response", ""),
                    "valid": r["valid"],
                    "elapsed": round(r["elapsed"], 3),
                    "error": r.get("error"),
                    "name_length": len(r.get("cleaned_name", "")),
                }
                for r in all_runs
            ],
        }

    return report


# ---------------------------------------------------------------------------
# Main test loop
# ---------------------------------------------------------------------------
def run_tests(config: dict, dry_run: bool = False) -> tuple[dict, list[list[dict]]]:
    """
    Execute the full test suite.

    Returns:
        (results_dict, playlists)
        results_dict keys are model names, values have 'runs' list and metadata.
    """
    pg = config["postgres"]
    tc = config["test_config"]
    models = [m for m in config["models"] if m.get("enabled", False)]

    if not models:
        print("ERROR: No models enabled in configuration.")
        sys.exit(1)

    num_runs = tc["num_runs_per_model"]
    num_playlists = tc["num_playlists"]
    songs_per = tc["songs_per_playlist"]
    timeout = tc.get("timeout_per_request", 120)
    total_songs = num_playlists * songs_per

    # Use sample_songs from config if present, otherwise fetch from DB
    sample_songs = config.get("sample_songs")
    if sample_songs:
        # Single playlist when using hardcoded songs (no point repeating identical lists)
        num_playlists = 1
        playlists = [sample_songs[:songs_per]]
        print(f"Using {len(sample_songs)} songs from config (sample_songs, 1 playlist)...\n")
    else:
        print(f"Fetching {total_songs} songs from database ({pg['container_name']})...")
        songs = fetch_songs_from_db(
            pg["container_name"], pg["user"], pg["database"], total_songs,
        )
        playlists = split_into_playlists(songs, num_playlists, songs_per)
        print(f"  OK - {len(songs)} songs split into {num_playlists} playlists of {songs_per}\n")

    # Build prompts (same for all models)
    # Use prompt from config if provided, otherwise fall back to hardcoded default
    prompt_template = config.get("prompt")
    prompts = [build_prompt(pl, prompt_template) for pl in playlists]

    if dry_run:
        print("=== DRY RUN MODE ===")
        print(f"Would test {len(models)} model(s), {num_playlists} playlist(s), {num_runs} run(s) each\n")
        for mi, m in enumerate(models):
            print(f"  Model {mi + 1}: {m['name']} ({m['provider']}) - {m['model_id']}")
        print(f"\nPlaylist 1 prompt preview (first 500 chars):")
        print(prompts[0][:500])
        print("...")
        return {}, playlists

    # Run tests
    results = {}
    total_models = len(models)
    connection_failures = set()

    for mi, model in enumerate(models):
        model_name = model["name"]
        print(f"[{mi + 1}/{total_models}] Testing: {model_name} ({model['provider']})")

        results[model_name] = {
            "provider": model["provider"],
            "model_id": model["model_id"],
            "url": model["url"],
            "runs": [],
        }

        # Skip if previous connection to same URL failed
        if model["url"] in connection_failures:
            print(f"  Skipping (connection to {model['url']} already failed)\n")
            for pi in range(num_playlists):
                for ri in range(num_runs):
                    results[model_name]["runs"].append({
                        "playlist_index": pi,
                        "run_index": ri,
                        "raw_response": "",
                        "cleaned_name": "",
                        "valid": False,
                        "elapsed": 0,
                        "error": "Skipped (connection failed)",
                    })
            continue

        model_valid = 0
        model_total = 0
        model_times = []
        abort_model = False

        for pi in range(num_playlists):
            for ri in range(num_runs):
                if abort_model:
                    results[model_name]["runs"].append({
                        "playlist_index": pi,
                        "run_index": ri,
                        "raw_response": "",
                        "cleaned_name": "",
                        "valid": False,
                        "elapsed": 0,
                        "error": "Skipped (connection failed)",
                    })
                    continue

                model_total += 1
                status_prefix = f"  Playlist {pi + 1}: Run {ri + 1}/{num_runs}..."

                result = call_model(model, prompts[pi], timeout)
                result["playlist_index"] = pi
                result["run_index"] = ri
                results[model_name]["runs"].append(result)

                if result["error"] == "Connection refused":
                    print(f"{status_prefix} FAIL  (connection refused)")
                    connection_failures.add(model["url"])
                    abort_model = True
                    continue

                if result["error"]:
                    print(f"{status_prefix} ERR   {result['elapsed']:.1f}s  {result['error']}")
                elif result["valid"]:
                    model_valid += 1
                    model_times.append(result["elapsed"])
                    print(f"{status_prefix} OK    {result['elapsed']:.1f}s  \"{result['cleaned_name']}\"")
                else:
                    name = result["cleaned_name"]
                    print(f"{status_prefix} INVALID {result['elapsed']:.1f}s  \"{name}\" ({len(name)} chars)")
                    model_times.append(result["elapsed"])

        # Model summary
        if model_total > 0 and not abort_model:
            avg_t = sum(model_times) / len(model_times) if model_times else 0
            rate = model_valid / model_total * 100
            print(f"  Result: {model_valid}/{model_total} valid ({rate:.1f}%), avg {avg_t:.1f}s\n")
        elif abort_model:
            print(f"  Result: Aborted (connection failed)\n")

    return results, playlists


def save_reports(results: dict, playlists: list[list[dict]], config: dict,
                 num_runs: int, output_dir: str, save_raw: bool):
    """Save TXT, HTML, and JSON reports to disk."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    file_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Console / TXT summary
    summary = generate_summary_table(results, timestamp)
    names_detail = generate_names_table(results, playlists, num_runs)
    full_txt = summary + "\n" + names_detail + "\n"

    print("\n" + full_txt)

    txt_path = os.path.join(output_dir, f"ai_naming_{file_ts}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_txt)
    print(f"TXT report saved: {txt_path}")

    # HTML report
    html = generate_html_report(results, playlists, num_runs, timestamp, config, save_raw)
    html_path = os.path.join(output_dir, f"ai_naming_{file_ts}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved: {html_path}")

    # JSON report
    json_data = generate_json_report(results, playlists, timestamp, config)
    json_path = os.path.join(output_dir, f"ai_naming_{file_ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"JSON report saved: {json_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="AudioMuse-AI - AI Playlist Naming Performance Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", "-c", type=str,
                        default="testing_suite/ai_naming_test_config.yaml",
                        help="Path to YAML config file (default: testing_suite/ai_naming_test_config.yaml)")
    parser.add_argument("--runs", "-n", type=int, default=None,
                        help="Override num_runs_per_model from config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch songs and build prompts, but don't call any APIs")

    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        print(f"Usage: python testing_suite/test_ai_naming.py --config path/to/config.yaml")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Merge provider defaults (url, api_key) into each model entry
    apply_defaults(config)

    # Apply CLI overrides
    if args.runs is not None:
        config["test_config"]["num_runs_per_model"] = args.runs

    num_runs = config["test_config"]["num_runs_per_model"]
    output_cfg = config.get("output", {})
    output_dir = output_cfg.get("directory", "testing_suite/reports/ai_naming")
    save_raw = output_cfg.get("save_raw_responses", True)

    print("=" * 60)
    print(" AudioMuse-AI - AI Playlist Naming Performance Test")
    print("=" * 60)

    enabled = [m for m in config["models"] if m.get("enabled", False)]
    print(f" Models:    {len(enabled)} enabled")
    print(f" Playlists: {config['test_config']['num_playlists']}")
    print(f" Runs/model: {num_runs}")
    print(f" Songs/playlist: {config['test_config']['songs_per_playlist']}")
    print("=" * 60 + "\n")

    # Run tests
    results, playlists = run_tests(config, dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry run complete. No API calls were made.")
        return

    if not results:
        print("No results to report.")
        return

    # Generate and save reports
    save_reports(results, playlists, config, num_runs, output_dir, save_raw)


if __name__ == "__main__":
    main()
