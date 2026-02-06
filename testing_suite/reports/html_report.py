"""
HTML Report Generator for AudioMuse-AI Testing Suite.

Generates a comprehensive, self-contained HTML report with:
  - Overall pass/fail summary with status badges
  - Per-category breakdowns with expandable details
  - Color-coded results (green/red/yellow/gray)
  - Performance charts (latency comparisons)
  - Filterable and sortable tables
  - Instance A vs B side-by-side comparisons
"""

import json
import os
from datetime import datetime
from typing import Dict

from testing_suite.utils import ComparisonReport, TestStatus


def generate_html_report(report: ComparisonReport, output_path: str) -> str:
    """Generate a self-contained HTML report and write it to output_path."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    report_dict = report.to_dict()
    categories = report_dict.get("categories", {})

    # Build category HTML sections
    category_sections = ""
    for cat_name, cat_data in categories.items():
        category_sections += _build_category_section(cat_name, cat_data)

    # Build performance chart data (if performance category exists)
    perf_chart_data = _build_performance_chart_data(categories.get("performance", {}))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AudioMuse-AI Comparison Report</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; line-height: 1.6; padding: 20px; }}
.container {{ max-width: 1400px; margin: 0 auto; }}
h1 {{ color: #58a6ff; margin-bottom: 10px; font-size: 28px; }}
h2 {{ color: #79c0ff; margin: 30px 0 15px; font-size: 22px; border-bottom: 1px solid #21262d; padding-bottom: 8px; }}
h3 {{ color: #c9d1d9; margin: 15px 0 10px; font-size: 16px; }}
.header {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 24px; margin-bottom: 24px; }}
.header-meta {{ color: #8b949e; font-size: 14px; margin-top: 8px; }}
.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 20px 0; }}
.summary-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; text-align: center; }}
.summary-card .number {{ font-size: 36px; font-weight: bold; }}
.summary-card .label {{ color: #8b949e; font-size: 14px; margin-top: 4px; }}
.status-pass {{ color: #3fb950; }}
.status-fail {{ color: #f85149; }}
.status-warn {{ color: #d29922; }}
.status-skip {{ color: #8b949e; }}
.status-error {{ color: #f85149; }}
.badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; text-transform: uppercase; }}
.badge-pass {{ background: #1b4332; color: #3fb950; }}
.badge-fail {{ background: #490202; color: #f85149; }}
.badge-warn {{ background: #3d2e00; color: #d29922; }}
.badge-skip {{ background: #21262d; color: #8b949e; }}
.badge-error {{ background: #490202; color: #f85149; }}
.category {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin-bottom: 20px; overflow: hidden; }}
.category-header {{ padding: 16px 20px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #21262d; }}
.category-header:hover {{ background: #1c2128; }}
.category-header .cat-title {{ font-size: 18px; font-weight: 600; }}
.category-header .cat-stats {{ display: flex; gap: 12px; font-size: 13px; }}
.category-body {{ display: none; }}
.category-body.open {{ display: block; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ background: #1c2128; padding: 10px 12px; text-align: left; color: #8b949e; font-weight: 600; position: sticky; top: 0; }}
td {{ padding: 10px 12px; border-bottom: 1px solid #21262d; }}
tr:hover {{ background: #1c2128; }}
.message-cell {{ max-width: 600px; word-wrap: break-word; white-space: pre-wrap; font-size: 12px; }}
.duration {{ color: #8b949e; font-size: 12px; }}
.filter-bar {{ margin: 10px 0; padding: 10px 16px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
.filter-btn {{ background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 4px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; }}
.filter-btn:hover, .filter-btn.active {{ background: #30363d; }}
.instances {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 10px 0; }}
.instance-box {{ background: #1c2128; border: 1px solid #30363d; border-radius: 6px; padding: 12px; }}
.instance-box h4 {{ color: #79c0ff; margin-bottom: 8px; }}
.toggle-arrow {{ transition: transform 0.2s; font-size: 12px; }}
.toggle-arrow.open {{ transform: rotate(90deg); }}
.perf-bar {{ display: flex; align-items: center; gap: 8px; margin: 2px 0; }}
.perf-bar-a {{ background: #388bfd; height: 18px; border-radius: 3px; min-width: 2px; }}
.perf-bar-b {{ background: #f0883e; height: 18px; border-radius: 3px; min-width: 2px; }}
.perf-legend {{ display: flex; gap: 16px; margin: 10px 0; font-size: 12px; }}
.perf-legend span {{ display: flex; align-items: center; gap: 4px; }}
.perf-legend .dot-a {{ width: 10px; height: 10px; background: #388bfd; border-radius: 50%; }}
.perf-legend .dot-b {{ width: 10px; height: 10px; background: #f0883e; border-radius: 50%; }}
.overall-badge {{ font-size: 20px; padding: 6px 20px; }}
footer {{ text-align: center; color: #484f58; margin-top: 40px; padding: 20px; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>AudioMuse-AI Comparison Report</h1>
    <div class="header-meta">
      Generated: {report.timestamp} UTC<br>
      Instance A: <strong>{report.instance_a_name}</strong> (branch: {report.instance_a_branch})<br>
      Instance B: <strong>{report.instance_b_name}</strong> (branch: {report.instance_b_branch})
    </div>
  </div>

  <div class="summary-grid">
    <div class="summary-card">
      <div class="number {_status_class(report.overall_status)}">{report.overall_status.value}</div>
      <div class="label">Overall Status</div>
    </div>
    <div class="summary-card">
      <div class="number">{report.total_tests}</div>
      <div class="label">Total Tests</div>
    </div>
    <div class="summary-card">
      <div class="number status-pass">{report.total_passed}</div>
      <div class="label">Passed</div>
    </div>
    <div class="summary-card">
      <div class="number status-fail">{report.total_failed}</div>
      <div class="label">Failed</div>
    </div>
    <div class="summary-card">
      <div class="number status-error">{report.total_errors}</div>
      <div class="label">Errors</div>
    </div>
    <div class="summary-card">
      <div class="number status-warn">{sum(c.warned for c in report.categories.values())}</div>
      <div class="label">Warnings</div>
    </div>
  </div>

  {category_sections}

  {_build_perf_visual(perf_chart_data) if perf_chart_data else ""}

</div>

<footer>
  AudioMuse-AI Testing &amp; Comparison Suite v1.0.0
</footer>

<script>
document.querySelectorAll('.category-header').forEach(header => {{
  header.addEventListener('click', () => {{
    const body = header.nextElementSibling;
    const arrow = header.querySelector('.toggle-arrow');
    body.classList.toggle('open');
    arrow.classList.toggle('open');
  }});
}});

function filterResults(cat, status) {{
  const table = document.getElementById('table-' + cat);
  if (!table) return;
  const rows = table.querySelectorAll('tbody tr');
  rows.forEach(row => {{
    if (status === 'all' || row.dataset.status === status) {{
      row.style.display = '';
    }} else {{
      row.style.display = 'none';
    }}
  }});
}}
</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


def _status_class(status) -> str:
    """Get CSS class for a status value."""
    if isinstance(status, TestStatus):
        status = status.value
    return f"status-{status.lower()}"


def _badge_html(status) -> str:
    """Generate a badge HTML element for a status."""
    if isinstance(status, TestStatus):
        status = status.value
    return f'<span class="badge badge-{status.lower()}">{status}</span>'


def _build_category_section(cat_name: str, cat_data: dict) -> str:
    """Build HTML section for a test category."""
    total = cat_data.get("total", 0)
    passed = cat_data.get("passed", 0)
    failed = cat_data.get("failed", 0)
    warned = cat_data.get("warned", 0)
    skipped = cat_data.get("skipped", 0)
    errors = cat_data.get("errors", 0)
    results = cat_data.get("results", [])

    # Category display name
    display_names = {
        "api": "API Endpoints",
        "database": "Database Quality",
        "docker": "Docker & Infrastructure",
        "performance": "Performance Benchmarks",
        "existing_tests": "Existing Test Suite",
    }
    display_name = display_names.get(cat_name, cat_name.replace("_", " ").title())

    # Build table rows
    rows = ""
    for r in results:
        status = r.get("status", "SKIP")
        duration = r.get("duration_seconds", 0)
        duration_str = f"{duration:.2f}s" if duration else "-"

        # Format values for display
        val_a = r.get("instance_a_value", "")
        val_b = r.get("instance_b_value", "")
        if isinstance(val_a, (dict, list)):
            val_a = json.dumps(val_a, indent=1, default=str)[:200]
        if isinstance(val_b, (dict, list)):
            val_b = json.dumps(val_b, indent=1, default=str)[:200]

        rows += f"""
        <tr data-status="{status.lower()}">
          <td>{_badge_html(status)}</td>
          <td>{r.get('name', '')}</td>
          <td class="message-cell">{_escape_html(r.get('message', ''))}</td>
          <td><small>{_escape_html(str(val_a)[:150])}</small></td>
          <td><small>{_escape_html(str(val_b)[:150])}</small></td>
          <td class="duration">{duration_str}</td>
        </tr>"""

    return f"""
  <div class="category">
    <div class="category-header">
      <div>
        <span class="toggle-arrow">&#9654;</span>
        <span class="cat-title">{display_name}</span>
      </div>
      <div class="cat-stats">
        <span class="status-pass">{passed} passed</span>
        <span class="status-fail">{failed} failed</span>
        <span class="status-warn">{warned} warn</span>
        <span class="status-skip">{skipped} skip</span>
        <span class="status-error">{errors} err</span>
        <span>({total} total)</span>
      </div>
    </div>
    <div class="category-body">
      <div class="filter-bar">
        <span style="color:#8b949e;font-size:12px;">Filter:</span>
        <button class="filter-btn" onclick="filterResults('{cat_name}','all')">All</button>
        <button class="filter-btn" onclick="filterResults('{cat_name}','pass')">Pass</button>
        <button class="filter-btn" onclick="filterResults('{cat_name}','fail')">Fail</button>
        <button class="filter-btn" onclick="filterResults('{cat_name}','warn')">Warn</button>
        <button class="filter-btn" onclick="filterResults('{cat_name}','error')">Error</button>
        <button class="filter-btn" onclick="filterResults('{cat_name}','skip')">Skip</button>
      </div>
      <table id="table-{cat_name}">
        <thead>
          <tr>
            <th style="width:70px">Status</th>
            <th style="width:250px">Test Name</th>
            <th>Message</th>
            <th style="width:150px">Instance A</th>
            <th style="width:150px">Instance B</th>
            <th style="width:70px">Duration</th>
          </tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </div>
  </div>"""


def _build_performance_chart_data(perf_category: dict) -> list:
    """Extract performance comparison data for visualization."""
    if not perf_category:
        return []

    chart_data = []
    for result in perf_category.get("results", []):
        name = result.get("name", "")
        if not name.startswith("Latency:"):
            continue

        val_a = result.get("instance_a_value", {})
        val_b = result.get("instance_b_value", {})

        if isinstance(val_a, dict) and isinstance(val_b, dict):
            mean_a = val_a.get("mean", 0)
            mean_b = val_b.get("mean", 0)
            if mean_a > 0 or mean_b > 0:
                chart_data.append({
                    "name": name.replace("Latency: ", ""),
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "p95_a": val_a.get("p95", 0),
                    "p95_b": val_b.get("p95", 0),
                })

    return chart_data


def _build_perf_visual(chart_data: list) -> str:
    """Build a visual performance comparison section."""
    if not chart_data:
        return ""

    max_val = max(
        max(d["mean_a"], d["mean_b"], d["p95_a"], d["p95_b"])
        for d in chart_data
    ) or 1

    bars = ""
    for d in chart_data:
        width_a = max(2, int(d["mean_a"] / max_val * 400))
        width_b = max(2, int(d["mean_b"] / max_val * 400))

        bars += f"""
      <div style="margin-bottom:12px;">
        <div style="font-size:13px;color:#c9d1d9;margin-bottom:4px;">{d['name']}</div>
        <div class="perf-bar">
          <div style="width:30px;text-align:right;font-size:11px;color:#8b949e;">A</div>
          <div class="perf-bar-a" style="width:{width_a}px;" title="Mean: {d['mean_a']*1000:.1f}ms, P95: {d['p95_a']*1000:.1f}ms"></div>
          <span style="font-size:11px;color:#8b949e;">{d['mean_a']*1000:.1f}ms</span>
        </div>
        <div class="perf-bar">
          <div style="width:30px;text-align:right;font-size:11px;color:#8b949e;">B</div>
          <div class="perf-bar-b" style="width:{width_b}px;" title="Mean: {d['mean_b']*1000:.1f}ms, P95: {d['p95_b']*1000:.1f}ms"></div>
          <span style="font-size:11px;color:#8b949e;">{d['mean_b']*1000:.1f}ms</span>
        </div>
      </div>"""

    return f"""
  <h2>Performance Visual Comparison (Mean Latency)</h2>
  <div class="perf-legend">
    <span><span class="dot-a"></span> Instance A</span>
    <span><span class="dot-b"></span> Instance B</span>
  </div>
  <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;">
    {bars}
  </div>"""


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))
