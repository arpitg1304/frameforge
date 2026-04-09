"""Generate a self-contained HTML benchmark report for GitHub Pages."""

from __future__ import annotations

import json
from pathlib import Path

from frameforge.benchmark.runner import BenchmarkResult

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmarks — frameforge</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {{
    --bg:#101010;--surface:#171717;--card:#1c1c1c;--card-hover:#222;
    --border:#2a2a2a;--text:#b0b0b0;--text-dim:#666;--text-bright:#e8e8e8;
    --accent:#e2a442;--teal:#3dbea0;
  }}
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:'Space Grotesk',-apple-system,sans-serif;background:var(--bg);color:var(--text);line-height:1.7;-webkit-font-smoothing:antialiased}}
  a{{color:var(--accent);text-decoration:none}}
  nav{{position:fixed;top:0;left:0;right:0;z-index:100;background:rgba(16,16,16,0.8);backdrop-filter:blur(20px);border-bottom:1px solid var(--border)}}
  nav .inner{{max-width:960px;margin:0 auto;padding:0 2rem;display:flex;align-items:center;justify-content:space-between;height:56px}}
  nav .logo{{font-weight:700;font-size:1.05rem;color:var(--text-bright)}}
  nav .logo span{{color:var(--accent)}}
  nav .links{{display:flex;gap:1.75rem;font-size:0.85rem;font-weight:500}}
  nav .links a{{color:var(--text-dim)}} nav .links a:hover{{color:var(--text-bright);text-decoration:none}}
  .content{{max-width:960px;margin:0 auto;padding:5rem 2rem 4rem}}
  .page-hero{{padding-top:4rem;margin-bottom:2.5rem}}
  .page-hero .ey{{font-size:0.78rem;font-weight:600;color:var(--accent);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.75rem}}
  .page-hero h1{{font-size:2.2rem;font-weight:700;color:var(--text-bright);letter-spacing:-0.03em;margin-bottom:0.4rem}}
  .page-hero p{{color:var(--text-dim);font-size:1rem}}
  .card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:1.5rem;margin-bottom:1.25rem}}
  .card h2{{color:var(--text-dim);font-size:0.72rem;text-transform:uppercase;font-weight:600;letter-spacing:0.06em;margin-bottom:1rem}}
  .chart-container{{max-width:800px;margin:0 auto}}
  .sys-pills{{display:flex;flex-wrap:wrap;gap:0.4rem}}
  .sys-pills span{{background:var(--surface);border:1px solid var(--border);padding:0.35rem 0.7rem;border-radius:5px;font-size:0.8rem;font-family:'JetBrains Mono',monospace}}
  table{{width:100%;border-collapse:collapse}}
  th,td{{text-align:left;padding:0.75rem 1rem}}
  thead th{{background:var(--surface);color:var(--text-dim);font-size:0.72rem;text-transform:uppercase;font-weight:600;letter-spacing:0.05em;border-bottom:1px solid var(--border)}}
  tbody td{{font-size:0.88rem;border-bottom:1px solid var(--border);font-variant-numeric:tabular-nums;font-family:'JetBrains Mono',monospace}}
  tbody tr:last-child td{{border-bottom:none}} tbody tr{{transition:background 0.1s}} tbody tr:hover{{background:var(--card-hover)}}
  footer{{border-top:1px solid var(--border);padding:2.5rem 2rem;text-align:center;color:var(--text-dim);font-size:0.82rem}}
</style>
</head>
<body>
<nav><div class="inner"><div class="logo"><a href="index.html" style="color:inherit"><span>frame</span>forge</a></div><div class="links"><a href="index.html">Home</a><a href="video-compression.html">Video Formats</a></div></div></nav>
<div class="content">
<div class="page-hero"><div class="ey">Performance</div><h1>Benchmark Report</h1><p>Video decoding backend comparison on real hardware.</p></div>
<div class="card"><h2>System</h2><div class="sys-pills" id="sysinfo"></div></div>
<div class="card"><h2>Decode Throughput (frames/sec)</h2><div class="chart-container"><canvas id="throughputChart"></canvas></div></div>
<div class="card" style="overflow:hidden;padding:0"><h2 style="padding:1.5rem 1.5rem 0">Latency Distribution</h2><div style="overflow-x:auto;padding:0 0 0.5rem"><table>
<thead><tr><th>Backend</th><th>Mode</th><th>P50 (ms)</th><th>P95 (ms)</th><th>P99 (ms)</th><th>Throughput</th><th>Peak Mem</th></tr></thead>
<tbody id="latencyTable"></tbody></table></div></div>
</div>
<footer><p><a href="index.html">frameforge</a></p></footer>
<script>
const results = {results_json};
const sysDiv = document.getElementById('sysinfo');
if (results.length > 0 && results[0].system_info) {{
  Object.entries(results[0].system_info).forEach(([k, v]) => {{
    if (!v) return;
    const s = document.createElement('span');
    s.textContent = k.replace(/_/g, ' ') + ': ' + v;
    sysDiv.appendChild(s);
  }});
}}
const ctx = document.getElementById('throughputChart').getContext('2d');
const colors = ['#e2a442','#d4944a','#3dbea0','#5b8fd9','#d95757'];
new Chart(ctx, {{
  type: 'bar',
  data: {{
    labels: results.map(r => r.backend + (r.seek_mode ? ' (' + r.seek_mode + ')' : '')),
    datasets: [{{
      label: 'Frames/sec',
      data: results.map(r => r.decode_throughput_fps),
      backgroundColor: results.map((_,i) => colors[i % colors.length]),
      borderRadius: 3, maxBarThickness: 60,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      y: {{ beginAtZero: true, grid: {{ color: '#2a2a2a' }}, ticks: {{ color: '#666', font: {{ family: 'JetBrains Mono', size: 11 }} }} }},
      x: {{ grid: {{ display: false }}, ticks: {{ color: '#666', font: {{ family: 'Space Grotesk', size: 12 }} }} }}
    }}
  }}
}});
const tbody = document.getElementById('latencyTable');
results.forEach(r => {{
  const tr = document.createElement('tr');
  tr.innerHTML = `<td style="color:var(--text-bright);font-family:'Space Grotesk',sans-serif;font-weight:600">${{r.backend}}</td>
    <td style="font-family:'Space Grotesk',sans-serif">${{r.seek_mode || '-'}}</td>
    <td>${{r.latency_p50_ms.toFixed(1)}}</td><td>${{r.latency_p95_ms.toFixed(1)}}</td><td>${{r.latency_p99_ms.toFixed(1)}}</td>
    <td style="color:var(--accent)">${{r.decode_throughput_fps.toFixed(0)}} fps</td><td>${{r.memory_peak_mb.toFixed(1)}} MB</td>`;
  tbody.appendChild(tr);
}});
</script>
</body>
</html>
"""


def generate_report(
    results: list[BenchmarkResult],
    output_path: Path | str = "docs/benchmarks.html",
) -> Path:
    """Generate a self-contained HTML benchmark report.

    Args:
        results: List of BenchmarkResult objects.
        output_path: Where to write the HTML file.

    Returns:
        Path to the generated HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_dicts = []
    for r in results:
        d = json.loads(r.to_json())
        results_dicts.append(d)

    html = _HTML_TEMPLATE.format(results_json=json.dumps(results_dicts, indent=2))
    output_path.write_text(html)

    return output_path
