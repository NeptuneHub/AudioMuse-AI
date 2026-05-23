/* timeline.js — Listening History Timeline chart and table logic */
(function () {
    'use strict';

    let chartInstance = null;
    let currentData = null;

    const limitSelect = document.getElementById('limit-select');
    const metricSelect = document.getElementById('chart-metric');
    const loadingEl = document.getElementById('timeline-loading');
    const errorEl = document.getElementById('timeline-error');
    const summaryEl = document.getElementById('timeline-summary');
    const tableSection = document.getElementById('timeline-table-section');
    const tbody = document.getElementById('timeline-tbody');

    function showError(msg) {
        loadingEl.style.display = 'none';
        errorEl.textContent = msg;
        errorEl.style.display = 'block';
    }

    function formatDate(dateStr) {
        if (!dateStr) return '—';
        try {
            const d = new Date(dateStr);
            if (isNaN(d.getTime())) return '—';
            return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
        } catch (e) {
            return '—';
        }
    }

    function getEnergyColor(energy) {
        if (energy == null) return '#888';
        if (energy < 0.33) return '#4ade80';
        if (energy < 0.66) return '#f59e0b';
        return '#ef4444';
    }

    function renderSummary(summary) {
        if (!summary) return;
        summaryEl.innerHTML = `
            <div class="summary-card"><div class="value">${summary.total_tracks || 0}</div><div class="label">Tracks</div></div>
            <div class="summary-card"><div class="value">${summary.total_plays || 0}</div><div class="label">Total Plays</div></div>
            <div class="summary-card"><div class="value">${summary.avg_energy != null ? summary.avg_energy : '—'}</div><div class="label">Avg Energy</div></div>
            <div class="summary-card"><div class="value">${summary.avg_tempo != null ? summary.avg_tempo : '—'}</div><div class="label">Avg BPM</div></div>
        `;
        summaryEl.style.display = '';
    }

    function renderTable(entries) {
        tbody.innerHTML = '';
        entries.forEach(function (e, i) {
            const energyWidth = e.energy != null ? Math.round(e.energy * 60) : 0;
            const energyHtml = e.energy != null
                ? `<span class="energy-bar" style="width:${energyWidth}px; background-color:${getEnergyColor(e.energy)};"></span> ${e.energy.toFixed(2)}`
                : '—';
            const tempoStr = e.tempo != null ? Math.round(e.tempo) + ' BPM' : '—';
            const keyStr = e.key ? (e.key + (e.scale ? ' ' + e.scale : '')) : '—';

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${i + 1}</td>
                <td>${escapeHtml(e.title)}</td>
                <td>${escapeHtml(e.artist)}</td>
                <td>${escapeHtml(e.album || '')}</td>
                <td><span class="play-badge">${e.play_count}</span></td>
                <td>${energyHtml}</td>
                <td>${tempoStr}</td>
                <td>${keyStr}</td>
                <td>${formatDate(e.last_played)}</td>
            `;
            tbody.appendChild(tr);
        });
        tableSection.style.display = '';
    }

    function renderChart(entries, metric) {
        const canvas = document.getElementById('timeline-chart');
        if (!canvas) return;

        const labels = entries.map(function (e, i) {
            return e.title.length > 20 ? e.title.substring(0, 18) + '…' : e.title;
        });
        const data = entries.map(function (e) {
            if (metric === 'energy') return e.energy;
            if (metric === 'tempo') return e.tempo;
            return e.play_count;
        });

        const colors = entries.map(function (e) {
            if (metric === 'energy') return getEnergyColor(e.energy);
            if (metric === 'tempo') return '#6366f1';
            return '#6366f1';
        });

        if (chartInstance) {
            chartInstance.destroy();
        }

        const metricLabels = { play_count: 'Play Count', energy: 'Energy', tempo: 'Tempo (BPM)' };

        chartInstance = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: metricLabels[metric] || metric,
                    data: data,
                    backgroundColor: colors,
                    borderRadius: 4,
                    borderSkipped: false,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            title: function (ctx) {
                                var idx = ctx[0].dataIndex;
                                return entries[idx].title + ' — ' + entries[idx].artist;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { maxRotation: 45, font: { size: 10 } },
                        grid: { display: false }
                    },
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(128,128,128,0.1)' }
                    }
                }
            }
        });
    }

    function fetchData() {
        var limit = limitSelect.value || 50;
        loadingEl.style.display = '';
        errorEl.style.display = 'none';
        tableSection.style.display = 'none';
        summaryEl.style.display = 'none';

        fetch('/timeline/data?limit=' + limit)
            .then(function (res) {
                if (!res.ok) throw new Error('Server error: ' + res.status);
                return res.json();
            })
            .then(function (json) {
                if (json.error) throw new Error(json.error);
                loadingEl.style.display = 'none';
                currentData = json;
                renderSummary(json.summary);
                renderTable(json.entries || []);
                renderChart(json.entries || [], metricSelect.value);
            })
            .catch(function (err) {
                showError('Failed to load timeline: ' + err.message);
            });
    }

    function escapeHtml(str) {
        if (!str) return '';
        var div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // Event listeners
    limitSelect.addEventListener('change', fetchData);
    metricSelect.addEventListener('change', function () {
        if (currentData && currentData.entries) {
            renderChart(currentData.entries, metricSelect.value);
        }
    });

    // Initial load
    fetchData();
})();
