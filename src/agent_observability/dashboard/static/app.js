/* agent-observability dashboard — vanilla JS + Chart.js */
'use strict';

// ---------------------------------------------------------------------------
// Tab navigation
// ---------------------------------------------------------------------------
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + target).classList.add('active');
    refreshActiveTab(target);
  });
});

function refreshActiveTab(tab) {
  if (tab === 'traces') loadTraces();
  else if (tab === 'spans') loadSpans();
  else if (tab === 'costs') loadCosts();
  else if (tab === 'latency') loadLatency();
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function fmtTime(ts) {
  if (!ts) return '-';
  return new Date(ts * 1000).toLocaleTimeString();
}
function fmtCost(usd) {
  return '$' + Number(usd).toFixed(6);
}
function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function showToast(msg) {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

async function apiFetch(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error('HTTP ' + res.status + ' from ' + path);
  return res.json();
}

function setLastRefresh() {
  document.getElementById('last-refresh').textContent =
    'Refreshed ' + new Date().toLocaleTimeString();
}

// ---------------------------------------------------------------------------
// Traces tab
// ---------------------------------------------------------------------------
async function loadTraces() {
  try {
    const agentFilter = document.getElementById('trace-agent-filter').value.trim();
    let url = '/api/traces?limit=200';
    if (agentFilter) url += '&agent_id=' + encodeURIComponent(agentFilter);

    const [tracesData, costsData, latencyData, healthData] = await Promise.all([
      apiFetch(url),
      apiFetch('/api/costs'),
      apiFetch('/api/latency'),
      apiFetch('/health'),
    ]);

    // Stats row
    document.getElementById('stat-traces').textContent = healthData.traces || tracesData.count;
    document.getElementById('stat-spans').textContent = healthData.spans || '-';
    document.getElementById('stat-cost').textContent = costsData.total_usd != null
      ? costsData.total_usd.toFixed(4) : '0.0000';
    document.getElementById('stat-p95').textContent = latencyData.p95 || '0';

    // Table
    const tbody = document.getElementById('traces-tbody');
    if (!tracesData.traces || tracesData.traces.length === 0) {
      tbody.innerHTML = '<tr><td colspan="8" class="empty-state">No traces recorded yet.</td></tr>';
      setLastRefresh();
      return;
    }
    tbody.innerHTML = tracesData.traces.slice().reverse().map(t => `
      <tr>
        <td class="mono">${escHtml(String(t.trace_id || '-').slice(0, 12))}...</td>
        <td>${escHtml(t.agent_id || '-')}</td>
        <td>${escHtml(t.provider || '-')}</td>
        <td>${escHtml(t.model || '-')}</td>
        <td>${escHtml(t.input_tokens || 0)}</td>
        <td>${escHtml(t.output_tokens || 0)}</td>
        <td class="mono">${fmtCost(t.cost_usd || 0)}</td>
        <td>${fmtTime(t.timestamp)}</td>
      </tr>
    `).join('');

    setLastRefresh();
  } catch (err) {
    showToast('Error loading traces: ' + err.message);
  }
}

// ---------------------------------------------------------------------------
// Spans tab
// ---------------------------------------------------------------------------
async function loadSpans() {
  try {
    const traceFilter = document.getElementById('span-trace-filter').value.trim();
    let url = '/api/spans?limit=500';
    if (traceFilter) url += '&trace_id=' + encodeURIComponent(traceFilter);

    const data = await apiFetch(url);
    const container = document.getElementById('span-tree-container');

    if (!data.spans || data.spans.length === 0) {
      container.innerHTML = '<div class="empty-state"><div class="icon">&#128203;</div>No spans found.</div>';
      setLastRefresh();
      return;
    }

    // Group by trace_id
    const byTrace = {};
    data.spans.forEach(span => {
      const traceId = span.trace_id || 'unknown';
      if (!byTrace[traceId]) byTrace[traceId] = [];
      byTrace[traceId].push(span);
    });

    // Compute max duration for bar widths
    const allDurations = data.spans.map(s => Number(s.duration_ms) || 0);
    const maxDuration = Math.max(...allDurations, 1);

    let html = '<div class="span-tree">';
    Object.entries(byTrace).forEach(([traceId, spans]) => {
      html += `<div style="margin-bottom:20px">
        <div style="font-size:12px;color:var(--text-muted);margin-bottom:8px">
          Trace: <span class="mono">${escHtml(traceId)}</span>
          <span style="margin-left:8px">(${spans.length} spans)</span>
        </div>`;

      // Build parent-child tree
      const spanMap = {};
      spans.forEach(s => { spanMap[s.span_id || s.name] = s; });

      const roots = spans.filter(s => !s.parent_span_id || !spanMap[s.parent_span_id]);
      const children = {};
      spans.forEach(s => {
        if (s.parent_span_id && spanMap[s.parent_span_id]) {
          if (!children[s.parent_span_id]) children[s.parent_span_id] = [];
          children[s.parent_span_id].push(s);
        }
      });

      function renderSpan(span, depth) {
        const dur = Number(span.duration_ms) || 0;
        const barWidth = Math.max(4, Math.round((dur / maxDuration) * 200));
        const name = span.name || span.operation || 'span';
        let nodeHtml = `<div class="span-node" style="padding-left:${8 + depth * 20}px">
          <div class="span-bar" style="width:${barWidth}px"></div>
          <span class="span-name">${escHtml(name)}</span>
          <span class="span-duration">${dur}ms</span>
        </div>`;
        const spanId = span.span_id || span.name;
        if (children[spanId]) {
          children[spanId].forEach(child => {
            nodeHtml += renderSpan(child, depth + 1);
          });
        }
        return nodeHtml;
      }

      roots.forEach(root => { html += renderSpan(root, 0); });
      html += '</div>';
    });
    html += '</div>';
    container.innerHTML = html;
    setLastRefresh();
  } catch (err) {
    showToast('Error loading spans: ' + err.message);
  }
}

// ---------------------------------------------------------------------------
// Cost breakdown tab
// ---------------------------------------------------------------------------
let costPieChart = null;

async function loadCosts() {
  try {
    const data = await apiFetch('/api/costs');
    const byModel = data.by_model || {};
    const total = data.total_usd || 0;

    // Pie chart
    const labels = Object.keys(byModel);
    const values = Object.values(byModel);
    const colors = [
      '#6366f1','#22c55e','#f59e0b','#ef4444','#3b82f6',
      '#8b5cf6','#ec4899','#14b8a6','#f97316','#64748b',
    ];

    const ctx = document.getElementById('cost-pie-chart').getContext('2d');
    if (costPieChart) costPieChart.destroy();
    costPieChart = new Chart(ctx, {
      type: 'pie',
      data: {
        labels: labels.length ? labels : ['No data'],
        datasets: [{
          data: values.length ? values : [1],
          backgroundColor: colors.slice(0, Math.max(labels.length, 1)),
          borderColor: getComputedStyle(document.documentElement).getPropertyValue('--surface').trim(),
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: getComputedStyle(document.documentElement).getPropertyValue('--text').trim() },
          },
        },
      },
    });

    // Table
    const tbody = document.getElementById('cost-tbody');
    if (labels.length === 0) {
      tbody.innerHTML = '<tr><td colspan="3" class="empty-state">No cost records yet.</td></tr>';
    } else {
      tbody.innerHTML = labels.map((model, i) => {
        const share = total > 0 ? ((values[i] / total) * 100).toFixed(1) + '%' : '-';
        return `<tr>
          <td>${escHtml(model)}</td>
          <td class="mono">${fmtCost(values[i])}</td>
          <td>${share}</td>
        </tr>`;
      }).join('');
    }
    document.getElementById('cost-total').textContent = fmtCost(total);
    setLastRefresh();
  } catch (err) {
    showToast('Error loading costs: ' + err.message);
  }
}

// ---------------------------------------------------------------------------
// Latency histogram tab
// ---------------------------------------------------------------------------
let latencyChart = null;

async function loadLatency() {
  try {
    const data = await apiFetch('/api/latency');

    const ctx = document.getElementById('latency-histogram').getContext('2d');
    if (latencyChart) latencyChart.destroy();

    const textColor = getComputedStyle(document.documentElement).getPropertyValue('--text').trim();
    const mutedColor = getComputedStyle(document.documentElement).getPropertyValue('--text-muted').trim();

    latencyChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: (data.buckets || []).map(b => b + 'ms'),
        datasets: [{
          label: 'Span Count',
          data: data.counts || [],
          backgroundColor: 'rgba(99,102,241,0.7)',
          borderColor: '#6366f1',
          borderWidth: 1,
          borderRadius: 4,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: textColor } },
        },
        scales: {
          x: {
            ticks: { color: mutedColor },
            grid: { color: 'rgba(255,255,255,0.05)' },
            title: { display: true, text: 'Latency Bucket (ms)', color: mutedColor },
          },
          y: {
            ticks: { color: mutedColor },
            grid: { color: 'rgba(255,255,255,0.05)' },
            title: { display: true, text: 'Span Count', color: mutedColor },
          },
        },
      },
    });

    // Stats table
    const tbody = document.getElementById('latency-stats-tbody');
    if (!data.buckets || data.buckets.length === 0) {
      tbody.innerHTML = '<tr><td colspan="2" class="empty-state">No span duration data yet.</td></tr>';
    } else {
      tbody.innerHTML = [
        ['Min', data.min_ms + ' ms'],
        ['Max', data.max_ms + ' ms'],
        ['P50 (median)', data.p50 + ' ms'],
        ['P95', data.p95 + ' ms'],
        ['Buckets', (data.buckets || []).length],
      ].map(([k, v]) => `<tr><td>${k}</td><td class="mono">${v}</td></tr>`).join('');
    }
    setLastRefresh();
  } catch (err) {
    showToast('Error loading latency: ' + err.message);
  }
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
loadTraces();
setInterval(() => {
  const activeTab = document.querySelector('.tab-btn.active');
  if (activeTab) refreshActiveTab(activeTab.dataset.tab);
}, 30000);
