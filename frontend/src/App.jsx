import React, { useState, useEffect, useCallback, useRef } from 'react';

// ─── Constants ──────────────────────────────────────────────────────────────

const SEV_ORDER = ['critical', 'high', 'medium', 'low'];

const SEV_META = {
  critical: { label: 'Critical', color: '#ff3b30', cssClass: 'critical' },
  high:     { label: 'High',     color: '#ff9500', cssClass: 'high'     },
  medium:   { label: 'Medium',   color: '#007aff', cssClass: 'medium'   },
  low:      { label: 'Low',      color: '#8e8e93', cssClass: 'low'      },
};

// ─── Donut Chart ─────────────────────────────────────────────────────────────

function DonutChart({ findings }) {
  const size = 140;
  const strokeWidth = 18;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const cx = size / 2;
  const cy = size / 2;

  const counts = SEV_ORDER.map(s => ({
    sev: s,
    count: findings.filter(f => f.severity === s).length,
  }));
  const total = counts.reduce((a, b) => a + b.count, 0);

  if (total === 0) {
    return (
      <div className="donut-wrap">
        <svg width={size} height={size} className="donut-svg">
          <circle
            cx={cx} cy={cy} r={radius}
            fill="none"
            stroke="#1a2236"
            strokeWidth={strokeWidth}
          />
          <text x={cx} y={cy + 1} textAnchor="middle" dominantBaseline="middle"
            fontSize="13" fill="#4a5568" fontFamily="-apple-system, sans-serif">
            No data
          </text>
        </svg>
      </div>
    );
  }

  let offset = 0;
  const segments = counts.map(({ sev, count }) => {
    const pct = count / total;
    const dash = circumference * pct;
    const gap  = circumference - dash;
    // rotate so segments start at 12 o'clock
    const rotation = -90 + (offset / total) * 360;
    offset += count;
    return { sev, count, dash, gap, rotation };
  }).filter(s => s.count > 0);

  return (
    <div className="donut-wrap">
      <svg width={size} height={size} className="donut-svg">
        {/* background track */}
        <circle cx={cx} cy={cy} r={radius} fill="none"
          stroke="#1a2236" strokeWidth={strokeWidth} />
        {segments.map(({ sev, dash, gap, rotation }) => (
          <circle
            key={sev}
            cx={cx} cy={cy} r={radius}
            fill="none"
            stroke={SEV_META[sev].color}
            strokeWidth={strokeWidth}
            strokeDasharray={`${dash} ${gap}`}
            strokeLinecap="round"
            transform={`rotate(${rotation} ${cx} ${cy})`}
            style={{ transition: 'stroke-dasharray 0.5s ease' }}
          />
        ))}
        {/* centre label */}
        <text x={cx} y={cy - 8} textAnchor="middle" dominantBaseline="middle"
          fontSize="26" fontWeight="700" fill="#e8eaf0"
          fontFamily="-apple-system, sans-serif">
          {total}
        </text>
        <text x={cx} y={cy + 14} textAnchor="middle" dominantBaseline="middle"
          fontSize="11" fill="#8892a4" fontFamily="-apple-system, sans-serif">
          findings
        </text>
      </svg>

      <div className="donut-legend">
        {counts.map(({ sev, count }) => (
          <div key={sev} className="legend-item">
            <div className="legend-left">
              <div className="legend-dot" style={{ background: SEV_META[sev].color }} />
              <span className="legend-name">{SEV_META[sev].label}</span>
            </div>
            <span className="legend-count">{count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Bar Chart ───────────────────────────────────────────────────────────────

function ScannerBarChart({ findings }) {
  const scannerCounts = findings.reduce((acc, f) => {
    acc[f.scanner_name] = (acc[f.scanner_name] || 0) + 1;
    return acc;
  }, {});

  const entries = Object.entries(scannerCounts)
    .sort((a, b) => b[1] - a[1]);

  const max = entries[0]?.[1] || 1;

  // Friendly display names
  const displayName = name => name
    .replace('_auditor', '')
    .replace('_scanner', '')
    .replace('_analyzer', '')
    .replace('_tester', '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());

  if (entries.length === 0) {
    return <p style={{ fontSize: 13, color: '#aeaeb2' }}>No data</p>;
  }

  return (
    <div className="bar-list">
      {entries.map(([name, count]) => (
        <div key={name} className="bar-item">
          <div className="bar-meta">
            <span className="bar-name">{displayName(name)}</span>
            <span className="bar-count">{count}</span>
          </div>
          <div className="bar-track">
            <div
              className="bar-fill"
              style={{ width: `${(count / max) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── Recommendations ──────────────────────────────────────────────────────────

function Recommendations({ findings }) {
  // Group by recommendation text, rank by (impact × count)
  const sevWeight = { critical: 4, high: 3, medium: 2, low: 1 };

  const recMap = {};
  findings.forEach(f => {
    if (!f.recommendation) return;
    const key = f.recommendation;
    if (!recMap[key]) recMap[key] = { fix: key, findings: [] };
    recMap[key].findings.push(f);
  });

  const ranked = Object.values(recMap)
    .map(r => ({
      ...r,
      score: r.findings.reduce((s, f) => s + (sevWeight[f.severity] || 1), 0),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);

  if (ranked.length === 0) return null;

  const badgeClass = findings => {
    const sevs = findings.map(f => f.severity);
    if (sevs.includes('critical')) return { cls: 'badge badge-critical', label: 'Critical' };
    if (sevs.includes('high'))     return { cls: 'badge badge-high',     label: 'High'     };
    if (sevs.includes('medium'))   return { cls: 'badge badge-medium',   label: 'Medium'   };
    return                               { cls: 'badge badge-low',       label: 'Low'      };
  };

  return (
    <div className="card recs-card">
      <h3>Top Recommendations</h3>
      <div className="rec-list">
        {ranked.map((r, i) => {
          const badge = badgeClass(r.findings);
          return (
            <div key={i} className="rec-item">
              <div className="rec-rank">{i + 1}</div>
              <div className="rec-body">
                <div className="rec-fix">{r.fix}</div>
                <div className="rec-meta">
                  Affects {r.findings.length} finding{r.findings.length !== 1 ? 's' : ''} ·{' '}
                  {[...new Set(r.findings.map(f => f.scanner_name))].join(', ')}
                </div>
              </div>
              <span className={badge.cls}>{badge.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Findings Table ───────────────────────────────────────────────────────────

function FindingsTable({ findings }) {
  const [search, setSearch] = useState('');
  const [sevFilter, setSevFilter] = useState('all');
  const [expandedIdx, setExpandedIdx] = useState(null);

  const filtered = findings.filter(f => {
    const matchSev = sevFilter === 'all' || f.severity === sevFilter;
    const q = search.toLowerCase();
    const matchSearch = !q || [
      f.title, f.scanner_name, f.file_path || '', f.description,
    ].some(s => s.toLowerCase().includes(q));
    return matchSev && matchSearch;
  });

  const toggle = idx => setExpandedIdx(prev => prev === idx ? null : idx);

  return (
    <div className="card findings-card">
      <h3>Findings</h3>

      <div className="findings-toolbar">
        <div className="search-wrap">
          <span className="search-icon">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
              <circle cx="5.5" cy="5.5" r="3.5" />
              <line x1="8.5" y1="8.5" x2="12.5" y2="12.5" />
            </svg>
          </span>
          <input
            className="search-input"
            placeholder="Search findings…"
            value={search}
            onChange={e => { setSearch(e.target.value); setExpandedIdx(null); }}
          />
        </div>
        <div className="filter-pills">
          {['all', ...SEV_ORDER].map(s => {
            const isActive = sevFilter === s;
            const activeClass = isActive ? `active-${s}` : '';
            return (
              <button
                key={s}
                className={`pill ${activeClass}`}
                onClick={() => { setSevFilter(s); setExpandedIdx(null); }}
              >
                {s === 'all' ? 'All' : SEV_META[s].label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="findings-count">
        {filtered.length} of {findings.length} finding{findings.length !== 1 ? 's' : ''}
      </div>

      {filtered.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">
            <svg width="40" height="40" viewBox="0 0 40 40" fill="none" stroke="#8892a4" strokeWidth="1.8" strokeLinecap="round">
              <circle cx="17" cy="17" r="10" />
              <line x1="24.5" y1="24.5" x2="35" y2="35" />
            </svg>
          </div>
          <p>No findings match your filters</p>
          <span>Try adjusting your search or severity filter</span>
        </div>
      ) : (
        <table className="findings-table">
          <thead>
            <tr>
              <th style={{ width: 28 }}></th>
              <th>Severity</th>
              <th>Title</th>
              <th>Scanner</th>
              <th>Location</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((f, idx) => {
              const isOpen = expandedIdx === idx;
              const loc = f.file_path
                ? `${f.file_path}${f.line_number ? `:${f.line_number}` : ''}`
                : '—';
              return (
                <React.Fragment key={idx}>
                  <tr
                    className={`finding-row${isOpen ? ' expanded' : ''}`}
                    data-severity={f.severity}
                    onClick={() => toggle(idx)}
                  >
                    <td><span className={`chevron${isOpen ? ' open' : ''}`}>›</span></td>
                    <td>
                      <span className={`badge badge-${f.severity}`}>
                        {SEV_META[f.severity]?.label || f.severity}
                      </span>
                    </td>
                    <td style={{ fontWeight: 500 }}>{f.title}</td>
                    <td><span className="tag">{f.scanner_name}</span></td>
                    <td><span className="filepath" title={loc}>{loc}</span></td>
                  </tr>
                  {isOpen && (
                    <tr className="finding-expand-row">
                      <td colSpan={5}>
                        <div className="expand-content">
                          <div className="expand-section">
                            <span className="expand-label">Description</span>
                            <span className="expand-text">{f.description}</span>
                          </div>
                          {f.recommendation && (
                            <div className="expand-section">
                              <span className="expand-label">Recommendation</span>
                              <span className="expand-fix">{f.recommendation}</span>
                            </div>
                          )}
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}

// ─── Export helpers ───────────────────────────────────────────────────────────

function triggerDownload(content, filename, mime) {
  const blob = new Blob([content], { type: mime });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function buildMarkdown(result) {
  const date = result.scan_date?.slice(0, 10) ?? 'unknown';
  const slug = (result.project_name ?? 'report').replace(/\s+/g, '-').toLowerCase();

  const lines = [
    `# AI-Scan Security Report — ${result.project_name}`,
    '',
    `**Scan date:** ${date}  `,
    `**Total findings:** ${result.total_findings}`,
    '',
    '## Severity Summary',
    '',
    '| Severity | Count |',
    '|----------|------:|',
    ...SEV_ORDER.map(s =>
      `| ${SEV_META[s].label} | ${result.by_severity?.[s] ?? 0} |`
    ),
    '',
    '## Findings',
    '',
    '| # | Severity | Title | Scanner | Location |',
    '|---|----------|-------|---------|----------|',
  ];

  const sorted = [...result.findings].sort(
    (a, b) => SEV_ORDER.indexOf(a.severity) - SEV_ORDER.indexOf(b.severity)
  );

  sorted.forEach((f, i) => {
    const loc = f.file_path
      ? `${f.file_path}${f.line_number ? `:${f.line_number}` : ''}`
      : '—';
    const title = f.title.replace(/\|/g, '\\|');
    lines.push(`| ${i + 1} | ${SEV_META[f.severity]?.label ?? f.severity} | ${title} | ${f.scanner_name} | ${loc} |`);
  });

  lines.push('', '## Finding Details', '');

  sorted.forEach((f, i) => {
    const loc = f.file_path
      ? `${f.file_path}${f.line_number ? `:${f.line_number}` : ''}`
      : null;
    lines.push(`### ${i + 1}. ${f.title}`, '');
    lines.push(`- **Severity:** ${SEV_META[f.severity]?.label ?? f.severity}`);
    lines.push(`- **Scanner:** ${f.scanner_name}`);
    if (loc) lines.push(`- **Location:** \`${loc}\``);
    lines.push('', f.description, '');
    if (f.recommendation) {
      lines.push(`> **Recommendation:** ${f.recommendation}`, '');
    }
  });

  return { md: lines.join('\n'), slug };
}

// ─── Export Menu ──────────────────────────────────────────────────────────────

function ExportMenu({ result }) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    if (!open) return;
    const handler = e => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  const exportJSON = () => {
    const slug = (result.project_name ?? 'report').replace(/\s+/g, '-').toLowerCase();
    triggerDownload(JSON.stringify(result, null, 2), `aiscan-${slug}.json`, 'application/json');
    setOpen(false);
  };

  const exportMarkdown = () => {
    const { md, slug } = buildMarkdown(result);
    triggerDownload(md, `aiscan-${slug}.md`, 'text/markdown');
    setOpen(false);
  };

  const exportPDF = () => {
    setOpen(false);
    // Small delay so the dropdown closes before the print dialog opens
    setTimeout(() => window.print(), 80);
  };

  return (
    <div className="export-wrap" ref={wrapRef}>
      <button className="btn-export" onClick={() => setOpen(o => !o)}>
        Export
        <span className={`export-chevron${open ? ' open' : ''}`}>›</span>
      </button>
      {open && (
        <div className="export-dropdown">
          <button className="export-item" onClick={exportJSON}>
            <span className="export-icon">{ }</span>
            <span>JSON</span>
            <span className="export-hint">.json</span>
          </button>
          <button className="export-item" onClick={exportMarkdown}>
            <span className="export-icon">#</span>
            <span>Markdown</span>
            <span className="export-hint">.md</span>
          </button>
          <div className="export-divider" />
          <button className="export-item" onClick={exportPDF}>
            <span className="export-icon">⎙</span>
            <span>PDF</span>
            <span className="export-hint">print</span>
          </button>
        </div>
      )}
    </div>
  );
}

// ─── App ─────────────────────────────────────────────────────────────────────

export default function App() {
  // Health
  const [health, setHealth] = useState('checking'); // 'checking' | 'healthy' | 'unhealthy'

  // Form
  const [projectPath, setProjectPath] = useState('');
  const [llmEndpoint, setLlmEndpoint] = useState('');
  const [projectName, setProjectName]  = useState('');

  // Scan state
  const [scanning, setScanning]   = useState(false);
  const [result, setResult]       = useState(null);   // ScanResult | null
  const [scanError, setScanError] = useState('');

  // Ping health endpoint on mount and every 30 s
  const checkHealth = useCallback(() => {
    fetch('/api/health')
      .then(r => r.ok ? setHealth('healthy') : setHealth('unhealthy'))
      .catch(() => setHealth('unhealthy'));
  }, []);

  useEffect(() => {
    checkHealth();
    const id = setInterval(checkHealth, 30_000);
    return () => clearInterval(id);
  }, [checkHealth]);

  const handleScan = async e => {
    e.preventDefault();
    if (!projectPath.trim()) return;

    setScanning(true);
    setScanError('');
    setResult(null);

    try {
      const body = { project_path: projectPath.trim() };
      if (llmEndpoint.trim()) body.llm_endpoint_url = llmEndpoint.trim();
      if (projectName.trim()) body.project_name     = projectName.trim();

      const res = await fetch('/api/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${res.status}`);
      }

      setResult(await res.json());
    } catch (err) {
      setScanError(err.message || 'Scan failed. Is the backend running?');
    } finally {
      setScanning(false);
    }
  };

  const findings = result?.findings ?? [];

  const countBySev = sev => result?.by_severity?.[sev] ?? 0;

  return (
    <>
      {/* ── Nav ── */}
      <nav className="nav">
        <div className="nav-brand">
          <div className="nav-title">AI<span>-Scan</span></div>
        </div>
        <div className="health-badge">
          <div className={`health-dot ${health === 'checking' ? '' : health}`} />
          {health === 'checking' ? 'Connecting…'
           : health === 'healthy' ? 'API healthy'
           : 'API unreachable'}
        </div>
      </nav>

      <main className="page">

        {/* ── Scan form ── */}
        <div className="card scan-card">
          <h2>Scan a project</h2>
          <p className="subtitle">
            Point AI-Scan at a local directory or a GitHub / GitLab repository URL.
          </p>

          <form onSubmit={handleScan}>
            <div className="form-grid">
              <div className="form-group full">
                <label>Project path or repository URL *</label>
                <input
                  value={projectPath}
                  onChange={e => setProjectPath(e.target.value)}
                  placeholder="./my-project  or  https://github.com/org/repo"
                  required
                />
              </div>
              <div className="form-group">
                <label>Project name (optional)</label>
                <input
                  value={projectName}
                  onChange={e => setProjectName(e.target.value)}
                  placeholder="My App"
                />
              </div>
              <div className="form-group">
                <label>LLM endpoint for injection testing (optional)</label>
                <input
                  value={llmEndpoint}
                  onChange={e => setLlmEndpoint(e.target.value)}
                  placeholder="http://localhost:8080/v1/chat/completions"
                />
              </div>
            </div>

            <div className="scan-actions">
              <button className="btn-scan" type="submit" disabled={scanning || !projectPath.trim()}>
                {scanning ? (
                  <><div className="spinner" />Scanning…</>
                ) : (
                  <>
                    Run scan
                    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" style={{ marginLeft: 6 }}>
                      <circle cx="5.5" cy="5.5" r="3.5" />
                      <line x1="8.5" y1="8.5" x2="12.5" y2="12.5" />
                    </svg>
                  </>
                )}
              </button>
              {result && !scanning && (
                <span style={{ fontSize: 13, color: '#30d158' }}>
                  ✓ Scan complete — {result.scan_date?.slice(0, 10)}
                </span>
              )}
            </div>

            {scanError && (
              <div className="scan-error">
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0, marginRight: 6 }}>
                  <path d="M7 1.5 L12.5 11.5 H1.5 Z" />
                  <line x1="7" y1="5.5" x2="7" y2="8" />
                  <circle cx="7" cy="9.8" r="0.5" fill="currentColor" stroke="none" />
                </svg>
                {scanError}
              </div>
            )}
          </form>
        </div>

        {/* ── Results (only after a scan) ── */}
        {result && (
          <>
            {/* Results header with export */}
            <div className="results-header">
              <p className="section-title">{result.project_name} · Security overview</p>
              <ExportMenu result={result} />
            </div>
            <div className="stats-grid">
              {SEV_ORDER.map(sev => (
                <div key={sev} className={`card stat-card ${SEV_META[sev].cssClass}`}>
                  <div className="stat-accent" />
                  <div className="stat-body">
                    <div className="stat-label">{SEV_META[sev].label}</div>
                    <div className="stat-value">{countBySev(sev)}</div>
                    <div className="stat-sub">
                      {countBySev(sev) === 1 ? 'finding' : 'findings'}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Charts */}
            {findings.length > 0 && (
              <div className="charts-row">
                <div className="card chart-card">
                  <h3>Severity breakdown</h3>
                  <DonutChart findings={findings} />
                </div>
                <div className="card chart-card">
                  <h3>Findings by scanner</h3>
                  <ScannerBarChart findings={findings} />
                </div>
              </div>
            )}

            {/* Recommendations */}
            {findings.length > 0 && <Recommendations findings={findings} />}

            {/* Findings table */}
            {findings.length > 0 ? (
              <FindingsTable findings={findings} />
            ) : (
              <div className="card">
                <div className="empty-state">
                  <div className="empty-icon">
                    <svg width="52" height="52" viewBox="0 0 52 52" fill="none">
                      <circle cx="26" cy="26" r="22" stroke="#4ade80" strokeWidth="2" />
                      <path d="M16 26 L22 32 L36 18" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </div>
                  <p>No findings — clean scan!</p>
                  <span>AI-Scan found no security issues in this project.</span>
                </div>
              </div>
            )}
          </>
        )}

        {/* Zero-state before first scan */}
        {!result && !scanning && (
          <div className="card">
            <div className="empty-state">
              <div className="empty-icon">
                <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                  <path d="M24 4 L40 11 V24 C40 34 24 44 24 44 C24 44 8 34 8 24 V11 Z" stroke="#00d4ff" strokeWidth="1.8" strokeLinejoin="round" />
                </svg>
              </div>
              <p>Ready to scan</p>
              <span>Enter a project path above and click Run scan to get started.</span>
            </div>
          </div>
        )}

      </main>
    </>
  );
}
