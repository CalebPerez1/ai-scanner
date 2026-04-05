import React, { useState, useEffect, useCallback } from 'react';

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
            stroke="#f2f2f7"
            strokeWidth={strokeWidth}
          />
          <text x={cx} y={cy + 1} textAnchor="middle" dominantBaseline="middle"
            fontSize="13" fill="#aeaeb2" fontFamily="-apple-system, sans-serif">
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
          stroke="#f2f2f7" strokeWidth={strokeWidth} />
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
          fontSize="26" fontWeight="700" fill="#1c1c1e"
          fontFamily="-apple-system, sans-serif">
          {total}
        </text>
        <text x={cx} y={cy + 14} textAnchor="middle" dominantBaseline="middle"
          fontSize="11" fill="#6e6e73" fontFamily="-apple-system, sans-serif">
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
          <span className="search-icon">⌕</span>
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
          <div className="empty-icon">🔍</div>
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
          <div className="nav-logo">🛡️</div>
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
                  <>⌕ &nbsp;Run scan</>
                )}
              </button>
              {result && !scanning && (
                <span style={{ fontSize: 13, color: '#34c759' }}>
                  ✓ Scan complete — {result.scan_date?.slice(0, 10)}
                </span>
              )}
            </div>

            {scanError && <div className="scan-error">⚠️ {scanError}</div>}
          </form>
        </div>

        {/* ── Results (only after a scan) ── */}
        {result && (
          <>
            {/* Stat cards */}
            <p className="section-title">{result.project_name} · Security overview</p>
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
                  <div className="empty-icon">✅</div>
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
              <div className="empty-icon">🛡️</div>
              <p>Ready to scan</p>
              <span>Enter a project path above and click Run scan to get started.</span>
            </div>
          </div>
        )}

      </main>
    </>
  );
}
