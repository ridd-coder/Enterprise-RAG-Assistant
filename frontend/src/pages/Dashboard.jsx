// src/pages/Dashboard.jsx
import { useState, useEffect } from 'react';
import { api } from '../api';

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.metrics(), api.vectorDbHealth()])
      .then(([m, h]) => { setMetrics(m); setHealth(h); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const formatUptime = (secs) => {
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    return h > 0 ? `${h}h ${m}m` : `${m}m`;
  };

  return (
    <>
      <div className="page-header">
        <h2>Dashboard</h2>
        <p>System overview and real-time metrics</p>
      </div>
      <div className="page-body">
        {loading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: 60 }}>
            <div className="spinner" style={{ width: 40, height: 40 }} />
          </div>
        ) : (
          <>
            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-value">{metrics?.total_documents ?? '—'}</div>
                <div className="stat-label">📄 Documents Indexed</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{health?.vectors_count ?? '—'}</div>
                <div className="stat-label">🔢 Vector Embeddings</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{metrics?.total_queries ?? '—'}</div>
                <div className="stat-label">❓ Total Queries</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">
                  {metrics ? `${Math.round(metrics.avg_query_latency_ms)}ms` : '—'}
                </div>
                <div className="stat-label">⚡ Avg Response Time</div>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
              {/* System health */}
              <div className="card">
                <div className="card-title">🔍 System Health</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  {[
                    { label: 'API Server', status: 'healthy' },
                    { label: 'Qdrant Vector DB', status: health?.status || 'unknown' },
                    { label: 'OpenAI API', status: 'configured' },
                  ].map((item) => (
                    <div key={item.label} style={{
                      display: 'flex', justifyContent: 'space-between',
                      alignItems: 'center', padding: '10px 0',
                      borderBottom: '1px solid var(--border)'
                    }}>
                      <span style={{ fontSize: 14, color: 'var(--text-secondary)' }}>{item.label}</span>
                      <span className={`pill ${item.status === 'healthy' || item.status === 'configured' ? 'pill-green' : 'pill-yellow'}`}>
                        {item.status === 'healthy' || item.status === 'configured' ? '● ' : '○ '}
                        {item.status}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Stack info */}
              <div className="card">
                <div className="card-title">🛠 Technology Stack</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  {['FastAPI', 'OpenAI GPT-4o', 'LangChain', 'Qdrant', 'PyMuPDF',
                    'text-embedding-3-small', 'PostgreSQL', 'Docker'].map((t) => (
                    <span key={t} className="pill pill-purple">{t}</span>
                  ))}
                </div>
              </div>
            </div>

            {/* Architecture */}
            <div className="card" style={{ marginTop: 20 }}>
              <div className="card-title">⚙️ RAG Pipeline</div>
              <div style={{
                display: 'flex', alignItems: 'center',
                gap: 8, flexWrap: 'wrap', fontSize: 13
              }}>
                {['📄 PDF Upload', '→', '🔍 PyMuPDF Extract', '→',
                  '✂️ Chunk (1000c)', '→', '🔢 Embed (OpenAI)', '→',
                  '🗄 Qdrant Store', '→', '❓ User Question', '→',
                  '🔎 Similarity Search', '→', '🧠 GPT-4o Generate', '→',
                  '✅ Grounded Answer'].map((s, i) => (
                  <span key={i} style={{
                    color: s === '→' ? 'var(--text-muted)' : 'var(--text-secondary)',
                    fontWeight: s.startsWith('→') ? 400 : 500,
                  }}>{s}</span>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </>
  );
}
