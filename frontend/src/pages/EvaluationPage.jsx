// src/pages/EvaluationPage.jsx

const MOCK_METRICS = [
  { label: 'Retrieval Hit Rate', value: 0, description: 'Run evaluation to see real results' },
  { label: 'Answer Correctness', value: 0, description: 'Keyword overlap with expected answers' },
  { label: 'Citation Accuracy', value: 0, description: 'Correct source document cited' },
  { label: 'Average Latency', value: null, unit: 'ms', description: '' },
];

export default function EvaluationPage() {
  return (
    <>
      <div className="page-header">
        <h2>📊 Evaluation Dashboard</h2>
        <p>RAG quality metrics — run the evaluation suite for real results</p>
      </div>
      <div className="page-body">

        <div className="alert alert-success" style={{ marginBottom: 24 }}>
          ℹ️ Run <code style={{ background: 'rgba(255,255,255,0.1)', padding: '2px 6px', borderRadius: 4 }}>python scripts/evaluate.py</code> to populate this dashboard with actual metrics.
          Real numbers only — no fabricated results.
        </div>

        {/* Metrics */}
        <div className="card" style={{ marginBottom: 24 }}>
          <div className="card-title">📈 Quality Metrics</div>
          {MOCK_METRICS.map((m) => (
            <div key={m.label} className="eval-metric">
              <div>
                <div className="eval-metric-label">{m.label}</div>
                {m.description && (
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>{m.description}</div>
                )}
              </div>
              <div className="eval-metric-bar">
                <div className="eval-metric-fill" style={{ width: `${m.value * 100}%` }} />
              </div>
              <div className="eval-metric-value">
                {m.unit === 'ms' ? '—' : m.value === 0 ? '—' : `${(m.value * 100).toFixed(0)}%`}
              </div>
            </div>
          ))}
        </div>

        {/* Eval steps */}
        <div className="card">
          <div className="card-title">🧪 How to Run Evaluation</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {[
              { step: '1', title: 'Upload your documents', desc: 'Go to Documents and upload PDF files for your knowledge base.' },
              { step: '2', title: 'Edit evaluation_dataset.json', desc: 'Add your Q&A pairs with expected answers and source filenames.' },
              { step: '3', title: 'Run the evaluator', desc: 'python scripts/evaluate.py --dataset app/evaluation/evaluation_dataset.json' },
              { step: '4', title: 'View results', desc: 'Results saved to evaluation_results.json — metrics are computed from actual API responses.' },
            ].map((s) => (
              <div key={s.step} style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
                <div style={{
                  width: 32, height: 32, borderRadius: '50%',
                  background: 'var(--accent-gradient)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 13, fontWeight: 700, flexShrink: 0
                }}>{s.step}</div>
                <div>
                  <div style={{ fontWeight: 600, fontSize: 14 }}>{s.title}</div>
                  <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 4 }}>{s.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Metrics explained */}
        <div className="card" style={{ marginTop: 24 }}>
          <div className="card-title">📖 Metrics Explained</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {[
              { name: 'Retrieval Hit Rate', desc: 'Did we retrieve at least one expected source document?' },
              { name: 'Answer Correctness', desc: 'Keyword overlap between actual and expected answer.' },
              { name: 'Citation Accuracy', desc: 'Did the answer cite the correct source document?' },
              { name: 'Hallucination Guard', desc: 'LLM refuses to answer when no context retrieved.' },
              { name: 'Precision@K', desc: 'Fraction of retrieved sources that are relevant.' },
              { name: 'Recall@K', desc: 'Fraction of expected sources that were retrieved.' },
            ].map((m) => (
              <div key={m.name} style={{
                background: 'var(--bg-primary)', borderRadius: 'var(--radius-sm)',
                padding: '12px 14px', border: '1px solid var(--border)'
              }}>
                <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4 }}>{m.name}</div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{m.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}
