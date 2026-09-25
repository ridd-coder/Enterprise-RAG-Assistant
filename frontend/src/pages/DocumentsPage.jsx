// src/pages/DocumentsPage.jsx
import { useState, useEffect, useRef } from 'react';
import { api } from '../api';

export default function DocumentsPage() {
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef();

  const loadDocs = () => {
    api.listDocuments()
      .then((r) => setDocuments(r.documents))
      .catch(() => {});
  };

  useEffect(() => { loadDocs(); }, []);

  const handleFiles = async (files) => {
    const file = files[0];
    if (!file) return;
    if (!file.name.endsWith('.pdf')) {
      setError('Only PDF files are supported.');
      return;
    }

    setUploading(true);
    setError('');
    setSuccess('');

    try {
      const result = await api.uploadDocument(file);
      setSuccess(`✓ "${result.filename}" ingested — ${result.page_count} pages, ${result.chunk_count} chunks.`);
      loadDocs();
    } catch (e) {
      setError(e.message || 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (id, name) => {
    if (!window.confirm(`Delete "${name}" and all its vectors?`)) return;
    try {
      await api.deleteDocument(id);
      setSuccess(`Deleted "${name}".`);
      loadDocs();
    } catch (e) {
      setError(e.message);
    }
  };

  const formatBytes = (b) => {
    if (b < 1024) return `${b} B`;
    if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
    return `${(b / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDate = (d) => new Date(d).toLocaleDateString('en-US', {
    year: 'numeric', month: 'short', day: 'numeric'
  });

  return (
    <>
      <div className="page-header">
        <h2>Document Library</h2>
        <p>Upload PDFs to build your knowledge base</p>
      </div>
      <div className="page-body">
        {error && <div className="alert alert-error">{error}</div>}
        {success && <div className="alert alert-success">{success}</div>}

        {/* Upload zone */}
        <div
          id="upload-zone"
          className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
          style={{ marginBottom: 28 }}
          onClick={() => fileRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            handleFiles(e.dataTransfer.files);
          }}
        >
          <div className="upload-icon">
            {uploading ? '⏳' : '📂'}
          </div>
          <h3>{uploading ? 'Ingesting document…' : 'Drop PDF here or click to browse'}</h3>
          <p>Supports PDF up to 50 MB · Text extraction · Automatic chunking & embedding</p>

          {uploading && (
            <div className="progress-bar" style={{ maxWidth: 300, margin: '16px auto 0' }}>
              <div className="progress-fill animate-pulse" style={{ width: '70%' }} />
            </div>
          )}
        </div>

        <input
          ref={fileRef}
          type="file"
          accept=".pdf"
          style={{ display: 'none' }}
          onChange={(e) => handleFiles(e.target.files)}
        />

        {/* Document list */}
        <div className="card-title" style={{ marginBottom: 16 }}>
          📁 Knowledge Base ({documents.length} documents)
        </div>

        {documents.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">📭</div>
            <h3>No documents yet</h3>
            <p>Upload your first PDF to get started</p>
          </div>
        ) : (
          <div className="documents-grid">
            {documents.map((doc) => (
              <div key={doc.document_id} className="document-row">
                <div className="doc-icon">📄</div>
                <div className="doc-info">
                  <div className="doc-name">{doc.filename}</div>
                  <div className="doc-meta">
                    {doc.page_count} pages · {doc.chunk_count} chunks ·{' '}
                    {formatBytes(doc.file_size_bytes)} · Uploaded {formatDate(doc.uploaded_at)}
                  </div>
                </div>
                <span className="doc-badge badge-success">Indexed</span>
                <button
                  id={`delete-${doc.document_id}`}
                  className="btn btn-danger btn-sm"
                  onClick={() => handleDelete(doc.document_id, doc.filename)}
                >
                  🗑 Delete
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}
