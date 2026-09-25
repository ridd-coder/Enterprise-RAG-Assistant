// src/api.js
// Centralised API client — all fetch calls go through here.

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

export const api = {
  // Health
  health: () => request('/api/health'),
  vectorDbHealth: () => request('/api/health/vector-db'),

  // Documents
  uploadDocument: (file) => {
    const form = new FormData();
    form.append('file', file);
    return fetch(`${BASE_URL}/api/documents/upload`, {
      method: 'POST',
      body: form,
    }).then(async (res) => {
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      return res.json();
    });
  },

  listDocuments: () => request('/api/documents'),

  deleteDocument: (id) =>
    request(`/api/documents/${id}`, { method: 'DELETE' }),

  // Chat
  chat: (question, conversationId, topK, documentIds) =>
    request('/api/chat', {
      method: 'POST',
      body: JSON.stringify({
        question,
        conversation_id: conversationId,
        top_k: topK,
        document_ids: documentIds,
      }),
    }),

  // Metrics
  metrics: () => request('/api/metrics'),
};
