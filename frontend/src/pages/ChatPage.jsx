// src/pages/ChatPage.jsx
import { useState, useRef, useEffect } from 'react';
import { api } from '../api';

export default function ChatPage() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I\'m your Enterprise Knowledge Assistant. Upload your documents in the **Documents** tab, then ask me anything about them. I\'ll only answer from the uploaded content — never from prior knowledge.',
      sources: [],
      latencyMs: 0,
      chunks: 0,
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [error, setError] = useState('');
  const messagesEndRef = useRef();
  const textareaRef = useRef();

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e) => {
    e?.preventDefault();
    const question = input.trim();
    if (!question || loading) return;

    setInput('');
    setError('');

    // Optimistic user message
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: question, sources: [] },
    ]);
    setLoading(true);

    try {
      const res = await api.chat(question, conversationId);

      if (!conversationId) setConversationId(res.conversation_id);

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: res.answer,
          sources: res.sources || [],
          latencyMs: res.latency_ms,
          chunks: res.chunks_retrieved,
          model: res.model_used,
          tokens: res.tokens_used,
        },
      ]);
    } catch (e) {
      setError(e.message || 'Request failed. Check that the API is running.');
      setMessages((prev) => prev.slice(0, -1)); // Remove optimistic message
    } finally {
      setLoading(false);
      textareaRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const resetConversation = () => {
    setConversationId(null);
    setMessages([{
      role: 'assistant',
      content: 'Conversation cleared. Ask a new question!',
      sources: [],
    }]);
  };

  return (
    <>
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2>💬 Knowledge Chat</h2>
            <p>Ask questions about your uploaded documents</p>
          </div>
          <button id="new-conversation-btn" className="btn btn-secondary btn-sm" onClick={resetConversation}>
            ↺ New Conversation
          </button>
        </div>
      </div>

      <div className="page-body" style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 140px)', padding: '20px 40px' }}>
        {error && <div className="alert alert-error">{error}</div>}

        {/* Messages */}
        <div className="chat-messages" style={{ flex: 1 }}>
          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <div className="message-avatar">
                {msg.role === 'user' ? '👤' : '🧠'}
              </div>
              <div className="message-content">
                <div className="message-bubble">
                  {msg.content.split('\n').map((line, j) => (
                    <span key={j}>{line}<br /></span>
                  ))}
                </div>

                {/* Sources */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="sources-list">
                    <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 6 }}>
                      📚 Sources ({msg.sources.length})
                    </div>
                    {msg.sources.map((s, si) => (
                      <div key={si} className="source-card">
                        <span className="source-icon">📄</span>
                        <div className="source-info">
                          <div className="source-name">{s.filename}</div>
                          <div className="source-detail">Page {s.page_number}</div>
                        </div>
                        <span className="relevance-badge">
                          {(s.relevance_score * 100).toFixed(0)}% match
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Meta */}
                {msg.role === 'assistant' && msg.latencyMs > 0 && (
                  <div className="message-meta">
                    <span>⚡ {msg.latencyMs}ms</span>
                    {msg.chunks > 0 && <span>· 🔍 {msg.chunks} chunks</span>}
                    {msg.model && <span>· {msg.model}</span>}
                    {msg.tokens && <span>· {msg.tokens} tokens</span>}
                  </div>
                )}
              </div>
            </div>
          ))}

          {/* Typing indicator */}
          {loading && (
            <div className="message assistant">
              <div className="message-avatar">🧠</div>
              <div className="message-content">
                <div className="message-bubble" style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  <div className="spinner" />
                  <span style={{ color: 'var(--text-muted)', fontSize: 13 }}>Searching and generating…</span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="chat-input-area">
          <div className="chat-input-wrapper" style={{ flex: 1 }}>
            <textarea
              ref={textareaRef}
              id="chat-input"
              className="input"
              placeholder="Ask a question about your documents… (Enter to send, Shift+Enter for new line)"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={2}
              disabled={loading}
              style={{ resize: 'none' }}
            />
          </div>
          <button
            id="send-btn"
            className="btn btn-primary"
            onClick={handleSubmit}
            disabled={loading || !input.trim()}
            style={{ height: 48, whiteSpace: 'nowrap' }}
          >
            {loading ? <><div className="spinner" style={{ width: 16, height: 16 }} /> Thinking</> : '↑ Ask AI'}
          </button>
        </div>

        {/* Conversation ID */}
        {conversationId && (
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8, textAlign: 'center' }}>
            Conversation: {conversationId}
          </div>
        )}
      </div>
    </>
  );
}
