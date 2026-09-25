// src/App.jsx
import { useState } from 'react';
import './index.css';
import Dashboard from './pages/Dashboard';
import DocumentsPage from './pages/DocumentsPage';
import ChatPage from './pages/ChatPage';
import EvaluationPage from './pages/EvaluationPage';

const PAGES = [
  { id: 'dashboard', label: 'Dashboard', icon: '🏠' },
  { id: 'documents', label: 'Documents', icon: '📁' },
  { id: 'chat', label: 'Chat', icon: '💬' },
  { id: 'evaluation', label: 'Evaluation', icon: '📊' },
];

export default function App() {
  const [activePage, setActivePage] = useState('chat');

  const renderPage = () => {
    switch (activePage) {
      case 'dashboard': return <Dashboard />;
      case 'documents': return <DocumentsPage />;
      case 'chat':      return <ChatPage />;
      case 'evaluation':return <EvaluationPage />;
      default:          return <ChatPage />;
    }
  };

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <h1>🧠 Enterprise RAG<br/>Knowledge Assistant</h1>
          <p>AI-powered document Q&amp;A</p>
        </div>

        <nav className="sidebar-nav">
          {PAGES.map((page) => (
            <button
              key={page.id}
              id={`nav-${page.id}`}
              className={`nav-item ${activePage === page.id ? 'active' : ''}`}
              onClick={() => setActivePage(page.id)}
            >
              <span className="nav-icon">{page.icon}</span>
              {page.label}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div>v1.0.0 · Production</div>
          <div style={{ marginTop: 4 }}>Powered by OpenAI + Qdrant</div>
        </div>
      </aside>

      {/* Main */}
      <main className="main-content">
        {renderPage()}
      </main>
    </div>
  );
}
