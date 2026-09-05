import React, { useState } from 'react';
import DetectorWorkbench from './components/DetectorWorkbench';
import ArchitectureInspector from './components/ArchitectureInspector';
import RaidBenchmark from './components/RaidBenchmark';

export default function App() {
  const [activeTab, setActiveTab] = useState('workbench');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleAnalyzeText = async (text) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // API call to FastAPI backend server
      const response = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned HTTP ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error("Backend API Connection Error:", err);
      // Zero Fallback Policy: Display explicit error message banner
      setError(err.message || "Failed to reach AI Text Detector backend service at http://localhost:8000.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="navbar">
        <div className="brand">
          <div className="brand-logo-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
              <path d="m9 12 2 2 4-4"/>
            </svg>
          </div>
          <div className="brand-text">
            <h1>DeepTrace AI</h1>
            <p>1D-CNN Multi-Kernel AI Text Detection Engine</p>
          </div>
        </div>

        <div className="nav-badges">
          <span className="tech-badge">PyTorch 1D-CNN</span>
          <span className="tech-badge">BERT Embeddings</span>
          <span className="tech-badge">RAID Dataset</span>
          <span className="tech-badge">FastAPI</span>
        </div>
      </header>

      {/* Tabs Navigation */}
      <nav className="tabs-nav">
        <button
          className={`tab-btn ${activeTab === 'workbench' ? 'active' : ''}`}
          onClick={() => setActiveTab('workbench')}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
            <line x1="16" y1="13" x2="8" y2="13"/>
            <line x1="16" y1="17" x2="8" y2="17"/>
            <polyline points="10 9 9 9 8 9"/>
          </svg>
          Detector Workbench
        </button>

        <button
          className={`tab-btn ${activeTab === 'architecture' ? 'active' : ''}`}
          onClick={() => setActiveTab('architecture')}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="4" y="4" width="16" height="16" rx="2" ry="2"/>
            <rect x="9" y="9" width="6" height="6"/>
            <line x1="9" y1="1" x2="9" y2="4"/>
            <line x1="15" y1="1" x2="15" y2="4"/>
            <line x1="9" y1="20" x2="9" y2="23"/>
            <line x1="15" y1="20" x2="15" y2="23"/>
            <line x1="20" y1="9" x2="23" y2="9"/>
            <line x1="20" y1="14" x2="23" y2="14"/>
            <line x1="1" y1="9" x2="4" y2="9"/>
            <line x1="1" y1="14" x2="4" y2="14"/>
          </svg>
          Model Architecture (1D-CNN)
        </button>

        <button
          className={`tab-btn ${activeTab === 'benchmark' ? 'active' : ''}`}
          onClick={() => setActiveTab('benchmark')}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="20" x2="18" y2="10"/>
            <line x1="12" y1="20" x2="12" y2="4"/>
            <line x1="6" y1="20" x2="6" y2="14"/>
          </svg>
          RAID Dataset Benchmark
        </button>
      </nav>

      {/* Main Content Area */}
      <main className="main-content">
        {activeTab === 'workbench' && (
          <DetectorWorkbench
            onAnalyze={handleAnalyzeText}
            loading={loading}
            result={result}
            error={error}
          />
        )}

        {activeTab === 'architecture' && (
          <ArchitectureInspector />
        )}

        {activeTab === 'benchmark' && (
          <RaidBenchmark />
        )}
      </main>
    </div>
  );
}
