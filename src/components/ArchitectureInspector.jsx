import React, { useState } from 'react';

export default function ArchitectureInspector() {
  const [selectedKernel, setSelectedKernel] = useState(3);
  const [windowPos, setWindowPos] = useState(0);

  const sampleTokens = ["Deep", "learning", "models", "generate", "fluent", "paragraphs", "with", "repetitive", "phrasing"];

  const maxPos = Math.max(0, sampleTokens.length - selectedKernel);

  return (
    <div className="card architecture-card">
      <div className="card-header">
        <div className="card-title">
          <div className="card-title-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="4" y="4" width="16" height="16" rx="2" ry="2"/>
              <rect x="9" y="9" width="6" height="6"/>
              <line x1="9" y1="1" x2="9" y2="4"/>
              <line x1="15" y1="1" x2="15" y2="4"/>
            </svg>
          </div>
          <h2>1D-CNN Model Architecture & Sliding Window Mechanics</h2>
        </div>
        <div className="meta-badge">PyTorch + BERT Feature Space</div>
      </div>

      <p className="section-desc">
        Unlike 2D CNNs used in computer vision, a <strong>1D Convolutional Neural Network</strong> processes temporal or sequential vectors ($L \times D$). 
        By sliding multi-size kernel filters over BERT token embeddings, the 1D CNN captures distinct n-gram stylistic patterns that reveal AI text generation signatures.
      </p>

      {/* Architecture Flow Diagram */}
      <div className="arch-flow">
        <div className="flow-step">
          <div className="step-badge">1. Input Text</div>
          <div className="step-box">Sequence of Tokens [L]</div>
        </div>
        <div className="flow-arrow">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="5" y1="12" x2="19" y2="12"/>
            <polyline points="12 5 19 12 12 19"/>
          </svg>
        </div>

        <div className="flow-step">
          <div className="step-badge">2. BERT Embeddings</div>
          <div className="step-box highlight-bert">BERT Base [L × 768]</div>
        </div>
        <div className="flow-arrow">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="5" y1="12" x2="19" y2="12"/>
            <polyline points="12 5 19 12 12 19"/>
          </svg>
        </div>

        <div className="flow-step">
          <div className="step-badge">3. 1D Conv Layers</div>
          <div className="step-box highlight-cnn">
            Conv1D (k=3, 5, 7) × 128 Filters
          </div>
        </div>
        <div className="flow-arrow">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="5" y1="12" x2="19" y2="12"/>
            <polyline points="12 5 19 12 12 19"/>
          </svg>
        </div>

        <div className="flow-step">
          <div className="step-badge">4. Max Pool & FC</div>
          <div className="step-box">Global Max Pool + Aux Metrics</div>
        </div>
        <div className="flow-arrow">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="5" y1="12" x2="19" y2="12"/>
            <polyline points="12 5 19 12 12 19"/>
          </svg>
        </div>

        <div className="flow-step">
          <div className="step-badge">5. Output</div>
          <div className="step-box highlight-out">AI / Human Probability</div>
        </div>
      </div>

      {/* Interactive Sliding Window Simulator */}
      <div className="simulator-section">
        <h3>Interactive Sliding Window Simulator</h3>
        <p className="section-desc">Select a kernel size to observe how 1D convolutions extract n-gram feature vectors:</p>

        <div className="kernel-tabs">
          <button 
            className={`kernel-tab ${selectedKernel === 3 ? 'active' : ''}`}
            onClick={() => { setSelectedKernel(3); setWindowPos(0); }}
          >
            Kernel k=3 (Trigrams)
          </button>
          <button 
            className={`kernel-tab ${selectedKernel === 5 ? 'active' : ''}`}
            onClick={() => { setSelectedKernel(5); setWindowPos(0); }}
          >
            Kernel k=5 (Pentagrams)
          </button>
          <button 
            className={`kernel-tab ${selectedKernel === 7 ? 'active' : ''}`}
            onClick={() => { setSelectedKernel(7); setWindowPos(0); }}
          >
            Kernel k=7 (Septagrams)
          </button>
        </div>

        {/* Tokens Track */}
        <div className="tokens-track-wrapper">
          <div className="tokens-track">
            {sampleTokens.map((tok, idx) => {
              const inWindow = idx >= windowPos && idx < windowPos + selectedKernel;
              return (
                <div 
                  key={idx} 
                  className={`token-cell ${inWindow ? 'in-window' : ''}`}
                >
                  <span className="token-text">{tok}</span>
                  <span className="token-idx">t_{idx}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Window Controls */}
        <div className="simulator-controls">
          <button 
            className="sim-btn" 
            onClick={() => setWindowPos(Math.max(0, windowPos - 1))}
            disabled={windowPos === 0}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="19" y1="12" x2="5" y2="12"/>
              <polyline points="12 19 5 12 12 5"/>
            </svg>
            Step Back
          </button>

          <span className="sim-info">
            Sliding Window Position: <strong>[{windowPos} .. {windowPos + selectedKernel - 1}]</strong>
          </span>

          <button 
            className="sim-btn" 
            onClick={() => setWindowPos(Math.min(maxPos, windowPos + 1))}
            disabled={windowPos >= maxPos}
          >
            Step Forward
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="5" y1="12" x2="19" y2="12"/>
              <polyline points="12 5 19 12 12 19"/>
            </svg>
          </button>
        </div>

        <div className="kernel-explanation">
          {selectedKernel === 3 && (
            <p><strong>Kernel 3 (Trigram Focus):</strong> Scans 3 consecutive token embeddings. Detects local word pairings, transitions, and sudden stylistic changes.</p>
          )}
          {selectedKernel === 5 && (
            <p><strong>Kernel 5 (Pentagram Focus):</strong> Scans 5 consecutive token embeddings. Captures clause-level syntactic structures and formulaic phrasing typical of LLMs.</p>
          )}
          {selectedKernel === 7 && (
            <p><strong>Kernel 7 (Septagram Focus):</strong> Scans 7 consecutive token embeddings. Analyzes overall sentence rhythm, structural repetition, and long-range n-gram patterns.</p>
          )}
        </div>
      </div>

      {/* Model Spec Table */}
      <div className="spec-table-container">
        <h3>Model Specifications</h3>
        <table className="spec-table">
          <thead>
            <tr>
              <th>Layer / Component</th>
              <th>Dimensions / Parameters</th>
              <th>Role in Detection</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>BERT Encoder</td>
              <td>110M parameters (768-dim)</td>
              <td>Extracts dense contextual representations per token</td>
            </tr>
            <tr>
              <td>Conv1D Branch (k=1)</td>
              <td>128 Filters, Weight: 1×768×128</td>
              <td>Pointwise token identity feature maps</td>
            </tr>
            <tr>
              <td>Conv1D Branch (k=3)</td>
              <td>128 Filters, Weight: 3×768×128</td>
              <td>Local trigram feature maps</td>
            </tr>
            <tr>
              <td>Conv1D Branch (k=5)</td>
              <td>128 Filters, Weight: 5×768×128</td>
              <td>Clause-level pentagram feature maps</td>
            </tr>
            <tr>
              <td>Conv1D Branch (k=7)</td>
              <td>128 Filters, Weight: 7×768×128</td>
              <td>Extended septagram sentence rhythm maps</td>
            </tr>
            <tr>
              <td>Auxiliary Feature Vector</td>
              <td>6 Metrics (Entropy, Repetition, Clause Variance, Root-TTR, Cosine Drift, Subword Density)</td>
              <td>Injects length-invariant NLP corpus features into fusion head</td>
            </tr>
            <tr>
              <td>Classifier Head</td>
              <td>Linear (512 CNN + 768 CLS + 6 Aux = 1286 ➔ 512 ➔ 128 ➔ 1)</td>
              <td>Dense multi-layer perceptron with LayerNorm & Dropout</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
