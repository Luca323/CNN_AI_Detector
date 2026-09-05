import React, { useState } from 'react';

const PRESETS = [
  {
    label: "Human Casual Post (RAID)",
    type: "human",
    text: "Honestly, I spent the entire weekend trying to fix a noisy wheel bearing on my 2012 Honda Civic. The rust on the lower control arm bolts was unreal. I broke two socket adapters before I finally tried heating them with a propane torch. After about an hour of hammering and applying PB Blaster, the bolt finally popped loose. If anyone else is attempting this job at home, definitely get a proper breaker bar beforehand or you'll regret it."
  },
  {
    label: "GPT-4 Technical Summary",
    type: "ai",
    text: "In conclusion, it is important to note that artificial intelligence systems operate by analyzing vast amounts of training data to identify recurring statistical patterns. Furthermore, machine learning models rely heavily on feature vectors to optimize predictive accuracy across complex multidated domains. It is crucial to consider that these automated mechanisms require continuous oversight and rigorous validation to ensure optimal performance and alignment with standardized benchmarks."
  },
  {
    label: "ChatGPT Technical Essay (RAID)",
    type: "ai",
    text: "In conclusion, artificial intelligence represents a paradigm shift in modern technological development. Furthermore, machine learning algorithms leverage deep neural networks to extract complex feature representations from unstructured datasets. It is crucial to note that continuous oversight and rigorous evaluation protocols are necessary to ensure optimum alignment with safety benchmarks."
  },
  {
    label: "Mixed AI & Human Text",
    type: "mixed",
    text: "I spent three hours yesterday attempting to debug a strange memory leak in our backend microservice. The culprit turned out to be an unclosed file handle inside a background worker loop. In conclusion, ensuring proper resource management is essential for maintaining system stability and preventing memory degradation over extended operational periods."
  }
];

export default function DetectorWorkbench({ onAnalyze, loading, result, error }) {
  const [inputText, setInputText] = useState(PRESETS[0].text);
  const [hoveredSentence, setHoveredSentence] = useState(null);

  const handlePresetSelect = (presetText) => {
    setInputText(presetText);
  };

  const wordCount = inputText.trim() ? inputText.trim().split(/\s+/).length : 0;
  const charCount = inputText.length;

  return (
    <div className="workbench-grid">
      {/* Input Section */}
      <div className="card input-card">
        <div className="card-header">
          <div className="card-title">
            <div className="card-title-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
              </svg>
            </div>
            <h2>Text Analysis Workbench</h2>
          </div>
          <div className="meta-badge">{wordCount} words | {charCount} chars</div>
        </div>

        {/* Preset Selector */}
        <div className="presets-bar">
          <span className="preset-label">Sample Presets:</span>
          {PRESETS.map((preset, idx) => (
            <button
              key={idx}
              className={`preset-btn ${preset.type}`}
              onClick={() => handlePresetSelect(preset.text)}
            >
              {preset.label}
            </button>
          ))}
        </div>

        <div className="textarea-wrapper">
          <textarea
            className="text-input"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Paste text here to analyze for AI generation patterns (minimum 20 words recommended)..."
            rows={10}
          />
        </div>

        <div className="card-footer">
          <button 
            className="clear-btn" 
            onClick={() => setInputText('')}
            disabled={loading}
          >
            Clear Text
          </button>
          <button 
            className="analyze-btn" 
            onClick={() => onAnalyze(inputText)}
            disabled={loading || !inputText.trim()}
          >
            {loading ? (
              <span className="spinner-wrap">
                <span className="spinner"></span> Analyzing Features...
              </span>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="11" cy="11" r="8"/>
                  <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                </svg>
                Scan for AI Content
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="error-banner">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
              <line x1="12" y1="9" x2="12" y2="13"/>
              <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
            {error}
          </div>
        )}
      </div>

      {/* Results & Heatmap Section */}
      {result && (
        <div className="results-container">
          {/* Main Confidence Gauge Card */}
          <div className={`card result-main-card ${result.prediction === 'AI-Generated' ? 'ai-detected' : 'human-detected'}`}>
            <div className="gauge-header">
              <div>
                <span className="result-badge">
                  {result.prediction === 'AI-Generated' ? 'AI-GENERATED CONTENT DETECTED' : 'HUMAN-WRITTEN CONTENT DETECTED'}
                </span>
                <h3 className="confidence-title">Confidence Level: {result.confidence_level}</h3>
              </div>

              <div className="score-pillar">
                <div className="score-val">{(result.ai_probability * 100).toFixed(1)}%</div>
                <div className="score-lbl">AI Probability Score</div>
              </div>
            </div>

            {/* Probability Bar */}
            <div className="prob-bar-wrapper">
              <div className="prob-bar-labels">
                <span>Human Written ({(result.human_probability * 100).toFixed(1)}%)</span>
                <span>AI Generated ({(result.ai_probability * 100).toFixed(1)}%)</span>
              </div>
              <div className="prob-bar-track">
                <div 
                  className="prob-bar-fill ai-fill" 
                  style={{ width: `${result.ai_probability * 100}%` }}
                ></div>
              </div>
            </div>
          </div>

          {/* Metric Cards Grid */}
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-icon-box">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                </svg>
              </div>
              <div className="metric-info">
                <div className="metric-val">{result.metrics.shannon_entropy ?? 0.0}</div>
                <div className="metric-lbl">Shannon Entropy</div>
                <div className="metric-desc">Vocabulary randomness degree</div>
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-icon-box">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="17 1 21 5 17 9"/>
                  <path d="M3 11V9a4 4 0 0 1 4-4h14"/>
                  <polyline points="7 23 3 19 7 15"/>
                  <path d="M21 13v2a4 4 0 0 1-4 4H3"/>
                </svg>
              </div>
              <div className="metric-info">
                <div className="metric-val">{result.metrics.repetition_index ?? 0.0}</div>
                <div className="metric-lbl">Repetition Index</div>
                <div className="metric-desc">N-gram repetition rate</div>
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-icon-box">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polygon points="12 2 2 7 12 12 22 7 12 2"/>
                  <polyline points="2 17 12 22 22 17"/>
                  <polyline points="2 12 12 17 22 12"/>
                </svg>
              </div>
              <div className="metric-info">
                <div className="metric-val">{result.metrics.clause_complexity ?? result.metrics.sentence_variation ?? 0.0}</div>
                <div className="metric-lbl">Clause Complexity</div>
                <div className="metric-desc">Syntactic clause variance</div>
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-icon-box">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
                  <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
                </svg>
              </div>
              <div className="metric-info">
                <div className="metric-val">{result.metrics.vocab_rarity ?? 0.0}</div>
                <div className="metric-lbl">Vocabulary Rarity</div>
                <div className="metric-desc">Root-TTR diversity ratio</div>
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-icon-box">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="20" x2="18" y2="10"/>
                  <line x1="12" y1="20" x2="12" y2="4"/>
                  <line x1="6" y1="20" x2="6" y2="14"/>
                </svg>
              </div>
              <div className="metric-info">
                <div className="metric-val">{result.metrics.cosine_drift ?? 0.5}</div>
                <div className="metric-lbl">Cosine Sim Drift</div>
                <div className="metric-desc">Sentence semantic continuity</div>
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-icon-box">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="3" width="7" height="7"/>
                  <rect x="14" y="3" width="7" height="7"/>
                  <rect x="14" y="14" width="7" height="7"/>
                  <rect x="3" y="14" width="7" height="7"/>
                </svg>
              </div>
              <div className="metric-info">
                <div className="metric-val">{result.metrics.subword_density ?? 0.0}</div>
                <div className="metric-lbl">Subword Density</div>
                <div className="metric-desc">BPE token fragmentation</div>
              </div>
            </div>
          </div>

          {/* Sliding Window Heatmap Visualizer */}
          <div className="card heatmap-card">
            <div className="card-header">
              <div className="card-title">
                <div className="card-title-icon">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="11" cy="11" r="8"/>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                  </svg>
                </div>
                <h3>Sliding Window Sentence Heatmap</h3>
              </div>
              <div className="heatmap-legend">
                <span className="legend-item low">Human (&lt;35%)</span>
                <span className="legend-item med">Neutral (35-55%)</span>
                <span className="legend-item high">Likely AI (55-75%)</span>
                <span className="legend-item crit">High AI (&gt;75%)</span>
              </div>
            </div>

            <p className="heatmap-subtitle">
              Hover over highlighted sentences to inspect local 1D CNN kernel activation risk scores:
            </p>

            <div className="heatmap-body">
              {result.sentences.map((sent) => {
                let riskClass = 'low';
                if (sent.ai_score >= 0.75) riskClass = 'crit';
                else if (sent.ai_score >= 0.55) riskClass = 'high';
                else if (sent.ai_score >= 0.35) riskClass = 'med';

                return (
                  <span
                    key={sent.id}
                    className={`heatmap-span ${riskClass}`}
                    onMouseEnter={() => setHoveredSentence(sent)}
                    onMouseLeave={() => setHoveredSentence(null)}
                  >
                    {sent.text}{' '}
                  </span>
                );
              })}
            </div>

            {/* Hover Tooltip Box */}
            {hoveredSentence && (
              <div className="sentence-tooltip">
                <div className="tooltip-header">
                  <strong>Sentence #{hoveredSentence.id} Risk Assessment:</strong>
                  <span className={`risk-tag ${hoveredSentence.risk.toLowerCase()}`}>
                    {hoveredSentence.risk} ({ (hoveredSentence.ai_score * 100).toFixed(1) }%)
                  </span>
                </div>
                <div className="tooltip-text">"{hoveredSentence.text}"</div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
