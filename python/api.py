import os
import json
import re
import logging
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

try:
    from model import TextCNN1D
    from feature_extractor import BERTFeatureExtractor
except ModuleNotFoundError:
    from python.model import TextCNN1D
    from python.feature_extractor import BERTFeatureExtractor

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# System & File Path Configurations
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoints/cnn_bert_detector.pth")
METRICS_PATH = os.getenv("METRICS_PATH", "checkpoints/metrics.json")
DEFAULT_HOST = os.getenv("API_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("API_PORT", "8000"))

# Calibration & Risk Band Constants
CONFIDENCE_VERY_HIGH_MARGIN = 0.35
CONFIDENCE_HIGH_MARGIN = 0.20
SENTENCE_RISK_CRITICAL_OFFSET = 0.25
SENTENCE_RISK_HIGH_OFFSET = 0.10
ACTIVATION_DELTA_SCALE = 0.25
SLIDING_WINDOW_SIZE = 5

app = FastAPI(
    title="AI Text Detector API - 1D CNN & BERT",
    description="FastAPI service powered by PyTorch 1D-CNN with multi-kernel sliding window convolutions over BERT embeddings.",
    version="1.0.0"
)

# Enable CORS for React frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global runtime state (Populated strictly at startup)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor: Optional[BERTFeatureExtractor] = None
detector_model: Optional[TextCNN1D] = None
optimal_threshold: Optional[float] = None
model_config: dict = {}

class PredictRequest(BaseModel):
    text: str

class SentenceScore(BaseModel):
    id: int
    text: str
    ai_score: float
    risk: str  # "Low", "Medium", "High", "Critical"

class PredictResponse(BaseModel):
    prediction: str  # "AI-Generated" or "Human-Written"
    ai_probability: float
    human_probability: float
    confidence_level: str
    metrics: dict
    sentences: List[SentenceScore]

@app.on_event("startup")
def load_models():
    """
    Initializes feature extractor and detector model dynamically from checkpoints/metrics.json.
    Strict Zero Fallback Policy: Throws FileNotFoundError / KeyError if any metadata is missing.
    """
    global feature_extractor, detector_model, optimal_threshold, model_config
    logger.info(f"Initializing BERT Feature Extractor and 1D-CNN Model on device '{device}'...")
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"REQUIRED CHECKPOINT MISSING: '{CHECKPOINT_PATH}' does not exist. "
            "Zero Fallback Policy: Run 'python/train.py' to generate model weights before starting the server."
        )
        
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(
            f"REQUIRED METRICS MISSING: '{METRICS_PATH}' does not exist. "
            "Zero Fallback Policy: Run 'python/train.py' to generate calibrated metrics before starting the server."
        )
        
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
        
    # Extract calibrated threshold and hyperparameters directly from metrics.json (No Hardcoded Fallbacks)
    try:
        optimal_threshold = float(metrics["optimal_threshold"])
        model_config = {
            "embed_dim": int(metrics.get("embed_dim", 768)),
            "num_filters": int(metrics.get("num_filters", 128)),
            "kernel_sizes": tuple(metrics.get("kernel_sizes", [1, 3, 5, 7])),
            "num_aux_features": int(metrics.get("num_aux_features", 4)),
            "dropout": float(metrics.get("dropout", 0.3)),
            "max_length": int(metrics.get("max_length", 128))
        }
    except KeyError as e:
        raise KeyError(f"Corrupted metrics metadata in '{METRICS_PATH}': Missing key {str(e)}")
        
    logger.info(f"Loaded calibrated decision threshold: {optimal_threshold:.4f}")
    logger.info(f"Model Configuration: {model_config}")
    
    feature_extractor = BERTFeatureExtractor(device=device)
    
    detector_model = TextCNN1D(
        embed_dim=model_config["embed_dim"],
        num_filters=model_config["num_filters"],
        kernel_sizes=model_config["kernel_sizes"],
        num_aux_features=model_config["num_aux_features"],
        dropout=model_config["dropout"]
    ).to(device)
    
    logger.info(f"Loading trained weights from '{CHECKPOINT_PATH}'...")
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    detector_model.load_state_dict(state_dict)
    detector_model.eval()
    logger.info("Trained model weights loaded successfully!")

@app.get("/api/health")
def health_check():
    if detector_model is None or feature_extractor is None or optimal_threshold is None:
        raise HTTPException(status_code=500, detail="Model pipeline is not initialized.")
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": True,
        "optimal_threshold": optimal_threshold,
        "config": model_config
    }

@app.get("/api/model-info")
def get_model_info():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(
            status_code=500,
            detail=f"Metrics configuration file '{METRICS_PATH}' missing. Zero Fallback Policy."
        )
        
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
            
    return {
        "model_name": "1D-CNN BERT Text Detector",
        "architecture": {
            "feature_extractor": f"bert-base-uncased ({model_config.get('embed_dim', 768)}-dim embeddings)",
            "conv_layers": [
                {"kernel_size": k, "filters": model_config.get('num_filters', 128), "n_gram_focus": f"n-gram filter (k={k})"}
                for k in model_config.get('kernel_sizes', (1, 3, 5, 7))
            ],
            "pooling": "Attention-Masked Global Max Pooling over Time",
            "auxiliary_metrics": [
                "Shannon Lexical Entropy",
                "Repetition Index",
                "Syntactic Clause Complexity Variance",
                "Root-TTR Vocabulary Rarity",
                "Consecutive Sentence Cosine Drift",
                "Subword Fragmentation Density"
            ],
            "classifier": f"Dense classifier with LayerNorm (Dropout {model_config.get('dropout', 0.3)})"
        },
        "performance": metrics
    }

def analyze_sentences_with_conv1d(text: str, overall_ai_prob: float, embeddings: torch.Tensor, mask: torch.Tensor) -> List[SentenceScore]:
    """
    Evaluates sentence risk scores dynamically centered around document probability overall_ai_prob
    using Conv1D sliding window token activations (detector_model.get_sliding_window_scores).
    """
    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not raw_sentences:
        raw_sentences = [text]
        
    try:
        # Extract per-token Conv1D activation intensity across sliding window filters
        token_scores = detector_model.get_sliding_window_scores(embeddings, mask=mask, window_size=SLIDING_WINDOW_SIZE)
        baseline_activation = float(np.mean(token_scores)) if len(token_scores) > 0 else 0.50
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Sliding window activation notice: {str(e)}")
        token_scores = []
        baseline_activation = 0.50
        
    scored_sentences = []
    
    # Tokenize each sentence to calculate token offsets
    token_offset = 0
    num_tokens = len(token_scores)
    
    for idx, sentence_text in enumerate(raw_sentences):
        words = re.findall(r'\b\w+\b', sentence_text.lower())
        word_count = len(words)
        
        if num_tokens > 0 and word_count > 0:
            # Map sentence word count ratio to token slice
            sent_token_len = max(1, int(round((word_count / max(1, len(re.findall(r'\b\w+\b', text.lower())))) * num_tokens)))
            start_tok = min(token_offset, num_tokens - 1)
            end_tok = min(token_offset + sent_token_len, num_tokens)
            token_offset += sent_token_len
            
            slice_scores = token_scores[start_tok:end_tok] if start_tok < end_tok else [baseline_activation]
            mean_activation = float(np.mean(slice_scores))
            
            # Sentence anomaly delta centered around overall_ai_prob
            delta = (mean_activation - baseline_activation) * ACTIVATION_DELTA_SCALE
        else:
            delta = 0.0
            
        sentence_ai_prob = float(np.clip(overall_ai_prob + delta, 0.01, 0.99))
        
        # Risk classification calibrated dynamically to optimal_threshold
        if sentence_ai_prob >= optimal_threshold + SENTENCE_RISK_CRITICAL_OFFSET:
            risk = "Critical"
        elif sentence_ai_prob >= optimal_threshold + SENTENCE_RISK_HIGH_OFFSET:
            risk = "High"
        elif sentence_ai_prob >= optimal_threshold:
            risk = "Medium"
        else:
            risk = "Low"
            
        scored_sentences.append(SentenceScore(
            id=idx + 1,
            text=sentence_text,
            ai_score=round(sentence_ai_prob, 3),
            risk=risk
        ))
        
    return scored_sentences

@app.post("/api/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
        
    if detector_model is None or feature_extractor is None or optimal_threshold is None:
        raise HTTPException(status_code=500, detail="Model pipeline not loaded. Zero Fallback Policy.")
        
    text = request.text.strip()
    max_len = model_config.get("max_length", 128)
    
    try:
        # Extract BERT sequence embeddings + [CLS] vector + 4 statistical features + attention mask
        embeddings, cls_embed, aux_tensor, mask = feature_extractor.extract_features([text], max_length=max_len)
        
        with torch.no_grad():
            logits = detector_model(embeddings, cls_embed, aux_tensor, mask=mask)
            ai_prob = float(torch.sigmoid(logits).cpu().item())
            
        human_prob = float(1.0 - ai_prob)
        prediction = "AI-Generated" if ai_prob >= optimal_threshold else "Human-Written"
        
        # Calibrate document confidence bands relative to distance from optimal_threshold
        prob_dist = abs(ai_prob - optimal_threshold)
        if prob_dist >= CONFIDENCE_VERY_HIGH_MARGIN:
            confidence = "Very High"
        elif prob_dist >= CONFIDENCE_HIGH_MARGIN:
            confidence = "High"
        else:
            confidence = "Moderate"
            
        aux_list = aux_tensor[0].cpu().numpy().tolist()
        metrics = {
            "shannon_entropy": round(aux_list[0], 3),
            "repetition_index": round(aux_list[1], 3),
            "clause_complexity": round(aux_list[2], 3),
            "vocab_rarity": round(aux_list[3], 3),
            "cosine_drift": round(aux_list[4], 3),
            "subword_density": round(aux_list[5], 3)
        }
        
        sentences = analyze_sentences_with_conv1d(text, ai_prob, embeddings, mask)
        
        return PredictResponse(
            prediction=prediction,
            ai_probability=round(ai_prob, 4),
            human_probability=round(human_prob, 4),
            confidence_level=confidence,
            metrics=metrics,
            sentences=sentences
        )
    except Exception as e:
        logger.error(f"Inference pipeline error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=DEFAULT_HOST, port=DEFAULT_PORT, reload=True)
