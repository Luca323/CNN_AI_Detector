import torch
import numpy as np
import re
from collections import Counter
from transformers import AutoTokenizer, AutoModel

class BERTFeatureExtractor:
    """
    Portfolio-Grade Feature Extractor returning:
    1. Full sequence BERT embeddings [Batch, Seq_Len, 768]
    2. [CLS] global sentence vector [Batch, 768]
    3. 6 Length-Invariant Auxiliary Feature Metrics:
       - Shannon Lexical Entropy (Vocabulary Randomness)
       - Repetition Index (Bigram/Trigram N-Gram Repetition Rate)
       - Clause Complexity Variance (Syntactic Punctuation Distribution)
       - Root-TTR Vocabulary Rarity (Guiraud R Index)
       - Consecutive Sentence Cosine Similarity Drift (Semantic Continuity)
       - Subword Fragmentation Density (BPE Subword-to-Word Ratio)
    4. Attention Mask tensor [Batch, Seq_Len] to eliminate padding token bias
    """
    def __init__(self, model_name="bert-base-uncased", device=None, unfreeze_top_layers=False):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Loading BERT model '{model_name}' on device: {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        if not unfreeze_top_layers:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            print("Unfreezing top 2 BERT transformer encoder layers for domain adaptation...")
            for param in self.model.parameters():
                param.requires_grad = False
            for layer in self.model.encoder.layer[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True
                    
    def compute_auxiliary_metrics(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            raise ValueError("Input text contains no valid words for feature extraction.")
            
        total_words = len(words)
        unique_words = len(set(words))
        
        # 1. Shannon Lexical Entropy H(X) = -sum(p_i * log2(p_i))
        word_counts = Counter(words)
        probs = [count / total_words for count in word_counts.values()]
        shannon_h = -sum(p * np.log2(p) for p in probs)
        max_h = np.log2(total_words) if total_words > 1 else 1.0
        shannon_entropy = float(np.clip(shannon_h / max_h, 0.05, 0.95))
        
        # 2. Repetition Index (Length-normalized bigram/trigram repetition rate)
        if total_words > 3:
            bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
            trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            bigram_rep = (len(bigrams) - len(set(bigrams))) / max(1, len(bigrams))
            trigram_rep = (len(trigrams) - len(set(trigrams))) / max(1, len(trigrams))
            repetition_index = float(bigram_rep * 0.5 + trigram_rep * 0.5)
        else:
            repetition_index = 0.0
            
        # 3. Syntactic Clause Complexity Variance (Replaces fragile word-count burstiness)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) > 1:
            clause_counts = [len(re.findall(r'[,;\-\(\)]', s)) for s in sentences]
            clause_mean = float(np.mean(clause_counts))
            clause_std = float(np.std(clause_counts))
            clause_complexity = float(np.clip(clause_std / (clause_mean + 1.0), 0.0, 1.0))
        else:
            clause_complexity = 0.0
            
        # 4. Root-TTR (Guiraud's R Index: unique / sqrt(total)) - Length invariant vocabulary diversity
        # PROVENANCE: Typical English texts (50-500 words) yield Guiraud R in range [1.5, 5.0].
        # Dividing by empirical factor 3.5 centers values around 0.5 for LayerNorm & Neural Net stability.
        root_ttr = (unique_words / np.sqrt(total_words)) / 3.5
        root_ttr = float(np.clip(root_ttr, 0.05, 0.95))
        
        # 5. Consecutive Sentence Cosine Similarity Drift (Semantic Continuity)
        if len(sentences) > 1:
            sent_encoded = self.tokenizer(sentences, padding=True, truncation=True, max_length=64, return_tensors="pt")
            s_input_ids = sent_encoded["input_ids"].to(self.device)
            s_mask = sent_encoded["attention_mask"].to(self.device)
            with torch.no_grad():
                s_outputs = self.model(input_ids=s_input_ids, attention_mask=s_mask)
                s_cls = s_outputs.last_hidden_state[:, 0, :]  # [num_sentences, 768]
                s_cls_norm = torch.nn.functional.normalize(s_cls, p=2, dim=1)
                
            sims = torch.sum(s_cls_norm[:-1] * s_cls_norm[1:], dim=1)
            mean_cosine_sim = float(sims.mean().cpu().item())
            cosine_drift = float(np.clip(mean_cosine_sim, 0.0, 1.0))
        else:
            cosine_drift = 0.50
            
        # 6. Subword Fragmentation Density (BPE/WordPiece Subword-to-Word Ratio)
        full_encoded = self.tokenizer(text, return_tensors="pt")
        num_subwords = full_encoded["input_ids"].size(1)
        subword_density = float(np.clip((num_subwords / max(1, total_words)) / 2.0, 0.05, 0.95))
        
        return [
            float(shannon_entropy),
            float(repetition_index),
            float(clause_complexity),
            float(root_ttr),
            float(cosine_drift),
            float(subword_density)
        ]

    def extract_features(self, text_list, max_length=128):
        encoded = self.tokenizer(
            text_list,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            cls_embedding = last_hidden_state[:, 0, :]
            
        aux_features_list = [self.compute_auxiliary_metrics(t) for t in text_list]
        aux_tensor = torch.tensor(aux_features_list, dtype=torch.float32).to(self.device)
        
        return last_hidden_state, cls_embedding, aux_tensor, attention_mask
