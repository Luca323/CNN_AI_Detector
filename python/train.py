import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

try:
    from feature_extractor import BERTFeatureExtractor
    from model import TextCNN1D
except ModuleNotFoundError:
    from python.feature_extractor import BERTFeatureExtractor
    from python.model import TextCNN1D

def extract_dataset_features(df, feature_extractor, batch_size=32, desc="Data"):
    texts = df['generation'].tolist()
    labels = df['label'].values
    all_embeddings = []
    all_cls = []
    all_aux = []
    all_masks = []
    
    print(f"Extracting BERT features for {desc} ({len(texts)} samples)...", flush=True)
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        emb, cls_v, aux, mask = feature_extractor.extract_features(batch_texts, max_length=128)
        all_embeddings.append(emb.cpu())
        all_cls.append(cls_v.cpu())
        all_aux.append(aux.cpu())
        all_masks.append(mask.cpu())
        if (i // batch_size) % 20 == 0:
            print(f"  [{desc}] Processed {min(i+batch_size, len(texts))}/{len(texts)} samples", flush=True)
            
    X_embed = torch.cat(all_embeddings, dim=0)
    X_cls = torch.cat(all_cls, dim=0)
    X_aux = torch.cat(all_aux, dim=0)
    X_mask = torch.cat(all_masks, dim=0)
    y_tensor = torch.tensor(labels, dtype=torch.float32)
    
    return TensorDataset(X_embed, X_cls, X_aux, X_mask, y_tensor)

def train_detector():
    # Set explicit seeds for full training reproducibility (§4.3)
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Advanced 1D-CNN AI Text Detector on RAID Dataset ---", flush=True)
    print(f"Device: {device}", flush=True)
    
    os.makedirs("checkpoints", exist_ok=True)
    
    # 1. Load Dataset
    dataset_path = "./my_local_raid_dataset"
    print(f"Loading RAID dataset from {dataset_path}...", flush=True)
    dataset = load_from_disk(dataset_path)
    
    train_df = dataset['train'].to_pandas()
    print(f"Total RAID Dataset rows: {len(train_df)}", flush=True)
    
    train_df['label'] = (train_df['model'] != 'human').astype(int)
    
    print("\nDataset Class Distribution:", flush=True)
    print(train_df['label'].value_counts(), flush=True)
    
    include_attacks = False
    if not include_attacks:
        print("\nFiltering for clean baseline samples only (attack == 'none')...", flush=True)
        train_df = train_df[train_df['attack'] == 'none']
    else:
        print("\nIncluding clean samples + adversarial attack variations...", flush=True)
    
    # 2. Balanced Sampling
    sample_size_per_class = 2500
    human_samples = train_df[train_df['label'] == 0].sample(n=min(sample_size_per_class, len(train_df[train_df['label'] == 0])), random_state=42)
    ai_samples = train_df[train_df['label'] == 1].sample(n=min(sample_size_per_class, len(train_df[train_df['label'] == 1])), random_state=42)
    
    sample_df = pd.concat([human_samples, ai_samples]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"\nBalanced Subset size: {len(sample_df)}", flush=True)
    
    # 3. SPLIT-BEFORE-EXTRACT (4-Partition Split: 70% Train / 10% Val / 10% Calibration / 10% Test)
    print("\nSplitting raw dataset into 4 isolated partitions before feature extraction...", flush=True)
    train_df, temp_df = train_test_split(sample_df, test_size=0.30, random_state=42, stratify=sample_df['label'])
    val_df, test_calib_df = train_test_split(temp_df, test_size=0.6667, random_state=42, stratify=temp_df['label'])
    calib_df, test_df = train_test_split(test_calib_df, test_size=0.50, random_state=42, stratify=test_calib_df['label'])
    
    print(f"Partition Sizes: Train={len(train_df)}, Val={len(val_df)}, Calib={len(calib_df)}, Test={len(test_df)}", flush=True)
    
    # 4. Extract Features independently per partition
    feature_extractor = BERTFeatureExtractor(device=device)
    
    train_ds = extract_dataset_features(train_df, feature_extractor, desc="Train Set")
    val_ds = extract_dataset_features(val_df, feature_extractor, desc="Val Set")
    calib_ds = extract_dataset_features(calib_df, feature_extractor, desc="Calibration Set")
    test_ds = extract_dataset_features(test_df, feature_extractor, desc="Test Set")
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    calib_loader = DataLoader(calib_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # 5. Initialize 1D-CNN with k=(1, 3, 5, 7) and CLS fusion
    model = TextCNN1D(
        embed_dim=768,
        num_filters=128,
        kernel_sizes=(1, 3, 5, 7),
        num_aux_features=6,
        dropout=0.3
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    
    epochs = 8
    best_val_acc = 0.0
    
    print("\nStarting Training & Validation Loop...", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for b_emb, b_cls, b_aux, b_mk, b_y in train_loader:
            b_emb, b_cls, b_aux, b_mk, b_y = b_emb.to(device), b_cls.to(device), b_aux.to(device), b_mk.to(device), b_y.to(device)
            
            optimizer.zero_grad()
            logits = model(b_emb, b_cls, b_aux, mask=b_mk)
            loss = criterion(logits, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation Evaluation
        model.eval()
        val_probs = []
        val_targets = []
        
        with torch.no_grad():
            for b_emb, b_cls, b_aux, b_mk, b_y in val_loader:
                b_emb, b_cls, b_aux, b_mk = b_emb.to(device), b_cls.to(device), b_aux.to(device), b_mk.to(device)
                logits = model(b_emb, b_cls, b_aux, mask=b_mk)
                probs = torch.sigmoid(logits).cpu().numpy()
                
                val_probs.extend(probs)
                val_targets.extend(b_y.numpy())
                
        val_preds = (np.array(val_probs) >= 0.5).astype(int)
        val_acc = accuracy_score(val_targets, val_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(val_targets, val_preds, average='binary')
        auc = roc_auc_score(val_targets, val_probs)
        
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | F1: {f1:.4f} | ROC-AUC: {auc:.4f}", flush=True)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/cnn_bert_detector.pth")
            
    # 6. Threshold Tuning on Calibration Split ONLY
    print("\n--- Tuning Decision Threshold on Calibration Split ---", flush=True)
    model.load_state_dict(torch.load("checkpoints/cnn_bert_detector.pth", weights_only=True))
    model.eval()
    
    calib_probs = []
    calib_targets = []
    with torch.no_grad():
        for b_emb, b_cls, b_aux, b_mk, b_y in calib_loader:
            b_emb, b_cls, b_aux, b_mk = b_emb.to(device), b_cls.to(device), b_aux.to(device), b_mk.to(device)
            logits = model(b_emb, b_cls, b_aux, mask=b_mk)
            calib_probs.extend(torch.sigmoid(logits).cpu().numpy())
            calib_targets.extend(b_y.numpy())
            
    best_thresh = 0.50
    best_calib_acc = 0.0
    for thresh in np.linspace(0.35, 0.65, 31):
        t_preds = (np.array(calib_probs) >= thresh).astype(int)
        acc = accuracy_score(calib_targets, t_preds)
        if acc > best_calib_acc:
            best_calib_acc = acc
            best_thresh = float(thresh)
            
    print(f"Optimal Decision Threshold (from Calibration split): {best_thresh:.4f} (Calib Acc: {best_calib_acc*100:.2f}%)", flush=True)

    # 7. Final Evaluation on Holdout Test Set using Calibrated Threshold
    print("\n--- Running Final Evaluation on Unseen Holdout Test Set ---", flush=True)
    test_probs = []
    test_targets = []
    with torch.no_grad():
        for b_emb, b_cls, b_aux, b_mk, b_y in test_loader:
            b_emb, b_cls, b_aux, b_mk = b_emb.to(device), b_cls.to(device), b_aux.to(device), b_mk.to(device)
            logits = model(b_emb, b_cls, b_aux, mask=b_mk)
            test_probs.extend(torch.sigmoid(logits).cpu().numpy())
            test_targets.extend(b_y.numpy())
            
    test_preds = (np.array(test_probs) >= best_thresh).astype(int)
    test_acc = accuracy_score(test_targets, test_preds)
    t_prec, t_rec, t_f1, _ = precision_recall_fscore_support(test_targets, test_preds, average='binary')
    t_auc = roc_auc_score(test_targets, test_probs)
    
    print(f"Holdout Test Accuracy: {test_acc*100:.2f}% | F1: {t_f1:.4f} | ROC-AUC: {t_auc:.4f}", flush=True)
    
    metrics_summary = {
        "val_accuracy": float(best_val_acc),
        "calib_accuracy": float(best_calib_acc),
        "test_accuracy": float(test_acc),
        "test_f1": float(t_f1),
        "test_auc": float(t_auc),
        "optimal_threshold": float(best_thresh),
        "embed_dim": 768,
        "kernel_sizes": [1, 3, 5, 7],
        "num_filters": 128,
        "num_aux_features": 6,
        "dropout": 0.3,
        "max_length": 128,
        "dataset": "RAID Baseline",
        "training_samples": len(sample_df)
    }
    with open("checkpoints/metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
        
    print(f"\nTraining & Evaluation Complete! Checkpoint saved to 'checkpoints/cnn_bert_detector.pth'", flush=True)

if __name__ == "__main__":
    train_detector()
