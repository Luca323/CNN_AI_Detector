import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN1D(nn.Module):
    """
    Length-Invariant Multi-Kernel 1D-CNN for AI Text Detection.
    Includes Attention Masking over padding tokens during Conv1D Max Pooling
    to guarantee identical scoring behavior across short (10 words) and long (500 words) texts.
    """
    def __init__(self, embed_dim=768, num_filters=128, kernel_sizes=(1, 3, 5, 7), num_aux_features=6, dropout=0.3):
        super(TextCNN1D, self).__init__()
        
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        
        # 1D Convolutional Layers (k=1 pointwise + k=3,5,7 n-gram sliding windows)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2
            )
            for k in kernel_sizes
        ])
        
        self.layernorms = nn.ModuleList([
            nn.LayerNorm(num_filters) for _ in kernel_sizes
        ])
        
        total_cnn_dim = num_filters * len(kernel_sizes)  # 128 * 4 = 512
        cls_dim = embed_dim  # 768
        
        self.ln_cls = nn.LayerNorm(cls_dim)
        
        total_input_dim = total_cnn_dim + cls_dim + num_aux_features
        self.ln_combined = nn.LayerNorm(total_input_dim)
        
        self.fc1 = nn.Linear(total_input_dim, 512)
        self.ln_fc1 = nn.LayerNorm(512)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(512, 128)
        self.ln_fc2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(dropout)
        
        self.out = nn.Linear(128, 1)
        
        # Zero-center output layer initialization
        nn.init.normal_(self.out.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.out.bias, 0.0)

    def forward(self, x_embed, cls_embed, aux_features, mask=None):
        """
        x_embed: [Batch, Seq_Len, 768] (BERT token sequence)
        cls_embed: [Batch, 768] (BERT CLS global vector)
        aux_features: [Batch, 4] (Length-normalized statistical metrics)
        mask: [Batch, Seq_Len] (Attention mask: 1 for real tokens, 0 for [PAD])
        """
        # Conv1d expects [Batch, Channels, Seq_Len]
        x = x_embed.permute(0, 2, 1)
        
        pooled_outputs = []
        for conv, ln in zip(self.convs, self.layernorms):
            c = F.relu(conv(x))
            c = ln(c.permute(0, 2, 1)).permute(0, 2, 1)
            
            # Mask out padding token positions before max pooling!
            if mask is not None:
                mask_expanded = mask.unsqueeze(1).to(c.device)  # [Batch, 1, Seq_Len]
                c = c.masked_fill(mask_expanded == 0, -1e9)
                
            # Max pooling over valid real tokens only
            p = F.max_pool1d(c, kernel_size=c.size(2)).squeeze(2)
            pooled_outputs.append(p)
            
        cnn_features = torch.cat(pooled_outputs, dim=1)  # [Batch, 512]
        
        norm_cls = self.ln_cls(cls_embed)
        
        combined = torch.cat([cnn_features, norm_cls, aux_features], dim=1)  # [Batch, 1284]
        combined_norm = self.ln_combined(combined)
        
        h = F.relu(self.ln_fc1(self.fc1(combined_norm)))
        h = self.dropout1(h)
        h = F.relu(self.ln_fc2(self.fc2(h)))
        h = self.dropout2(h)
        
        logits = self.out(h)
        return logits.squeeze(1)

    def get_sliding_window_scores(self, x_embed, mask=None, window_size=5):
        self.eval()
        with torch.no_grad():
            x = x_embed.permute(0, 2, 1)
            conv_activations = []
            for conv, ln in zip(self.convs, self.layernorms):
                c = F.relu(conv(x))
                c = ln(c.permute(0, 2, 1)).permute(0, 2, 1)
                if mask is not None:
                    mask_exp = mask.unsqueeze(1).to(c.device)
                    c = c * mask_exp
                conv_activations.append(c)
                
            concat_act = torch.cat(conv_activations, dim=1)
            pos_importance = concat_act.abs().mean(dim=1).squeeze(0)
            
            if mask is not None:
                valid_len = int(mask.squeeze(0).sum().item())
                pos_importance = pos_importance[:valid_len]
                
            pos_importance_np = pos_importance.cpu().numpy()
            if len(pos_importance_np) == 0:
                raise ValueError("No valid tokens found for sliding window scoring.")
                
            smoothed = []
            half_w = window_size // 2
            for i in range(len(pos_importance_np)):
                start = max(0, i - half_w)
                end = min(len(pos_importance_np), i + half_w + 1)
                smoothed.append(float(pos_importance_np[start:end].mean()))
                
            min_v, max_v = min(smoothed), max(smoothed)
            if max_v > min_v:
                norm_scores = [(s - min_v) / (max_v - min_v + 1e-8) for s in smoothed]
            else:
                norm_scores = [0.5] * len(smoothed)
                
            return norm_scores
