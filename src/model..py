import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModel(nn.Module):
    def __init__(self, audio_dim=192, image_dim=2048, proj_dim=512, fused_dim=512, num_classes=2, dropout=0.0):
        super().__init__()
        # Projections
        self.a_proj = nn.Linear(audio_dim, proj_dim)
        self.v_proj = nn.Linear(image_dim, proj_dim)
        
        # Bidirectional Cross-Attention
        # batch_first=True expects [Batch, Seq, Feature]
        self.cross1 = nn.MultiheadAttention(proj_dim, 4, batch_first=True)
        self.cross2 = nn.MultiheadAttention(proj_dim, 4, batch_first=True)
        
        # Layer Norms
        self.norm_a = nn.LayerNorm(proj_dim)
        self.norm_v = nn.LayerNorm(proj_dim)
        
        # Fusion Head
        if dropout > 0:
            self.fuse = nn.Sequential(
                nn.Linear(proj_dim, fused_dim), 
                nn.GELU(), 
                nn.Dropout(dropout), 
                nn.LayerNorm(fused_dim)
            )
        else:
            self.fuse = nn.Sequential(
                nn.Linear(proj_dim, fused_dim), 
                nn.GELU(), 
                nn.LayerNorm(fused_dim)
            )
            
        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(self, a, v):
        """
        a: Audio Embeddings [Batch, Audio_Dim]
        v: Visual Embeddings [Batch, Image_Dim]
        """
        # 1. Normalize inputs
        a = F.normalize(a, p=2, dim=1)
        v = F.normalize(v, p=2, dim=1)
        
        # 2. Project to shared dimension and add sequence dim -> [Batch, 1, Proj_Dim]
        a_proj = self.a_proj(a).unsqueeze(1)
        v_proj = self.v_proj(v).unsqueeze(1)
        
        # 3. Cross Attention
        # Query=Audio, Key=Video, Value=Video
        a_att, _ = self.cross1(a_proj, v_proj, v_proj)
        # Query=Video, Key=Audio, Value=Audio
        v_att, _ = self.cross2(v_proj, a_proj, a_proj)
        
        # 4. Residual + Norm
        a_out = self.norm_a(a_proj.squeeze(1) + a_att.squeeze(1))
        v_out = self.norm_v(v_proj.squeeze(1) + v_att.squeeze(1))
        
        # 5. Fusion (Element-wise sum followed by MLP)
        fused = self.fuse(a_out + v_out)
        
        # 6. Classification Logits
        logits = self.classifier(fused)
        
        # Return intermediates for FOP loss
        return logits, a_out, v_out, fused