import os
import argparse
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve

from src.model import FusionModel
from src.dataset import MultiVectorPairDataset
from src.losses import fop_loss, nt_xent_loss

def parse_args():
    parser = argparse.ArgumentParser(description="Train CAFNet")
    parser.add_argument("--audio_emb_dir", type=str, required=True, help="Dir containing audio CSVs")
    parser.add_argument("--face_emb_dir", type=str, required=True, help="Dir containing face CSVs")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda_fop", type=float, default=0.02)
    parser.add_argument("--pairs_per_id", type=int, default=150)
    return parser.parse_args()

def load_data(root_dir):
    data = {}
    files = glob.glob(os.path.join(root_dir, "*.csv"))
    print(f"Loading {len(files)} CSVs from {root_dir}...")
    for f in files:
        # ID is assumed to be filename (id0001.csv -> id0001)
        uid = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f)
        feat_cols = [c for c in df.columns if c.startswith("feature_")]
        data[uid] = df[feat_cols].values.astype(np.float32)
    return data

def compute_eer(labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    return fpr[idx]

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Load Data
    aud_data = load_data(args.audio_emb_dir)
    img_data = load_data(args.face_emb_dir)
    
    # Find common IDs
    common_ids = list(set(aud_data.keys()) & set(img_data.keys()))
    if not common_ids:
        print("Error: No common IDs found between Audio and Face embeddings.")
        return
    print(f"Found {len(common_ids)} common identities.")
    
    # 2. Setup Dataset
    dataset = MultiVectorPairDataset(aud_data, img_data, common_ids, args.pairs_per_id, neg_ratio=1.5)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 3. Model
    model = FusionModel(audio_dim=192, image_dim=2048, proj_dim=512, fused_dim=512).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 4. Training Loop
    print("\nStarting Training...")
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        dataset.on_epoch_end() # Regenerate pairs
        
        running_loss = 0.0
        
        for a, v, labels in loader:
            a, v, labels = a.to(device), v.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, a_out, v_out, fused = model(a, v)
            
            # Loss Calculation
            ce_loss = F.cross_entropy(logits, labels)
            fop = fop_loss(a_out, v_out, fused)
            
            total_loss = ce_loss + (args.lambda_fop * fop)
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(args.save_dir, "last_model.pt"))
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))
            print("  [Saved Best Model]")

    print("Training Complete.")

if __name__ == "__main__":
    main()