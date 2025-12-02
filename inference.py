import os
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.model import FusionModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--audio_emb_dir", type=str, required=True)
    parser.add_argument("--face_emb_dir", type=str, required=True)
    parser.add_argument("--pairs_file", type=str, required=True, help="Txt file with lines: ID AudioPath FacePath")
    parser.add_argument("--output_file", type=str, default="scores.txt")
    return parser.parse_args()

def load_embeddings_flat(root_dir):
    """Loads all embeddings into a single dict keyed by 'Label' (filename)"""
    emb_map = {}
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".csv"):
                df = pd.read_csv(os.path.join(root, f))
                feat_cols = [c for c in df.columns if c.startswith("feature_")]
                for _, row in df.iterrows():
                    # Key is the label column in CSV (usually "folder_filename")
                    emb_map[row['label']] = row[feat_cols].values.astype(np.float32)
    return emb_map

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model
    print("Loading Model...")
    model = FusionModel(audio_dim=192, image_dim=2048).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # 2. Load Embeddings
    print("Loading Audio Embeddings...")
    aud_map = load_embeddings_flat(args.audio_emb_dir)
    print("Loading Face Embeddings...")
    img_map = load_embeddings_flat(args.face_emb_dir)
    
    # 3. Process Pairs
    print(f"Reading pairs from {args.pairs_file}...")
    results = []
    
    with open(args.pairs_file, 'r') as f:
        lines = f.readlines()
        
    for line in tqdm(lines):
        parts = line.strip().split()
        if len(parts) < 3: continue
        
        pair_id, aud_key, img_key = parts[0], parts[1], parts[2]
        
        # Logic to match filename in text file to label in CSV
        # User might need to adjust logic here depending on exact string format
        # Current assumption: Input file contains paths/keys that match CSV 'label' column
        
        # Simple filename matching fallback
        aud_label = os.path.basename(aud_key) 
        img_label = os.path.basename(img_key)
        
        # Try finding key containing filename (partial match)
        # In optimized production, exact keys are preferred.
        
        a_vec = None
        v_vec = None
        
        # Exact match attempt (if keys in txt match keys in CSV)
        if aud_key in aud_map: a_vec = aud_map[aud_key]
        if img_key in img_map: v_vec = img_map[img_key]
        
        # Fallback: Search by simple filename if full key fails
        if a_vec is None:
            for k, v in aud_map.items():
                if aud_label in k: 
                    a_vec = v
                    break
        
        if v_vec is None:
            for k, v in img_map.items():
                if img_label in k: 
                    v_vec = v
                    break
                    
        if a_vec is not None and v_vec is not None:
            a_tensor = torch.tensor(a_vec).float().unsqueeze(0).to(device)
            v_tensor = torch.tensor(v_vec).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, _, _, _ = model(a_tensor, v_tensor)
                score = F.softmax(logits, dim=1)[0, 1].item() # Prob of "Same" class
                
            results.append(f"{pair_id} {score:.6f}")
        else:
            # Missing embedding handling
            results.append(f"{pair_id} 0.000000")

    # 4. Save
    with open(args.output_file, 'w') as f:
        f.write("\n".join(results))
    print(f"Scores saved to {args.output_file}")

if __name__ == "__main__":
    main()