import os
import argparse
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from speechbrain.pretrained import EncoderClassifier

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading ECAPA-TDNN...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/ecapa",
        run_opts={"device": device}
    )

    # Process by ID folder
    for root, dirs, files in os.walk(args.input_dir):
        audio_files = [f for f in files if f.endswith(('.wav', '.flac', '.mp3'))]
        if not audio_files: continue
        
        # Assume folder name is ID
        id_name = os.path.basename(root)
        print(f"Processing {id_name}...")
        
        rows = []
        for f in audio_files:
            path = os.path.join(root, f)
            try:
                sig, sr = torchaudio.load(path)
                # Resample to 16k if needed
                if sr != 16000:
                    sig = torchaudio.transforms.Resample(sr, 16000)(sig)
                
                sig = sig.to(device)
                
                # Encode
                with torch.no_grad():
                    emb = classifier.encode_batch(sig).squeeze().cpu().numpy()
                
                row = {"label": f"{id_name}_{f}"}
                row.update({f"feature_{i+1}": v for i, v in enumerate(emb)})
                rows.append(row)
            except Exception as e:
                print(f"Skipping {f}: {e}")
        
        if rows:
            out_path = os.path.join(args.output_dir, f"{id_name}.csv")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)

if __name__ == "__main__":
    main()