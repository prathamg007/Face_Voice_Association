"""
Generate speaker embeddings using SpeechBrain ECAPA-TDNN.
One CSV per ID per language, same format as before.
"""

import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# SpeechBrain ECAPA-TDNN model
from speechbrain.pretrained import EncoderClassifier

# ===========================
# CONFIGURATION
# ===========================

DATASET_ROOT = "./mavceleb_v3_tran/voices_diarized_cleaned"
OUTPUT_DIR = "./V3Embeddings/SpeechBrain_EcapaTDNNembeddings_diarized_cleaned"

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg")
LANG_FOLDERS = {"english", "german"}

MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"   # Pretrained ECAPA TDNN

BATCH_SIZE = 2
TARGET_SR = 16000
MIN_SECONDS = 0.5


# ===========================
# HELPERS
# ===========================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_groups(dataset_root):
    """Find (id, language) groups and collect all audio files for that pair."""
    files_by_group = defaultdict(list)

    for root, _, files in os.walk(dataset_root):
        parts = [p.lower() for p in root.split(os.sep)]

        lang_folder = next((p for p in parts if p in LANG_FOLDERS), None)
        id_folder = next((p for p in parts if p.startswith("id")), None)

        if id_folder and lang_folder:
            for f in files:
                if f.lower().endswith(AUDIO_EXT):
                    full_path = os.path.join(root, f)
                    files_by_group[(id_folder, lang_folder.capitalize())].append(full_path)

    return files_by_group


def load_and_resample(path, target_sr=TARGET_SR):
    try:
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        return wav.squeeze(0), target_sr
    except Exception:
        return None, None


# ===========================
# MAIN
# ===========================

def main():
    if not os.path.isdir(DATASET_ROOT):
        print(f"❌ Dataset directory not found at '{DATASET_ROOT}'")
        return

    ensure_dir(OUTPUT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load SpeechBrain ECAPA model
    print(f"Loading SpeechBrain ECAPA model: {MODEL_NAME}")
    try:
        classifier = EncoderClassifier.from_hparams(
            source=MODEL_NAME,
            run_opts={"device": device},
            savedir=os.path.join(OUTPUT_DIR, "ecapa_model_cache")
        )
        print("✅ SpeechBrain ECAPA loaded successfully.")
    except Exception as e:
        print("❌ Error loading SpeechBrain ECAPA model:", e)
        return

    print("\n🔍 Scanning dataset...")
    files_by_group = find_groups(DATASET_ROOT)
    if not files_by_group:
        print("❌ No audio files found.")
        return
    print(f"Found {len(files_by_group)} (ID, language) groups.\n")

    # LOOP OVER SPEAKERS + LANGUAGES
    for (id_folder, lang_folder), audio_files in files_by_group.items():
        print(f"\nProcessing {id_folder}/{lang_folder} ({len(audio_files)} files)...")

        lang_output_dir = os.path.join(OUTPUT_DIR, lang_folder)
        ensure_dir(lang_output_dir)
        output_csv = os.path.join(lang_output_dir, f"{id_folder}.csv")

        rows = []
        skipped = 0

        # Batch processing
        for i in tqdm(range(0, len(audio_files), BATCH_SIZE),
                      desc=f"{id_folder}-{lang_folder}",
                      unit="batch"):

            batch_paths = audio_files[i:i + BATCH_SIZE]
            batch_wavs = []
            batch_labels = []

            # LOAD WAVS
            for audio_path in batch_paths:
                wav, sr = load_and_resample(audio_path)
                if wav is None or len(wav) < TARGET_SR * MIN_SECONDS:
                    skipped += 1
                    continue

                parent_folder = os.path.basename(os.path.dirname(audio_path))
                filename = os.path.basename(audio_path)
                label = f"{parent_folder}_{filename}"

                batch_wavs.append(wav)
                batch_labels.append(label)

            if not batch_wavs:
                continue

            # PAD WAVS TO SAME LENGTH
            max_len = max([len(w) for w in batch_wavs])
            padded = torch.stack([torch.nn.functional.pad(w, (0, max_len - len(w))) for w in batch_wavs])

            try:
                # RUN ECAPA EMBEDDING EXTRACTION
                with torch.no_grad():
                    emb = classifier.encode_batch(padded.to(device))

                # Squeeze out length-1 dimensions
                emb = emb.squeeze()

                # Handle 3D case [B, 1, D]
                if emb.ndim == 3:
                    emb = emb[:, 0, :]

                # If single embedding accidentally collapsed to 1D -> re-expand
                if emb.ndim == 1:
                    emb = emb.unsqueeze(0)

                # Final check
                emb = emb.float()
                assert emb.ndim == 2, f"Embedding bad shape: {emb.shape}"

                # L2 normalize
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)

                emb_np = emb.cpu().numpy()

                # Save rows
                for label, vec in zip(batch_labels, emb_np):
                    row = {"label": label}
                    row.update({f"feature_{i+1}": float(v) for i, v in enumerate(vec)})
                    rows.append(row)

            except Exception as e:
                print("⚠️ Batch failed:", e)
                skipped += len(batch_wavs)

        # SAVE CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_csv, index=False)
            print(f"✅ Saved: {output_csv}   (rows={len(rows)}, skipped={skipped})")
        else:
            print(f"⚠️ No embeddings for {id_folder}/{lang_folder}")

    print("\n🎯 All SpeechBrain ECAPA embeddings generated successfully!\n")


# Small helper used in original script
def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


if __name__ == "__main__":
    main()
