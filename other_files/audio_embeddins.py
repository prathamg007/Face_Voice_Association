"""
Generate TRAIN speaker embeddings using SpeechBrain ECAPA-TDNN.
Matches EXACTLY the format of the TEST embeddings.
"""

import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from speechbrain.pretrained import SpeakerRecognition

# ===========================
# CONFIGURATION
# ===========================

TRAIN_ROOT = "./mavceleb_v3_tran/voices_diarized_cleaned"
OUTPUT_DIR = "./V3Embeddings/SpeechBrain_EcapaTDNNembeddings_diarized_cleaned"

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg")
TARGET_SR = 16000
MIN_SECONDS = 0.5

SB_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

LANG_MAP = {
    "english": "English",
    "german": "German",
}

# ===========================
# HELPERS
# ===========================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_id_audio(train_root):
    """Return dict: (lang, id) -> list of file paths."""
    groups = defaultdict(list)

    for root, _, files in os.walk(train_root):
        parts = [p.lower() for p in root.split(os.sep)]

        lang = None
        for p in parts:
            if p in LANG_MAP:
                lang = LANG_MAP[p]
                break

        id_folder = None
        for p in parts:
            if p.startswith("id"):
                id_folder = p
                break

        if lang and id_folder:
            for f in files:
                if f.lower().endswith(AUDIO_EXT):
                    groups[(lang, id_folder)].append(os.path.join(root, f))

    return groups


def load_audio_if_ok(path):
    """Return waveform @ 16k if valid; otherwise None."""
    try:
        info = torchaudio.info(path)
        duration = info.num_frames / info.sample_rate
        if duration < MIN_SECONDS:
            return None

        wav, sr = torchaudio.load(path)
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        return wav
    except:
        return None


# ===========================
# MAIN
# ===========================

def main():

    ensure_dir(OUTPUT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading SpeechBrain ECAPA model: {SB_MODEL}")
    model = SpeakerRecognition.from_hparams(
        source=SB_MODEL,
        run_opts={"device": device}
    )
    print("Model loaded.\n")

    print("Scanning train dataset...")
    groups = find_id_audio(TRAIN_ROOT)
    print(f"Found {len(groups)} speakers.\n")

    for (lang, id_folder), file_list in groups.items():

        print(f"Processing {lang}/{id_folder} ({len(file_list)} files)...")

        lang_dir = os.path.join(OUTPUT_DIR, lang)
        ensure_dir(lang_dir)

        output_csv = os.path.join(lang_dir, f"{id_folder}.csv")
        rows = []

        for audio_path in tqdm(file_list):

            wav = load_audio_if_ok(audio_path)
            if wav is None:
                continue

            wav = wav.to(device)

            # Extract embedding
            emb = model.encode_batch(wav).detach().cpu().numpy().squeeze()

            # Normalize
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            filename = os.path.basename(audio_path)
            label = f"{lang}_{filename}"

            row = {"label": label}
            for i, v in enumerate(emb):
                row[f"feature_{i+1}"] = float(v)

            rows.append(row)

        # Save speaker CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_csv, index=False)
            print(f"Saved: {output_csv} ({len(rows)} rows)")
        else:
            print(f"No valid audio for {lang}/{id_folder}")

    print("\n🎉 ALL TRAIN AUDIO EMBEDDINGS GENERATED!")


if __name__ == "__main__":
    main()
