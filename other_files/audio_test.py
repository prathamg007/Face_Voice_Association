"""
Generate test speaker embeddings using SpeechBrain ECAPA-TDNN.
Outputs ONE CSV containing all test-file embeddings.
Label format: English_<filename>, German_<filename>
"""

import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm

from speechbrain.pretrained import SpeakerRecognition


# ===========================
# CONFIGURATION
# ===========================

TEST_ROOT = "./mavceleb_v3_test"

LANGUAGE_MAP = {
    "English_test": "English",
    "German_test": "German",
}

OUTPUT_CSV = "./V3Embeddings_Test/audio_embeddings_test.csv"

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg")
TARGET_SR = 16000
MIN_SECONDS = 0.5

SB_MODEL = "speechbrain/spkrec-ecapa-voxceleb"


# ===========================
# HELPERS
# ===========================

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_audio_duration(path):
    """Duration in seconds."""
    try:
        info = torchaudio.info(path)
        return info.num_frames / info.sample_rate
    except Exception:
        try:
            wav, sr = torchaudio.load(path)
            return wav.shape[1] / sr
        except Exception:
            return 0.0


# ===========================
# MAIN
# ===========================

def main():

    ensure_dir(OUTPUT_CSV)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading SpeechBrain model: {SB_MODEL}")
    model = SpeakerRecognition.from_hparams(
        source=SB_MODEL,
        run_opts={"device": device}
    )
    print("Model loaded.\n")

    rows = []
    skipped = 0

    # Traverse language folders
    for lang_folder in LANGUAGE_MAP.keys():

        # 🔥 CHANGED FOR YOU — updated folder name
        full_lang_dir = os.path.join(TEST_ROOT, lang_folder, "voice_diarized_cleaned")

        if not os.path.isdir(full_lang_dir):
            print(f"⚠️ Missing directory: {full_lang_dir}")
            continue

        lang_name = LANGUAGE_MAP[lang_folder]

        audio_files = [
            os.path.join(full_lang_dir, f)
            for f in os.listdir(full_lang_dir)
            if f.lower().endswith(AUDIO_EXT)
        ]

        print(f"{lang_folder}: found {len(audio_files)} audio files.")

        for audio_path in tqdm(audio_files, desc=lang_name):

            duration = load_audio_duration(audio_path)
            if duration < MIN_SECONDS:
                skipped += 1
                continue

            try:
                # Load audio
                wav, sr = torchaudio.load(audio_path)

                # Resample
                if sr != TARGET_SR:
                    wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

                wav = wav.to(device)

                # Extract embedding
                emb = model.encode_batch(wav).detach().cpu().numpy().squeeze()

                # Normalize
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm

                filename = os.path.basename(audio_path)
                label = f"{lang_name}_{filename}"

                row = {"label": label}
                row.update({f"feature_{i+1}": float(v) for i, v in enumerate(emb)})
                rows.append(row)

            except Exception as e:
                print(f"⚠️ Failed: {audio_path} -> {e}")
                skipped += 1

    # Save CSV
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Saved embeddings to {OUTPUT_CSV}")
        print(f"Rows: {len(rows)}, Skipped: {skipped}")
    else:
        print("❌ No embeddings produced.")


if __name__ == "__main__":
    main()
