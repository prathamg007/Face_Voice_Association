"""
Generate ECAPA-TDNN (audio) and VGGFace2 (image) embeddings
for mavceleb_v1_test dataset, stored as CSVs for each language.
"""

import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from speechbrain.inference.classifiers import EncoderClassifier
import tensorflow as tf

# ===========================
# CONFIGURATION
# ===========================
TEST_ROOT = "./mavceleb_v1_test"
AUDIO_OUTPUT = "./ecapa_csv_embeddings_test"
IMAGE_OUTPUT = "./vggface2_csv_embeddings_test"

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg")
IMAGE_EXT = (".jpg", ".jpeg", ".png")

LANGUAGES = ["English", "Urdu"]

# ===========================
# HELPERS
# ===========================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# -------------------
# AUDIO EMBEDDINGS
# -------------------

def generate_audio_embeddings():
    print("\n🎧 Generating ECAPA-TDNN audio embeddings...")

    ensure_dir(AUDIO_OUTPUT)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )

    for lang in LANGUAGES:
        lang_path = os.path.join(TEST_ROOT, "voices", lang)
        if not os.path.isdir(lang_path):
            print(f"⚠️ Skipping {lang} — not found.")
            continue

        ensure_dir(os.path.join(AUDIO_OUTPUT, lang))
        print(f"\nProcessing language: {lang}")

        # group by person ID (e.g. 00000, 00001, …)
        files_by_id = defaultdict(list)
        for root, _, files in os.walk(lang_path):
            for f in files:
                if f.lower().endswith(AUDIO_EXT):
                    person_id = os.path.basename(root)
                    files_by_id[person_id].append(os.path.join(root, f))

        for pid, files in files_by_id.items():
            print(f"  ID {pid} — {len(files)} files")

            rows = []
            for fpath in tqdm(files, desc=f"{lang}-{pid}", unit="file"):
                try:
                    signal, fs = torchaudio.load(fpath)
                    if signal.shape[0] > 1:
                        signal = torch.mean(signal, dim=0, keepdim=True)
                    signal = signal.to(device)
                    with torch.no_grad():
                        emb = classifier.encode_batch(signal)
                    emb = emb.squeeze().cpu().numpy()

                    label = os.path.basename(fpath)
                    row = {"label": f"{pid}_{label}"}
                    row.update({f"feature_{i+1}": v for i, v in enumerate(emb)})
                    rows.append(row)
                except Exception as e:
                    print(f"⚠️ Skipped {fpath}: {e}")

            if rows:
                df = pd.DataFrame(rows)
                out_csv = os.path.join(AUDIO_OUTPUT, lang, f"{pid}.csv")
                df.to_csv(out_csv, index=False)
                print(f"✅ Saved: {out_csv}")

    print("\n🎯 Finished generating audio embeddings.\n")

# -------------------
# IMAGE EMBEDDINGS
# -------------------

def preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize(target_size)
        arr = np.asarray(img, dtype="float32")
        arr = np.expand_dims(arr, axis=0)
        return preprocess_input(arr, version=2)
    except Exception:
        return None


def generate_image_embeddings():
    print("\n🖼️ Generating VGGFace2 image embeddings...")

    ensure_dir(IMAGE_OUTPUT)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) detected: {gpus}")
    else:
        print("⚠️ No GPU detected — will be slow.")

    model = VGGFace(model="resnet50", include_top=False,
                    input_shape=(224, 224, 3), pooling="avg")

    for lang in LANGUAGES:
        lang_path = os.path.join(TEST_ROOT, "faces", lang)
        if not os.path.isdir(lang_path):
            print(f"⚠️ Skipping {lang} — not found.")
            continue

        ensure_dir(os.path.join(IMAGE_OUTPUT, lang))
        print(f"\nProcessing language: {lang}")

        files_by_id = defaultdict(list)
        for root, _, files in os.walk(lang_path):
            for f in files:
                if f.lower().endswith(IMAGE_EXT):
                    person_id = os.path.basename(root)
                    files_by_id[person_id].append(os.path.join(root, f))

        for pid, files in files_by_id.items():
            print(f"  ID {pid} — {len(files)} images")
            rows = []
            for fpath in tqdm(files, desc=f"{lang}-{pid}", unit="img"):
                processed = preprocess_image(fpath)
                if processed is not None:
                    emb = model.predict(processed, verbose=0).flatten()
                    label = os.path.basename(fpath)
                    row = {"label": f"{pid}_{label}"}
                    row.update({f"feature_{i+1}": v for i, v in enumerate(emb)})
                    rows.append(row)
                else:
                    print(f"⚠️ Skipped {fpath}")

            if rows:
                df = pd.DataFrame(rows)
                out_csv = os.path.join(IMAGE_OUTPUT, lang, f"{pid}.csv")
                df.to_csv(out_csv, index=False)
                print(f"✅ Saved: {out_csv}")

    print("\n🎯 Finished generating image embeddings.\n")


# ===========================
# MAIN
# ===========================
if __name__ == "__main__":
    generate_audio_embeddings()
    generate_image_embeddings()
