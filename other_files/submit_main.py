#!/usr/bin/env python3
"""
SPCup FINAL Submission Script
- Uses last-4-digit matching for face embeddings
- Loads face_english.csv and face_german.csv
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from zipfile import ZipFile

# ---------------- PATHS ----------------
AUDIO_CSV  = "V3Embeddings_Test/audio_embeddings_test.csv"

FACE_EN_CSV = "V3Embeddings_Test/face_english.csv"
FACE_DE_CSV = "V3Embeddings_Test/face_german.csv"

ENGLISH_TXT = "V3Embeddings_Test/English_test.txt"
GERMAN_TXT  = "V3Embeddings_Test/German_test.txt"

CKPT_ENG = "best_eng_shubh.pt"
CKPT_GER = "best_german_shubh.pt"

OUT_DIR = "V3Embeddings_Test/submission"

OUT_EN_HEARD     = f"{OUT_DIR}/sub_score_English_heard.txt"
OUT_EN_UNHEARD   = f"{OUT_DIR}/sub_score_English_unheard.txt"
OUT_GER_HEARD    = f"{OUT_DIR}/sub_score_German_heard.txt"
OUT_GER_UNHEARD  = f"{OUT_DIR}/sub_score_German_unheard.txt"

ZIP_OUT = f"{OUT_DIR}/submission_scores.zip"

# ---------------- MODEL DIMS ----------------
AUDIO_DIM = 192
IMAGE_DIM = 2048
PROJ_DIM  = 512
FUSED_DIM = 512
NUM_CLASSES = 2

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- MODEL ----------------
class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_proj = nn.Linear(AUDIO_DIM, PROJ_DIM)
        self.v_proj = nn.Linear(IMAGE_DIM, PROJ_DIM)

        self.cross1 = nn.MultiheadAttention(PROJ_DIM, 4, batch_first=True)
        self.cross2 = nn.MultiheadAttention(PROJ_DIM, 4, batch_first=True)

        self.norm_a = nn.LayerNorm(PROJ_DIM)
        self.norm_v = nn.LayerNorm(PROJ_DIM)

        self.fuse = nn.Sequential(
            nn.Linear(PROJ_DIM, FUSED_DIM),
            nn.GELU(),
            nn.LayerNorm(FUSED_DIM),
        )

        self.classifier = nn.Linear(FUSED_DIM, NUM_CLASSES)

    def forward(self, a, v):
        a = F.normalize(a, dim=1)
        v = F.normalize(v, dim=1)

        a_proj = self.a_proj(a).unsqueeze(1)
        v_proj = self.v_proj(v).unsqueeze(1)

        a_att, _ = self.cross1(a_proj, v_proj, v_proj)
        v_att, _ = self.cross2(v_proj, a_proj, a_proj)

        a_out = self.norm_a(a_proj.squeeze(1) + a_att.squeeze(1))
        v_out = self.norm_v(v_proj.squeeze(1) + v_att.squeeze(1))

        fused = self.fuse(a_out + v_out)
        logits = self.classifier(fused)
        return logits


# ---------------- HELPERS ----------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def extract_4digit_id(path):
    """Extract last 4-digit ID from filename/path."""
    fname = os.path.basename(str(path))
    m = re.search(r"(\d{4})\D*$", fname)
    if m:
        return m.group(1)  # string like "0037"
    return None


def load_audio_embeddings():
    df = pd.read_csv(AUDIO_CSV)
    feat_cols = [c for c in df.columns if c.startswith("feature_")]

    emb = {}
    for _, row in df.iterrows():
        key = row["label"]
        vec = row[feat_cols].to_numpy(np.float32)
        emb[key] = vec

    print(f"Loaded {len(emb)} audio embeddings.")
    return emb


# FACE EMBEDDING LOADER (last 4-digit matching)
def load_image_embeddings_csv():
    img_map = {"English_test": {}, "German_test": {}}

    def load_one(path_csv, lang):
        df = pd.read_csv(path_csv)
        feat_cols = [c for c in df.columns if c.startswith("feature_")]

        for _, row in df.iterrows():
            id4 = extract_4digit_id(row["label"])
            if id4 is None:
                continue
            img_map[lang][id4] = row[feat_cols].to_numpy(np.float32)

    load_one(FACE_EN_CSV, "English_test")
    load_one(FACE_DE_CSV, "German_test")

    print(
        f"Loaded {len(img_map['English_test'])} English faces, "
        f"{len(img_map['German_test'])} German faces."
    )
    return img_map


def parse_list(path):
    out = []
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 3:
                out.append((p[0], p[1], p[2]))
    return out


# UPDATED run_pairs() — match by last 4-digit ID
def run_pairs(model, audio_map, image_map, entries, audio_prefix, img_dict):
    res = []
    missing_audio = 0
    missing_image = 0
    processed = 0

    img_block = image_map[img_dict]

    print(f"\n{audio_prefix} → {img_dict}")

    for uid, a_path, i_path in tqdm(entries):
        processed += 1

        # AUDIO
        a_file = os.path.basename(a_path)
        akey = f"{audio_prefix}_{a_file}"
        if akey not in audio_map:
            missing_audio += 1
            continue

        # IMAGE — match by last 4-digit ID
        id4 = extract_4digit_id(i_path)
        if id4 not in img_block:
            missing_image += 1
            continue

        avec = torch.tensor(audio_map[akey]).float().unsqueeze(0).to(device)
        ivec = torch.tensor(img_block[id4]).float().unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(avec, ivec)
            prob = F.softmax(logits, dim=1)[0, 1].item()

        res.append((uid, prob))

    print(f"Processed pairs:    {processed}")
    print(f"Missing audio:      {missing_audio}")
    print(f"Missing images:     {missing_image}")
    print(f"Successful pairs:   {len(res)}")

    return res


# ---------------- MAIN ----------------
def main():
    ensure_dir(OUT_DIR)

    print("Loading embeddings...")
    audio_map = load_audio_embeddings()
    image_map = load_image_embeddings_csv()

    print("Loading test pairs...")
    en_pairs = parse_list(ENGLISH_TXT)
    de_pairs = parse_list(GERMAN_TXT)

    print("Loading English model...")
    eng_model = FusionModel().to(device)
    eng_model.load_state_dict(torch.load(CKPT_ENG, map_location=device))
    eng_model.eval()

    print("Loading German model...")
    ger_model = FusionModel().to(device)
    ger_model.load_state_dict(torch.load(CKPT_GER, map_location=device))
    ger_model.eval()

    print("\n=== RUNNING SCORING ===")

    # 1) English heard → English model
    en_heard = run_pairs(
        eng_model, audio_map, image_map,
        en_pairs, "English", "English_test"
    )

    # 2) English unheard → German model
    en_unheard = run_pairs(
        ger_model, audio_map, image_map,
        en_pairs, "English", "English_test"
    )

    # 3) German heard → German model
    ger_heard = run_pairs(
        ger_model, audio_map, image_map,
        de_pairs, "German", "German_test"
    )

    # 4) German unheard → English model
    ger_unheard = run_pairs(
        eng_model, audio_map, image_map,
        de_pairs, "German", "German_test"
    )

    # Write all outputs
    def write(path, data):
        with open(path, "w") as f:
            for uid, sc in data:
                f.write(f"{uid} {sc:.6f}\n")

    write(OUT_EN_HEARD,   en_heard)
    write(OUT_EN_UNHEARD, en_unheard)
    write(OUT_GER_HEARD,  ger_heard)
    write(OUT_GER_UNHEARD, ger_unheard)

    # Zip everything
    with ZipFile(ZIP_OUT, "w") as z:
        z.write(OUT_EN_HEARD,    arcname="sub_score_English_heard.txt")
        z.write(OUT_EN_UNHEARD,  arcname="sub_score_English_unheard.txt")
        z.write(OUT_GER_HEARD,   arcname="sub_score_German_heard.txt")
        z.write(OUT_GER_UNHEARD, arcname="sub_score_German_unheard.txt")

    print("\n🎉 DONE — submission_scores.zip generated!")


if __name__ == "__main__":
    main()
