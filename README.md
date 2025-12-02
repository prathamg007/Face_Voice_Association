# Face_Voice_Association
# CAFNet: Cross-Attentive Face-Voice Fusion Network

![CAFNet Architecture](assets/cafnet_arch.png)

This repository contains the official PyTorch implementation of **CAFNet** (Cross-Attentive Face-Voice Fusion Network).

**CAFNet** is a multimodal fusion framework designed for robust face-voice association and verification, particularly in challenging multilingual environments. It introduces a bidirectional cross-attention mechanism to align speaker embeddings (ECAPA-TDNN) and face embeddings (VGGFace2), optimized with a Fusion Orthogonal Projection (FOP) loss to enforce identity separation in the joint latent space.

## 📄 Abstract
Face-voice association seeks to verify whether a face image and a speech segment belong to the same identity. Real-world audio-visual data are noisy, multilingual, and frequently multi-speaker; robustness to these conditions is critical for deployed systems. 

We present CAFNet, which combines a practical audio preprocessing pipeline (diarization + cleaning) with strong modality-specific embeddings and a bidirectional cross-attention fusion head. We report significant improvements over baseline methods on the MAV-Celeb dataset, demonstrating that the architecture consistently improves cross-modal alignment and is compact enough to train on limited multimodal data.

## ✨ Key Features
- **Advanced Fusion**: Bidirectional Cross-Attention (Audio queries Face / Face queries Audio) followed by Gated Fusion.
- **State-of-the-Art Backbones**: 
  - **Audio**: ECAPA-TDNN (SpeechBrain)
  - **Visual**: VGGFace2 (ResNet50)
- **Robust Training**: Utilizes Fusion Orthogonal Projection (FOP) loss to enforce orthogonality between different identities.
- **Preprocessing Pipeline**: Integrated support for Speaker Diarization (`pyannote.audio`) and signal cleaning (Wiener filtering).

## 📂 Repository Structure
```text
CAFNet/
├── assets/             # Architecture diagrams
├── features/           # Scripts for extracting embeddings (ECAPA/VGGFace2)
├── preprocessing/      # Audio cleaning and diarization tools
├── src/                # Core model definitions, datasets, and loss functions
├── checkpoints/        # Saved models (created during training)
├── train.py            # Main training script
├── inference.py        # Evaluation/Testing script
└── requirements.txt    # Python dependencies
```

## Installation
1. Clone the repository
   git clone [https://github.com/YOUR_USERNAME/CAFNet.git](https://github.com/YOUR_USERNAME/CAFNet.git)
   cd CAFNet
2. Install dependencies: It is recommended to use a virtual environment (Conda or venv).
   pip install -r requirements.txt

## Data Preparation
This repository is designed to work with any audio-visual dataset organized by identity.
Expected Directory Structure:
MyDataset/
├── audio/
│   ├── id0001/
│   │   ├── session1.wav
│   │   └── ...
│   └── id0002/
│       └── ...
└── faces/
    ├── id0001/
    │   ├── img1.jpg
    │   └── ...
    └── id0002/
        └── ...

## Usage
1. Audio Preprocessing
Real-world audio often contains silence or background noise. We provide a pipeline to diarize (isolate the speaker) and clean the audio.

# Step 1: Diarize (Requires HuggingFace Token for pyannote.audio)
python preprocessing/diarize_audio.py \
  --input_dir dataset/raw_audio \
  --output_dir dataset/diarized_audio \
  --hf_token YOUR_HF_TOKEN

# Step 2: Clean (Bandpass + Wiener Filter)
python preprocessing/clean_audio.py \
  --input_dir dataset/diarized_audio \
  --output_dir dataset/cleaned_audio


2. Feature Extraction
Convert raw media files into embedding CSVs for efficient training.

# Extract Audio Embeddings (ECAPA-TDNN)
python features/extract_audio_emb.py \
  --input_dir dataset/cleaned_audio \
  --output_dir embeddings/audio

# Extract Face Embeddings (VGGFace2)
python features/extract_face_emb.py \
  --input_dir dataset/faces \
  --output_dir embeddings/faces

3. Training
Train the CAFNet model using the generated embeddings.

python train.py \
  --audio_emb_dir embeddings/audio \
  --face_emb_dir embeddings/faces \
  --save_dir checkpoints \
  --epochs 15 \
  --batch_size 64 \
  --lr 0.0002 \
  --lambda_fop 0.02


4. Inference / Evaluation
Evaluate the model on a list of test pairs.

python inference.py \
  --model_path checkpoints/best_model.pt \
  --audio_emb_dir embeddings/audio \
  --face_emb_dir embeddings/faces \
  --pairs_file test_pairs.txt \
  --output_file results.txt


5. Format for test_pairs.txt:

pair_id_1  audio_filename_1  face_filename_1
pair_id_2  audio_filename_2  face_filename_2


## Results
The model was evaluated on the MAV-Celeb dataset (English-Urdu and English-German splits) under the Unseen-Unheard protocol
Split,Method,EER (%)
English-Urdu,WavLM Baseline,27.02
English-Urdu,CAFNet (Ours),26.31
English-German,Baseline (FOP),40.20
English-German,CAFNet (Ours),39.46


## Authors
Lakshya Gupta (IIT Kanpur)

Pratham Gupta (IIT Kanpur)

Shubham Yadav (IIT Kanpur)

