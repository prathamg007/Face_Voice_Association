import os
import argparse
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt, stft, istft

def parse_args():
    parser = argparse.ArgumentParser(description="Clean audio files (Bandpass + Wiener)")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sr", type=int, default=16000)
    return parser.parse_args()

def butter_bandpass(lowcut, highcut, fs, order=6):
    return butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')

def wiener_denoise(y, sr, nperseg=512):
    f, t, Zxx = stft(y, sr, nperseg=nperseg)
    power = np.abs(Zxx) ** 2
    noise_psd = np.percentile(power, 10, axis=1, keepdims=True)
    gain = np.maximum(1 - noise_psd / (power + 1e-8), 0.05)
    _, out = istft(Zxx * gain, sr)
    return out.astype(np.float32)

def rms_normalize(y, target_db=-25.0):
    rms = np.sqrt(np.mean(y ** 2)) + 1e-9
    gain = (10 ** (target_db / 20)) / rms
    return np.clip(y * gain, -1.0, 1.0)

def clean_file(in_path, out_path, sr):
    try:
        y, _ = librosa.load(in_path, sr=sr, mono=True)
        if len(y) < 0.3 * sr: return # Skip short files
        
        # 1. Bandpass
        sos = butter_bandpass(80, 7500, sr)
        y = sosfilt(sos, y)
        
        # 2. Wiener
        y = wiener_denoise(y, sr)
        
        # 3. Normalize & Trim
        y = rms_normalize(y)
        y, _ = librosa.effects.trim(y, top_db=25)
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, y, sr)
    except Exception as e:
        print(f"Error cleaning {in_path}: {e}")

if __name__ == "__main__":
    args = parse_args()
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.lower().endswith(('.wav', '.flac', '.mp3')):
                in_file = os.path.join(root, f)
                rel_path = os.path.relpath(in_file, args.input_dir)
                out_file = os.path.join(args.output_dir, rel_path)
                clean_file(in_file, out_file, args.sr)