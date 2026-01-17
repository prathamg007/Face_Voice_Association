import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt, stft, istft

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = "mavceleb_v3_test"
LANGUAGES = ["English", "German"]

TARGET_SR = 16000
HP_CUTOFF = 80         # Hz
LP_CUTOFF = 7500       # Hz
MIN_SEC = 0.3          # ignore files < 0.3 sec


# ============================================================
# FILTERS
# ============================================================

def butter_bandpass(lowcut, highcut, fs, order=6):
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sos

def apply_bandpass(y, sr):
    sos = butter_bandpass(HP_CUTOFF, LP_CUTOFF, sr, order=6)
    return sosfilt(sos, y)


# ============================================================
# WIENER
# ============================================================

def wiener_denoise(y, sr, nperseg=512):
    f, t, Zxx = stft(y, sr, nperseg=nperseg)
    power = np.abs(Zxx) ** 2

    noise_psd = np.percentile(power, 10, axis=1, keepdims=True)
    gain = np.maximum(1 - noise_psd / (power + 1e-8), 0.05)

    Y = Zxx * gain
    _, out = istft(Y, sr)
    return out.astype(np.float32)


# ============================================================
# NORMALIZE + TRIM
# ============================================================

def rms_normalize(y, target_db=-25.0):
    rms = np.sqrt(np.mean(y ** 2)) + 1e-9
    target_rms = 10 ** (target_db / 20)
    gain = target_rms / rms
    y = y * gain
    return np.clip(y, -1.0, 1.0)

def trim_silence(y, sr):
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    return y_trimmed if len(y_trimmed) > 0 else y


# ============================================================
# CLEAN SINGLE FILE
# ============================================================

def clean_audio_file(in_path, out_path):
    try:
        y, sr = librosa.load(in_path, sr=TARGET_SR, mono=True)

        if len(y) < MIN_SEC * sr:
            return False

        y = apply_bandpass(y, sr)
        y = wiener_denoise(y, sr)
        y = rms_normalize(y, target_db=-25)
        y = trim_silence(y, sr)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, y, sr)

        return True

    except Exception as e:
        print(f"ERROR processing {in_path}: {e}")
        return False


# ============================================================
# PROCESS LANGUAGE
# ============================================================

def process_language(lang):
    input_dir = os.path.join(BASE_DIR, f"{lang}_test", "voice_diarized")
    output_dir = os.path.join(BASE_DIR, f"{lang}_test", "voice_diarized_cleaned")

    if not os.path.exists(input_dir):
        print(f"[ERROR] Missing input directory: {input_dir}")
        return

    print(f"\n=== Cleaning {lang} diarized files ===")
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")

    total = 0
    cleaned = 0

    for root, _, files in os.walk(input_dir):
        for f in files:
            if not f.lower().endswith(".wav"):
                continue

            total += 1

            in_path  = os.path.join(root, f)
            rel_path = os.path.relpath(in_path, input_dir)
            out_path = os.path.join(output_dir, rel_path)

            ok = clean_audio_file(in_path, out_path)
            if ok:
                cleaned += 1

            print(f"[{cleaned}/{total}] {rel_path}")

    print(f"Completed {lang}: {cleaned}/{total} files cleaned.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    for lang in LANGUAGES:
        process_language(lang)
