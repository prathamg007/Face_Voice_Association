import os
from collections import defaultdict
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ==============================================================================
# HUGGING FACE AUTHENTICATION
# ==============================================================================
HUGGING_FACE_TOKEN = "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"


def concatenate_audio_in_folder(folder_path):
    """Concatenate all .wav files in a folder and return combined AudioSegment + metadata."""
    wav_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.wav')])
    if not wav_files:
        return None, []

    combined = AudioSegment.empty()
    file_metadata = []
    for f in wav_files:
        file_path = os.path.join(folder_path, f)
        try:
            audio = AudioSegment.from_wav(file_path)
            combined += audio
            file_metadata.append((f, len(audio)))  # duration in ms
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return combined, file_metadata


def diarize_and_extract_main_speaker(user_id, language, input_base_dir="voices", output_base_dir="voices_diarized"):
    """Concatenate audio, diarize, extract main speaker, and re-split into original filenames."""
    input_path = os.path.join(input_base_dir, user_id, language)
    if not os.path.exists(input_path):
        print(f"Input directory not found for {user_id}/{language}. Skipping.")
        return

    print(f"\n--- Running diarization for '{user_id}' in '{language}' ---")

    try:
        token_to_use = HUGGING_FACE_TOKEN if HUGGING_FACE_TOKEN else True
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token_to_use
        )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        print("Diarization model loaded successfully.")
    except Exception as e:
        print(f"Error loading pyannote pipeline: {e}")
        return

    for root, dirs, _ in os.walk(input_path):
        if not dirs:  # Leaf folder (e.g. voices/id0001/English/h_vamljcIHE)
            code_folder_path = root
            print(f"\nProcessing folder: {code_folder_path}")

            combined_audio, file_metadata = concatenate_audio_in_folder(code_folder_path)
            if not combined_audio or not file_metadata:
                print(f"No valid WAV files found in {code_folder_path}. Skipping.")
                continue

            # Save temporary file
            os.makedirs(output_base_dir, exist_ok=True)
            temp_path = os.path.join(output_base_dir, "temp_combined.wav")
            combined_audio.export(temp_path, format="wav")

            print("Running diarization...")
            diarization = pipeline(temp_path)
            os.remove(temp_path)
            print("Diarization complete.")

            # Identify main speaker
            speaker_durations = defaultdict(float)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_durations[speaker] += turn.end - turn.start

            if not speaker_durations:
                print("No speakers detected. Skipping.")
                continue

            main_speaker = max(speaker_durations, key=speaker_durations.get)
            print(f"Main speaker: {main_speaker} ({speaker_durations[main_speaker]:.2f}s)")

            # Extract all main speaker segments
            main_audio = AudioSegment.empty()
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker == main_speaker:
                    start_ms, end_ms = int(turn.start * 1000), int(turn.end * 1000)
                    main_audio += combined_audio[start_ms:end_ms]

            if len(main_audio) == 0:
                print("Main speaker has no valid segments. Skipping.")
                continue

            # Re-split main speaker audio to same number of files as input
            num_original_files = len(file_metadata)
            total_duration_ms = len(main_audio)
            chunk_length_ms = total_duration_ms // num_original_files
            if chunk_length_ms <= 0:
                print("Audio too short to split. Skipping folder.")
                continue

            original_filenames = [meta[0] for meta in file_metadata]
            rel_path = os.path.relpath(code_folder_path, input_path)
            output_dir = os.path.join(output_base_dir, user_id, language, rel_path)
            os.makedirs(output_dir, exist_ok=True)

            for i in range(num_original_files):
                start_ms = i * chunk_length_ms
                end_ms = (i + 1) * chunk_length_ms if i < num_original_files - 1 else total_duration_ms
                chunk = main_audio[start_ms:end_ms]
                if len(chunk) == 0:
                    continue
                output_filename = original_filenames[i]
                output_path = os.path.join(output_dir, output_filename)
                chunk.export(output_path, format="wav")
                print(f"Saved split chunk {i + 1}/{num_original_files} to '{output_path}'")

    print(f"\nDiarization complete for {user_id} ({language}).")


# ======================================================================
# MAIN EXECUTION
# ======================================================================
if __name__ == "__main__":
    LOWER_ID_LIMIT = 1
    UPPER_ID_LIMIT = 50

    for LANGUAGE_TO_PROCESS in ["English", "German"]:
        print(f"=== Starting diarization for user IDs {LOWER_ID_LIMIT}-{UPPER_ID_LIMIT} ({LANGUAGE_TO_PROCESS}) ===")

        for user_num in range(LOWER_ID_LIMIT, UPPER_ID_LIMIT + 1):
            user_id = f"id{user_num:04d}"
            print(f"\n{'='*20} USER: {user_id} {'='*20}")
            diarize_and_extract_main_speaker(
                user_id,
                LANGUAGE_TO_PROCESS,
                input_base_dir="mavceleb_v3_test/English_test/voice",
                output_base_dir="mavceleb_v3_test/German_test/voice"
            )

        print("\n=== Batch diarization complete ===")
