import os
import argparse
import torch
from collections import defaultdict
from pyannote.audio import Pipeline
from pydub import AudioSegment

def parse_args():
    parser = argparse.ArgumentParser(description="Diarize audio folders")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to raw audio folders")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save diarized audio")
    parser.add_argument("--hf_token", type=str, required=True, help="HuggingFace Token for pyannote")
    return parser.parse_args()

def concatenate_audio_in_folder(folder_path):
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

def process_folder(pipeline, input_path, output_path):
    for root, dirs, _ in os.walk(input_path):
        if not dirs:  # Leaf folder
            print(f"Processing: {root}")
            combined_audio, file_metadata = concatenate_audio_in_folder(root)
            if not combined_audio:
                continue

            # Temp file for diarization
            os.makedirs(output_path, exist_ok=True)
            temp_path = os.path.join(output_path, "temp.wav")
            combined_audio.export(temp_path, format="wav")

            try:
                diarization = pipeline(temp_path)
                
                # Identify main speaker
                speaker_durations = defaultdict(float)
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    speaker_durations[speaker] += turn.end - turn.start

                if not speaker_durations:
                    os.remove(temp_path)
                    continue

                main_speaker = max(speaker_durations, key=speaker_durations.get)
                
                # Extract main speaker
                main_audio = AudioSegment.empty()
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if speaker == main_speaker:
                        start = int(turn.start * 1000)
                        end = int(turn.end * 1000)
                        main_audio += combined_audio[start:end]

                # Re-split
                num_files = len(file_metadata)
                total_ms = len(main_audio)
                if num_files > 0 and total_ms > 0:
                    chunk_len = total_ms // num_files
                    
                    rel_path = os.path.relpath(root, input_path)
                    final_out_dir = os.path.join(output_path, rel_path)
                    os.makedirs(final_out_dir, exist_ok=True)

                    for i in range(num_files):
                        start = i * chunk_len
                        end = (i+1) * chunk_len if i < num_files - 1 else total_ms
                        chunk = main_audio[start:end]
                        if len(chunk) > 0:
                            out_name = file_metadata[i][0]
                            chunk.export(os.path.join(final_out_dir, out_name), format="wav")
            except Exception as e:
                print(f"Error diarizing {root}: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

if __name__ == "__main__":
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading pipeline on {device}...")
    
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=args.hf_token
    )
    pipeline.to(device)
    
    process_folder(pipeline, args.input_dir, args.output_dir)