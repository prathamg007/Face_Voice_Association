import os
import numpy as np
import tensorflow as tf
from PIL import Image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
# This is the base path to your 'faces' directory
DATASET_ROOT = r"C:\SP_CUP\mavceleb_v1_train\faces"

# The file where the final embeddings will be saved
OUTPUT_FILE = 'vggface2_embeddings_first_10_folders_english_only.csv'  # <-- CHANGED


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads and preprocesses an image for VGGFace2 model.
    (This function is unchanged)
    """
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(target_size)
        pixels = np.asarray(img, dtype='float32')
        sample = np.expand_dims(pixels, axis=0)
        processed_sample = preprocess_input(sample, version=2)  # version=2 for ResNet50
        return processed_sample
    except Exception as e:
        return None


def main():
    """
    Main function to walk through the dataset, generate, and save embeddings.
    """
    if not os.path.isdir(DATASET_ROOT):
        print(f"Error: Base dataset directory not found at '{DATASET_ROOT}'")
        print("Please update the DATASET_ROOT variable in the script.")
        return

    # --- Model Loading ---
    print("Loading VGGFace2 (ResNet50) model...")
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    print("Model loaded successfully.")

    data_for_df = []
    skipped_files = []

    # --- Find the first 10 existing subfolders ---
    print(f"\nScanning '{DATASET_ROOT}' for subfolders...")
    try:
        all_entries = os.listdir(DATASET_ROOT)
        all_dirs = [d for d in all_entries if os.path.isdir(os.path.join(DATASET_ROOT, d))]
        all_dirs.sort()
        target_folders = all_dirs[:10]

        if not target_folders:
            print("Error: No subfolders found in the dataset directory.")
            return

    except Exception as e:
        print(f"Error scanning for directories: {e}")
        return

    print(f"Found {len(target_folders)} folders to process:")
    for i, folder_name in enumerate(target_folders):
        print(f"  {i + 1}. {folder_name}")

    # --- Collect image paths from the 'english' subfolder ONLY ---
    print("\nFinding all images in the 'english' subfolder of each target folder...")
    image_paths_to_process = []

    # Loop through the target ID folders (e.g., 'id0001', 'id0002', ...)
    for folder_name in target_folders:

        # <-- NEW: Build the path directly to the 'english' folder
        english_folder_path = os.path.join(DATASET_ROOT, folder_name, 'english')

        # Check if the 'english' folder actually exists before trying to walk it
        if not os.path.isdir(english_folder_path):
            print(f"Warning: 'english' folder not found at '{english_folder_path}', skipping.")
            continue  # Skip to the next 'idXXXX' folder

        # Now, walk *only* this specific 'english' folder and its subdirectories
        for root, dirs, files in os.walk(english_folder_path):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths_to_process.append(os.path.join(root, filename))
    # --- End of changed section ---

    if not image_paths_to_process:
        print("No images found in the 'english' subfolders of the target directories.")
        return

    print(f"Found {len(image_paths_to_process)} images to process.")

    # --- Process images with a progress bar (Unchanged) ---
    for image_path in tqdm(image_paths_to_process, desc="Generating Embeddings", unit="image"):
        processed_image = preprocess_image(image_path)

        if processed_image is not None:
            embedding = model.predict(processed_image, verbose=0).flatten()
            row = {'image_path': image_path}
            emb_dict = {f'emb_{i}': val for i, val in enumerate(embedding)}
            row.update(emb_dict)
            data_for_df.append(row)
        else:
            skipped_files.append(image_path)

    # --- Final Summary (Unchanged) ---
    print(f"\nFinished processing. Generated embeddings for {len(data_for_df)} images.")

    if skipped_files:
        print(f"\n--- Skipped Files Report ---")
        print(f"Could not process {len(skipped_files)} files.")
        for i, f_path in enumerate(skipped_files[:10]):
            print(f"  {i + 1}. {f_path}")
        if len(skipped_files) > 10:
            print("  ...")
        print("--------------------------")

    # --- Saving Embeddings (Unchanged) ---
    if data_for_df:
        print("\nConverting data to DataFrame...")
        final_df = pd.DataFrame(data_for_df)

        print(f"\nSaving embeddings to '{OUTPUT_FILE}'...")
        final_df.to_csv(OUTPUT_FILE, index=False)

        print("Embeddings saved successfully.")
    else:
        print("No embeddings were generated. Please check your dataset directory and image files.")


if __name__ == '__main__':
    # (This section is unchanged)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU(s) detected: {gpus}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            main()
        except RuntimeError as e:
            print(e)
    else:
        print("Warning: No GPU detected. This process will be very slow on a CPU.")
        main()