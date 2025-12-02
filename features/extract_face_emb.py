import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()

def load_img(path):
    try:
        img = Image.open(path).convert('RGB')
        img = img.resize((224, 224))
        arr = np.asarray(img, dtype='float32')
        arr = np.expand_dims(arr, axis=0)
        return preprocess_input(arr, version=2)
    except:
        return None

def main():
    args = parse_args()
    
    # Check GPU for TensorFlow
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    
    print("Loading VGGFace2 (ResNet50)...")
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    
    for root, dirs, files in os.walk(args.input_dir):
        imgs = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not imgs: continue
        
        id_name = os.path.basename(root)
        print(f"Processing {id_name}...")
        
        rows = []
        for f in tqdm(imgs):
            path = os.path.join(root, f)
            x = load_img(path)
            if x is None: continue
            
            emb = model.predict(x, verbose=0).flatten()
            
            row = {"label": f"{id_name}_{f}"}
            row.update({f"feature_{i+1}": v for i, v in enumerate(emb)})
            rows.append(row)
            
        if rows:
            out_path = os.path.join(args.output_dir, f"{id_name}.csv")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)

if __name__ == "__main__":
    main()