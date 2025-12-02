import torch
import numpy as np
import random
from torch.utils.data import Dataset

class MultiVectorPairDataset(Dataset):
    """
    Dataset loader that generates pairs of Face and Voice embeddings.
    It expects data dictionaries where keys are IDs and values are lists/arrays of embeddings.
    """
    def __init__(self, aud_dict, img_dict, ids, pairs_per_id, neg_ratio):
        self.aud = aud_dict
        self.img = img_dict
        # Intersection of IDs present in both audio and image data
        self.ids = [i for i in ids if i in aud_dict and i in img_dict]
        self.pairs_per_id = pairs_per_id
        self.neg_ratio = neg_ratio
        self._pairs = []
        self._gen()

    def _gen(self):
        """Generates positive and negative pairs for the epoch."""
        pairs = []
        for id_ in self.ids:
            a_pool = self.aud[id_]
            v_pool = self.img[id_]
            
            if len(a_pool) == 0 or len(v_pool) == 0:
                continue
                
            for _ in range(self.pairs_per_id):
                # Positive Pair (Same Identity)
                a_idx = np.random.randint(0, len(a_pool))
                v_idx = np.random.randint(0, len(v_pool))
                pairs.append((id_, a_idx, id_, v_idx, 1))
                
                # Negative Pairs (Different Identity)
                for _ in range(int(self.neg_ratio)):
                    neg = random.choice(self.ids)
                    while neg == id_:
                        neg = random.choice(self.ids)
                    
                    if len(self.img[neg]) > 0:
                        v2_idx = np.random.randint(0, len(self.img[neg]))
                        pairs.append((id_, a_idx, neg, v2_idx, 0))
                    
        random.shuffle(pairs)
        self._pairs = pairs

    def on_epoch_end(self):
        """Regenerate pairs for the next epoch to increase variety."""
        self._gen()

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        aid, aind, vid, vind, lbl = self._pairs[idx]
        a = self.aud[aid][aind]
        v = self.img[vid][vind]
        return torch.tensor(a).float(), torch.tensor(v).float(), torch.tensor(lbl).long()