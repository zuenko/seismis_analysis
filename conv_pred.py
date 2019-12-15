import h5py
import numpy as np

#filename = 'random_label3D.h5'
filename = 'our_predictions.h5'

with h5py.File(filename, 'r') as f:
    with h5py.File("preds.h5", 'w') as w:
        preds = f['predictions']
        w['label'] = preds[0]