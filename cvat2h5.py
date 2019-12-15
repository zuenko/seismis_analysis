import h5py, cv2, os, re
import numpy as np
from tqdm import tqdm

ANN_DIR = 'ann'
labels_map = dict()

CUBE_FILE = 'Parihaka_PSTM_near_stack.sgy'

y1,x1 = 281, 315
y2,x2 = 1008, 1203

V_SHAPE = (922, 1126, 1168)
# TARGET_SHAPE = (128, 128)
# TARGET_SHAPE = V_SHAPE[1:]

with open(f"{ANN_DIR}/colormap.txt", 'r') as f:
    for line in f.readlines():
        for ret in re.findall(r'(.*?)\s*:\s*(\d+),\s*(\d+),\s*(\d+)', line):
            name, r,g,b = ret
            r,g,b = int(r), int(g), int(b)
            labels_map[name] = (r,g,b)

RGB_TO_USE = labels_map['background']

masks_all = dict()
imgs_all = []
masks_annotated_count = 0
for filename in tqdm(list(os.listdir(ANN_DIR))):
    #print(filename)
    ret = re.match(r'slice-(\d+).png', filename)
    if ret:
        masks_annotated_count += 1
        slice_num = int(ret.group(1))
        mask = cv2.imread(f"{ANN_DIR}/{filename}")
        mask = mask[y1:y2,x1:x2]
        mask = cv2.resize(mask, (V_SHAPE[1], V_SHAPE[0]))
        mask = np.any(mask != RGB_TO_USE, axis=-1).astype(np.int64)
        if np.any(mask != 0):
            masks_all[slice_num] = mask

print("non_zero_masks_count", len(masks_all), "out of", masks_annotated_count)

# TODO: how to train Unet 3d with sparce annotations ?
# masks_all = np.array(list(masks_all.values()))
# print("masks_all.shape", masks_all.shape)

masks_all2 = np.zeros(V_SHAPE, dtype=np.int16)
for k, v in masks_all.items():
    # (1126, 922)
    masks_all2[:,:,k] = np.fliplr(v)

print("masks_all2.shape", masks_all2.shape)

filename = 'our.h5'
with h5py.File(filename, 'w') as f:
    # 128, 128, 128 0..1
    f.create_dataset('label', data=masks_all2)    

del masks_all, masks_all2

import segyio
V = segyio.tools.cube(CUBE_FILE)

v_min = V.min()
v_max = V.max()
col_mean = np.nanmean(V, axis=(0,1))

V = (V - v_min)/(v_max - v_min)

# print("has_nan", np.any(np.isnan(V)))

inds = np.where(np.isnan(V))
V[inds] = np.take(col_mean, inds[1])

print("second stage")
v_min = V.min()
v_max = V.max()
V = (V - v_min)/(v_max - v_min)

print("V.shape", V.shape, V.dtype, np.min(V), np.max(V))

with h5py.File(filename, 'a') as f:
    # 128, 128, 128 0..1
    f.create_dataset('raw', data=V)

del V

# print("check")
# with h5py.File(filename, 'a') as f:
#     for k, v in f.items():
#         print(k, v.shape, v.dtype, np.min(v), np.max(v))
