import numpy as np
import pickle

# Load the NPZ file
npz_path = 'data/smplx_lockedhead_20230207/female/model.npz'  # Your NPZ file
data = dict(np.load(npz_path, allow_pickle=True))  # Load as dict

# Save as PKL
pkl_path = 'data/smplx_lockedhead_20230207/smplh/SMPLH_FEMALE.pkl'
with open(pkl_path, 'wb') as f:
    pickle.dump(data, f)

print(f'Converted {npz_path} to {pkl_path}')