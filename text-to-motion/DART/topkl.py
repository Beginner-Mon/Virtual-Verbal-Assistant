import numpy as np
import pickle

# Load the NPZ file
npz_path = 'smplh/male/model.npz'  # Your NPZ file
data = dict(np.load(npz_path, allow_pickle=True))  # Load as dict

# Save as PKL
pkl_path = 'smplh/SMPLH_MALE.pkl'
with open(pkl_path, 'wb') as f:
    pickle.dump(data, f)

print(f'Converted {npz_path} to {pkl_path}')