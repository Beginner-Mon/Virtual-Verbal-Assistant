import numpy as np
from scipy.spatial.transform import Rotation as R
import os

path = "outputs/motion_64e6c614-0c7.npz"

data = np.load(path)

poses = data["poses"].copy()

correction = R.from_euler('y', 180, degrees=True)

for i in range(len(poses)):
    root = poses[i, :3]

    r = R.from_rotvec(root)
    new_r = correction * r

    poses[i, :3] = new_r.as_rotvec()

# keep all original arrays
save_dict = {k: data[k] for k in data.files}
save_dict["poses"] = poses

output_path = "outputs/motion_fixed.npz"

np.savez(output_path, **save_dict)

print("Saved file to:", os.path.abspath(output_path))